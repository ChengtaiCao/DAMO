"""
    Implementation of Model
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from fightingcv_attention.attention.CBAM import CBAMBlock
from model.Backbone import resnet50
from model.SubModel import DeepLabHead, DeeplabNeck
from model.transformer import build_transformer


class IntermediateLayerGetter(nn.ModuleDict):
    """ obtain intermediate layer """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class ChannelAdjust(nn.Module):
    """ Channel Adjustment Layer """
    def __init__(self, input_channels=25, output_channels=768):
        super(ChannelAdjust, self).__init__()
        self.adjust = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adjust(x)


class ActionClassifier(nn.Module):
    """ Action Classification Head """
    def __init__(self, input_dim=768, output_dim=4):
        super(ActionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, representation):
        return self.fc(representation)


class ReasonClassifier(nn.Module):
    """ Reason Classification Head """
    def __init__(self, input_dim=768, output_dim=21):
        super(ReasonClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, representation):
        return self.fc(representation)


class Embedding_adjust(nn.Module):
    """ Reason Classification Head """
    def __init__(self, input_dim=768, output_dim=768):
        super(Embedding_adjust, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, representation):
        return self.fc(representation)


class FinalModel(nn.Module):
    """ Final whole model """
    def __init__(self, backbone, cbam, assp, neck, channel_adjust, embedding_adjust, transfomer,
                 action_classifier, reason_classifier):
        super(FinalModel, self).__init__()

        self.backbone = backbone
        self.cbam = cbam
        self.classifier = assp
        self.neck = neck
        self.channel_adjust = channel_adjust
        self.embedding_adjust = embedding_adjust
        self.action_classifier = action_classifier
        self.reason_classifier = reason_classifier
        self.transformer = transfomer

    def count_parameters(self):
        num_reason_params = sum(p.numel() for p in self.action_classifier.parameters() if p.requires_grad)
        num_action_params = sum(p.numel() for p in self.reason_classifier.parameters() if p.requires_grad)
        num_transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        print(f"reason parameters: {num_reason_params}")
        print(f"action parameters: {num_action_params}")
        print(f"transformer parameters: {num_transformer_params}")

    def get_action_output(self, representation):
        """Obtain the action prediction from feature representation"""
        return self.action_classifier(representation)  # [Batch_size, 4]

    def get_reason_output(self, representation):
        """Obtain the reason prediction from feature representation"""       
        return self.reason_classifier(representation)  # [Batch_size, 21]

    def get_representation(self, x, label_embedding):
        """Get the shared feature representation"""  
        # backbone
        features = self.backbone(x)
        x = features["out"]  # [Batch_size, 2048, 90, 160]

        # cbam
        x = self.cbam(x)  # [Batch_size, 2048, 90, 160]

        # classifier
        x = self.classifier(x)  # [Batch_size, 256, 90, 160]

        # neck
        x = self.neck(x)  # [Batch_size, 25, 18, 32]

        # Adjust the number of channel to 768
        x = self.channel_adjust(x)  # [Batch_size, 768, 18, 32]

        if x.shape[1] != 768:
            label_embedding = self.embedding_adjust(label_embedding)

        hs = self.transformer(x, label_embedding, None)[0]
        hs = hs.squeeze(0)
        hs = hs.mean(dim=1)  # [Batch_size, 768]

        return hs

    def forward(self, x, label_embedding):
        # Get shared feature representation
        hs = self.get_representation(x, label_embedding)

        # Get the output of two tasks
        out1 = self.get_action_output(hs)  # [batch_size, 4]
        out2 = self.get_reason_output(hs)  # [batch_size, 21]

        return [out1, out2]


def get_model(config, args, pretrained=True):
    """ Get Whole Model """  
    out_inplanes = config["out_inplanes"]
    action_class = config["action_class"]
    reason_class = config["reason_class"]
    total_class = action_class + reason_class

    # Backbone
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # CBAM
    cbam = CBAMBlock(channel=out_inplanes, reduction=16, kernel_size=7)

    # ASPP
    assp = DeepLabHead(out_inplanes)

    # Head
    neck = DeeplabNeck(out_channels=total_class)

    # channel_adjust
    channel_adjust = ChannelAdjust(input_channels=25, output_channels=args.adjust_channel)

    # labelEmbedding_adjust
    embedding_adjust = Embedding_adjust(input_dim=768, output_dim=args.adjust_channel)

    # transformer
    transfomer = build_transformer(args)

    # action_classifier
    action_classifier = ActionClassifier(input_dim=args.adjust_channel, output_dim=4)

    # #reason_classifier
    reason_classifier = ReasonClassifier(input_dim=args.adjust_channel, output_dim=21)

    model = FinalModel(backbone, cbam, assp, neck, channel_adjust, embedding_adjust, transfomer,
                       action_classifier, reason_classifier)

    if pretrained:
        weights_dict = torch.load("./weight/bdd10k_resnet50_1.pth", map_location='cpu')
        weights_dict = weights_dict["model"]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys1: ", missing_keys)
            print("unexpected_keys1: ", unexpected_keys)

    return model


if __name__ == "__main__":
    net = get_model()
    input_tensor = torch.FloatTensor(2, 3, 720, 1280)
    output = net(input_tensor)
    print(output)
