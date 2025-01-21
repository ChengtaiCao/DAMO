""" 
    Implementation of Training Utils
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
from utils.MetricLogger import MetricLogger, SmoothedValue
from torch.autograd import Variable


def criterion(output1, output2, target1, target2, device):
    """ Loss Function """
    losses = {}
    class_weights1 = [1, 1, 2, 2]
    w1 = torch.FloatTensor(class_weights1).to(device)
    losses['action'] = torch.nn.functional.binary_cross_entropy_with_logits(output1, target1, weight=w1)
    losses['reason'] = torch.nn.functional.binary_cross_entropy_with_logits(output2, target2)
    return 0.5 * losses['action'] + 0.75 * losses['reason']


def create_lr_scheduler(optimizer,
                        num_step,
                        epochs,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3
                        ):
    """ Learning Rate Update Strategy """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def List2Arr(List):
    """ Convert List to Array """
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])

    return np.vstack((Arr1, Arr2))


def train_one_epoch(model,
                    optimizer,
                    dataloader_train,
                    label_embedding,
                    device,
                    epoch,
                    lr_scheduler,
                    print_freq,
                    scaler
                    ):
    """ Training One Epoch"""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target, _ in metric_logger.log_every(dataloader_train, print_freq, header):
        image = image.to(device)
        target[0] = target[0].to(device)
        target[1] = target[1].to(device)

        tasks = ['action', 'reason']
        grads = {t: [] for t in tasks}
        loss_data = {t: 0 for t in tasks}
        scale = {t: 0 for t in tasks}
        class_weights1 = [1, 1, 2, 2]
        w1 = torch.FloatTensor(class_weights1).to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            rep = model.get_representation(image, label_embedding)

        rep_variable = Variable(rep.data.clone(), requires_grad=True)

        # Compute gradients for each task
        for t in tasks:
            optimizer.zero_grad()
            if t == 'action':
                out_t = model.get_action_output(rep_variable)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out_t, target[0], weight=w1)
            else:  # reason
                out_t = model.get_reason_output(rep_variable)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out_t, target[1])

            loss_data[t] = loss.item()
            loss.backward()
            grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
            rep_variable.grad.data.zero_()

        # Normalize gradients
        gn = gradient_normalizers(grads, loss_data, 'l2')
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Find minimum norm solution
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])

        # Scaled back-propagation
        optimizer.zero_grad()
        output = model(image, label_embedding)
        loss_action = torch.nn.functional.binary_cross_entropy_with_logits(
            output[0], target[0], weight=w1)
        loss_reason = torch.nn.functional.binary_cross_entropy_with_logits(
            output[1], target[1])

        loss = scale['action'] * loss_action + scale['reason'] * loss_reason

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, label_embedding, device):
    """ Get Evaluation Results """
    model.eval()
    with torch.no_grad():
        val_loss = 0
        Target_ActionArr = []
        Pre_ActionArr = []
        Target_ReasonArr = []
        Pre_ReasonArr = []

        for image, target, _ in tqdm(data_loader):
            image = image.to(device)
            output = model(image, label_embedding)
            output1 = output[0]
            output2 = output[1]
            # print("-----output1", output1)
            # output3 = output[2]
            target[0] = target[0].to(device)
            target[1] = target[1].to(device)
            # loss
            loss = criterion(output1, output2, target[0], target[1], device)
            val_loss += loss

            # calculate the F1 score
            predict_action = torch.sigmoid(output1) > 0.5
            preds_action = predict_action.cpu().numpy()
            predict_reason = torch.sigmoid(output2) > 0.5
            preds_reason = predict_reason.cpu().numpy()

            a_targets = target[0].cpu().numpy()
            e_targets = target[1].cpu().numpy()

            Target_ActionArr.append(a_targets)
            Pre_ActionArr.append(preds_action)
            Target_ReasonArr.append(e_targets)
            Pre_ReasonArr.append(preds_reason)

        Target_ActionArr = List2Arr(Target_ActionArr)
        Pre_ActionArr = List2Arr(Pre_ActionArr)
        Target_ReasonArr = List2Arr(Target_ReasonArr)
        Pre_ReasonArr = List2Arr(Pre_ReasonArr)

        # action
        Action_F1_overall = f1_score(Target_ActionArr, Pre_ActionArr, average='samples')
        Action_Per_action = f1_score(Target_ActionArr, Pre_ActionArr, average=None)
        Action_F1_mean = np.mean(Action_Per_action)

        # reason
        Reason_F1_overall = f1_score(Target_ReasonArr, Pre_ReasonArr, average='samples')
        Reason_Per_action = f1_score(Target_ReasonArr, Pre_ReasonArr, average=None)
        Reason_F1_mean = np.mean(Reason_Per_action)

    res_dict = {}
    res_dict["Val_loss"] = val_loss.item()
    res_dict["Action_overall"] = np.mean(Action_F1_overall)
    res_dict["Reason_overall"] = np.mean(Reason_F1_overall)
    res_dict["F1_action"] = Action_Per_action
    res_dict["F1_action_average"] = Action_F1_mean
    res_dict["F1_reason"] = Reason_Per_action
    res_dict["F1_reason_average"] = Reason_F1_mean

    return res_dict
