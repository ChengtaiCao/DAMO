# DAMO: Dual-Attention with Multi-Objective Optimization for Explainable Autonomous Driving

![](https://img.shields.io/badge/python-3.9-green)
![](https://img.shields.io/badge/torch-2.5.1-green)
![](https://img.shields.io/badge/cudatoolkit-12.4-green)

This repo provides a reference implementation of "DAMO: Dual-Attention with Multi-Objective Optimization for Explainable Autonomous Driving."

# Abstract
Deep learning has revolutionized autonomous driving; nevertheless, its inherent opacity impedes explainability, a crucial factor for public trust and regulatory compliance. Existing research in explainable autonomous driving adopts a multi-task paradigm that concurrently generates driving actions and their corresponding explanations (collectively referred to as categories). Predominant approaches employ a two-stage framework of extracting category-related features and modeling category correlations. While effective, these methods treat feature extraction and correlation modeling as separate processes, disregarding their potential synergistic interaction. Furthermore, their reliance on simplistic linear combinations of task-specific losses fails to achieve an optimal balance between action and explanation objectives. To address these limitations, we propose Dual-Attention with Multi-Objective optimization (DAMO). DAMO incorporates a dual-attention mechanism that alternates between cross-attention for category representation learning and self-attention for category correlation modeling, enabling mutual enhancement. Moreover, we develop a multi-objective optimization algorithm that ensures dynamic task balancing with theoretical guarantees of Pareto optimality. Extensive evaluations on two benchmarks demonstrate that DAMO outperforms ten state-of-the-art baselines and a large vision-language model, achieving performance improvements of up to 13.9% and enhanced generalization across diverse driving scenarios.

# Dataset
## BDD-OIA
BDD-OIA, a subset of [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning], contains 22,924 video frames, each annotated with 4 action decisions and 21 human-defined explanations. Head to [Explainable Object-induced Action Decision for Autonomous Vehicles] to download the dataset. Following [Explainable Object-induced Action Decision for Autonomous Vehicles], only the final frame of each video clip is used, leading to a training set of 16,082 images, a validation set of 2,270 images, and a test set of 4,572 images.
## PSI
PSI includes 11,902 keyframes, each annotated with 3 actions and explanations provided in natural language. Head to [PSI: A Pedestrian Behavior Dataset for Socially Intelligent Autonomous Car] to download the dataset. Following [Attention-Based Interrelation Modeling for Explainable Automated Driving], all samples are split into training, validation, and test sets with a ratio of 7/1/2.

# Environmental Settings
Our experiments are conducted on Ubuntu 22.04, a single NVIDIA GeForce RTX 4090 GPU, 128GB RAM, and Intel i9-14900K. DAMO is implemented by `Python 3.9`, `PyTorch 2.5.1`, and `Cuda 12.4`.

**Step 1**: Install [Anaconda]  

**Step 2**: Create a virtual environment and install the required packages
```shell
# create a new environment
conda create -n DAMO python=3.9

# activate environment
conda activate DAMO

# install Pytorch
pip install torch torchvision torchaudio

# install other required packages
pip install -r requirements.txt
```

# Usage
**Step 0**: Create some folder
```shell
mkdir bddoia log save_model weight
```

**Step 1**: Download datasets: Put BDD-OIA data in folder "bddoia."

**Step 2**: Download pre-trained weight from [NLE-DM: Natural-Language Explanations for Decision Making of Autonomous Driving Based on Semantic Scene Understanding] and put it in folder "weight."

**Step 3**: Generate sentence embeddings.
```shell
python LabelSemantic.py
```

**Step 4**: Train model.
```shell
python train_OIA.py
```

**Step 5**: Test model.
```shell
python prediction_OIA.py
```

# Default hyperparameter settings

Unless otherwise specified, we use the following default hyperparameter settings.

Param|Value|Param|Value
:---|---:|:---|---:
learning rate|0.001|batch_size|2
momentum|0.9|epoches|40
weight decay|0.0001|attention heads|4
attention dimension|768|attention layers|3
dropout|0.1|return_intermediate_dec|False

# Results
| | BDD-OIA [xu2020explainable] | | | | PSI [chen2021psi] | | | |
|----------------|---------|---------|----------|----------|-----------|-----------|----------|----------|
| | Act_mF1 | Act_oF1 | Exp_mF1 | Exp_oF1 | Act_mAcc | Act_oAcc | Exp_mF1 | Exp_oF1 |
| GPT-4V | 0.436 | 0.537 | 0.191 | 0.284 | 0.577 | 0.618 | 0.127 | 0.143 |
| CBM | 0.610 | 0.661 | 0.292 | 0.412 | 0.626 | 0.651 | 0.127 | 0.192 |
| OIA | 0.718 | 0.734 | 0.208 | 0.422 | 0.593 | 0.643 | 0.110 | 0.189 |
| NLE-DM | 0.723 | 0.733 | 0.312 | 0.517 | 0.732 | 0.747 | 0.209 | 0.274 |
| ABIM | 0.701 | 0.722 | 0.335 | 0.537 | 0.699 | 0.712 | 0.191 | 0.278 |
| InAction | 0.694 | 0.714 | 0.347 | 0.565 | 0.722 | 0.734 | 0.223 | 0.285 |
| F-Transformer | 0.703 | 0.735 | 0.353 | 0.538 | 0.736 | 0.743 | 0.268 | 0.303 |
| MM-XAD | 0.723 | 0.743 | 0.360 | 0.535 | 0.741 | 0.747 | 0.270 | 0.325 |
| EDL | 0.705 | 0.727 | 0.368 | 0.581 | 0.712 | 0.726 | 0.253 | 0.274 |
| NeSy | 0.703 | 0.721 | 0.347 | 0.481 | 0.716 | 0.730 | 0.281 | 0.334 |
| SGDCL | _0.733_ | _0.753_ | _0.386_ | _0.582_ | _0.764_ | _0.770_ | _0.309_ | _0.347_ |
| **DAMO** | **0.751±0.002** | **0.768±0.001** | **0.423±0.002** | **0.614±0.002** | **0.779±0.001** | **0.788±0.001** | **0.352±0.002** | **0.393±0.003** |