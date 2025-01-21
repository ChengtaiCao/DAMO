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
