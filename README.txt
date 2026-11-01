****************************************VGG6 CIFAR-10***************************************************************
|                         |
|Name:Prateek Maheshwari  |
|Roll No: CS25M525        |
|_________________________|

1. Project Description
This project implements a VGG6 convolutional neural network (CNN) to classify images from the CIFAR-10 dataset.
Experiments are conducted using different activation functions, optimizers, learning rates, batch sizes, and numbers of epochs.

2. Environment Details
    Operating System:           Windows 10/11 (64-bit)
    Programming Language:       Python
    Deep Learning Framework:    PyTorch
    Dataset:                    CIFAR-10
    Hardware:                   CPU and GPU(if available)

3. Dependencies
    The project requires the following Python packages are torch, torchvision, numpy, Pillow, os  and wandb.

4. Environment Setup
    4.1 Create and activate a virtual environment by command: python -m venv .venv
    4.2 Upgrade pip by command: python -m pip install --upgrade pip
    4.3 Install the compatible PyTorch and torchvision versions required by the project.

5. Random Seed Configuration is SEED = 22

6. Running the Project by command:
    Required: main.py, VGG6.py, testModel.py, Images(folder to store images), Models(to store models), cifar-10-batces-py (for cifar10 dataset)
    To run: python main.py
    Then select appropriate choice of operation to be executed.
    Choice 1: Evaluate Model For Best Validation Accuracy
    Choice 2: To try GELU NAdam Model put images inside Images folder(some files are already present) at same location of main.py and then continue.
    Choice 3: Train Validate and Test Model your choice
    Choice 4: Exit

7. Dataset: cifar-10-batches-py dataset to be present at same location as of main.py
