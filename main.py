# -*- coding: utf-8 -*-
"""
Deep Learning Assignment 1
Prateek Maheshwari
CS25M525
"""
import torch
import torch.nn as nn
import torch.nn.functional as functonal
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import random
import itertools
from VGG6 import VGG6
from testModel import loadAndRunModel

# Function to calculate Mean and Std for normalization of data
def calculateMeanStd():
    for_mean_train_data = torchvision.datasets.CIFAR10(root=dataFilePathLocation, train=True,
                                                       transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(for_mean_train_data, batch_size=32, shuffle=False)
    #
    # 2. Calculate Mean and Std
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in train_loader:
        # images shape: [batch_size, 3, 32, 32]
        batch_size = images.size(0)

        # Average across batch (0), height (2), width (3)
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size
        total_images += batch_size

    mean /= total_images
    std /= total_images

    print("CIFAR-10 Mean:", [round(x, 4) for x in mean.tolist()])
    print("CIFAR-10 Std: ", [round(x, 4) for x in std.tolist()])

    return mean.tolist(), std.tolist()


# Function to train different models based on config
def test():
    mean, std = calculateMeanStd()
    # Create Test Transformer as we only have to normalize the data and no Data Augumentation is required
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)  # mean and standard deviation per channel layer
    ])
    test_data = torchvision.datasets.CIFAR10(root=dataFilePathLocation, train=False, transform=testTransform,
                                             download=True)
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)


def train(config_params):
    mean, std = calculateMeanStd()
    print("Traning Called....")
    # Training Transform with
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)  # mean and standard deviation per channel layer
    ])

    with wandb.init(mode="offline", config=config_params, project='Vgg6OnCifar10'):
        config = wandb.config
        random.seed(22)
        batch_size = config.batch_size
        momentumRate = getattr(config, "momentum", 0.9)
        learningRate = config.lr

        wandb.run.name = (
            f"act_{config.activation}_bs_{batch_size}_opt_{config.optimizer}_lr_{learningRate}_epoch_{config.epochs+1}"
        )

        train_data = torchvision.datasets.CIFAR10(root=dataFilePathLocation, train=True, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

        image, label = train_data[0]
        #print("Shape: ", image.shape)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch_directml.device()
        print(f"Using device: {device}")
        vgg6NeuralNetwork = VGG6(config.activation).to(device)  # For Sigmoid

        if config.optimizer == "Nesterov-SGD":
            optimizer = optim.SGD(vgg6NeuralNetwork.parameters(), lr=learningRate, momentum=momentumRate,nesterov=True)

        elif config.optimizer == "SGD":
            optimizer = optim.SGD(vgg6NeuralNetwork.parameters(), lr=learningRate, momentum=momentumRate )

        else:
            opt_class = getattr(optim, config.optimizer)
            optimizer = opt_class(vgg6NeuralNetwork.parameters(), lr=learningRate)

        print("Activation Function: ", config.activation, " Optimization: ", config.optimizer)

        loss_function = nn.CrossEntropyLoss()

        for epoch in range(config.epochs):
            print(f'    Training epoch {epoch + 1} started.....')
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for i2, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = vgg6NeuralNetwork(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            print(f'    Training epoch {epoch + 1} Done.....')

            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            print(f'      Loss: {running_loss / len(train_loader):.4f}')
            wandb.log({'epoch': epoch, 'loss': running_loss / len(train_loader)})

            print("Testing Started... ")
            #Testing Phase
            # Create Test Transformer as we only have to normalize the data and no Data Augumentation is required
            testTransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)  # mean and standard deviation per channel layer
            ])
            test_data = torchvision.datasets.CIFAR10(root=dataFilePathLocation, train=False, transform=testTransform,
                                                     download=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

            vgg6NeuralNetwork.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = vgg6NeuralNetwork(inputs)
                    loss = loss_function(outputs, labels)

                    running_val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = running_val_loss / len(test_loader)
            val_acc = 100.0 * val_correct / val_total

            # ==================== W&B LOGGING ====================
            print(
                f"Epoch [{epoch + 1}/{config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            # Log all four metrics to W&B automatically
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_accuracy": val_acc,  # Alias for sweep metric matching Q1/Q4
                }
            )

            print("Training and Testing End.")
        print('Saving Model ',f"vgg6_cifar10_act_{config.activation}_opt_{config.optimizer}_epoch_{epoch}_bs_{batch_size}_lr_{learningRate}.pth")
        torch.save(vgg6NeuralNetwork.state_dict(),
                   f"vgg6_cifar10_act_{config.activation}_opt_{config.optimizer}_epoch_{epoch}.pth")


def get_model_config():
    # Available configurations
    activations = ['Relu', 'Sigmoid', 'Tanh', 'SiLU', 'GELU']
    optimizers = ['SGD', 'Nesterov-SGD', 'Adam', 'Adagrad', 'NAdam', 'RMSprop']

    # 1. Select activation function
    print("\nSelect Activation Function:")
    for index, activation in enumerate(activations, start=1):
        print(f"{index}- {activation} ", end="")

    while True:
        try:
            choice = int(input("\n  Enter activation choice (1-5): "))
            if 1 <= choice <= len(activations):
                activation = activations[choice - 1]
                break
            print("Invalid choice. Please select between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

    # 2. Select optimizer
    print("\nSelect Optimizer:")
    for index, optimizer in enumerate(optimizers, start=1):
        print(f"{index}- {optimizer} ", end="")

    while True:
        try:
            choice = int(input("\n  Enter optimizer choice (1-6): "))
            if 1 <= choice <= len(optimizers):
                optimizer = optimizers[choice - 1]
                break
            print("Invalid choice. Please select between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")

    # 3. Enter batch size
    while True:
        try:
            batch_size = int(input("\nEnter batch size: "))
            if batch_size > 0:
                break
            print("Batch size must be greater than zero.")
        except ValueError:
            print("Please enter a valid integer.")

    # 4. Enter learning rate
    while True:
        try:
            lr = float(input("Enter learning rate (0-1): "))
            if 0 < lr < 1:
                break
            print("Learning rate must be greater than 0 and less than 1.")
        except ValueError:
            print("Please enter a valid number.")

    # 5. Enter number of epochs
    while True:
        try:
            epochs = int(input("Enter number of epochs: "))
            if epochs > 0:
                break
            print("Epochs must be greater than zero.")
        except ValueError:
            print("Please enter a valid integer.")

    # Fixed configuration parameters
    config_params = {
        "activation": activation,
        "optimizer": optimizer,
        "lr": lr,
        "epochs": epochs,
        "momentum": 0.9,
        "seed": 22,
        "batch_size": batch_size,
    }

    return config_params


if __name__ == "__main__":
    print('|---------------------------------------------------------------------------|')
    print('|-------------------------Deep Learning Assignment 1------------------------|')
    print('|-----------------------------Prateek Maheshwari----------------------------|')
    print('|----------------------------------CS25M525---------------------------------|')
    print('|---------------------------------------------------------------------------|')

    dataFilePathLocation = '.'
    while True:
        print("1. Execute Best Model")
        print("2. Train Model")
        print("3. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            image_path = input("Put all images in \"Images\" folder and then press any key to continue..")
            images_folder = r"Images"
            loadAndRunModel("Models/vgg6_cifar10_act_SiLU_opt_NAdam_epoch_49.pth", images_folder)
            # now call best model with  input image

        elif choice == 2:
            # Run the configuration menu
            config_params = get_model_config()
            print('Selected Model Configuration: ',end="")
            for key in ["activation", "optimizer", "batch_size", "lr", "epochs"]:
                print(f"{key}: {config_params[key]} ", end="")
            print("Started.")
            train(config_params)
            print("Done.")

        elif choice == 3:
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")

    exit()
    ##################



    #For Question 2c Learning Rate Variation and its plot generation and model training
    lr_List = [0.001, 0.005, 0.01]
    for e in lr_List:
        config_params = {
            "activation": 'SiLU',
            "optimizer": 'NAdam',
            "lr": e,
            "epochs": 20,
            "momentum": 0.9,
            "seed": 22,
            "batch_size": 32
        }
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
        train(config_params)
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    #For Question 2c Epoch Variation and its plot generation and model training
    epoch_List = [10,20,25,30,50]
    for e in epoch_List:
        config_params = {
            "activation": 'SiLU',
            "optimizer": 'NAdam',
            "lr": e,
            "epochs": e,
            "momentum": 0.9,
            "seed": 22,
            "batch_size": 32
        }
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
        train(config_params)
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    #For Question 2c Batch Size Variation and its plot generation and model training
    bs_List = [32, 64, 128]
    for e in bs_List:
        config_params = {
            "activation": 'SiLU',
            "optimizer": 'NAdam',
            "lr": e,
            "epochs": e,
            "momentum": 0.9,
            "seed": 22,
            "batch_size": e
        }
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
        train(config_params)
        print(
            f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    #For Question 2 a and b plot generation and model training
    #for activations and optimizers combination
    #For offline code executing
    # Define local grid search parameters
    # activations = ['Relu', 'Sigmoid', 'Tanh', 'SiLU', 'GELU']
    # optimizers = ['SGD', 'Nesterov-SGD', 'Adam', 'Adagrad', 'NAdam', 'RMSprop']
    #
    # # Generate all combinations locally without W&B server
    # grid = list(itertools.product(activations, optimizers))
    #
    # for act, opt in grid:
    #     config_params = {
    #         "activation": act,
    #         "optimizer": opt,
    #         "lr": 0.001,
    #         "epochs": 1,
    #         "momentum": 0.9,
    #         "seed": 22
    #     }
    #     print(f"\n---> Starting Offline Run: Activation={act}, Optimizer={opt}")
    #     train(config_params)