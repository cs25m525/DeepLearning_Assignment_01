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
import numpy as np
import os

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

#To train model based on config_params
def train(config_params, mean, std):
    SEED = 22
    print("Training Called....")

    # -------------------- TRANSFORMS --------------------

    # Training transform with data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation and testing transform without augmentation
    validationTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    with wandb.init(
        mode="offline",
        config=config_params,
        project='Vgg6OnCifar10New'
    ):

        config = wandb.config

        #--Seed Configurations
        random.seed(SEED)
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)


        batch_size = config.batch_size
        momentumRate = getattr(config, "momentum", 0.9)
        learningRate = config.lr

        wandb.run.name = (
            f"act_{config.activation}_bs_{batch_size}_"
            f"opt_{config.optimizer}_lr_{learningRate}_"
            f"epoch_{config.epochs}"
        )

        #Split into training and Validation dataset--------------------
        train_data = torchvision.datasets.CIFAR10(
            root=dataFilePathLocation,
            train=True,
            transform=transform,
            download=True
        )
        validation_data = torchvision.datasets.CIFAR10(
            root=dataFilePathLocation,
            train=True,
            transform=validationTransform,
            download=True
        )

        # Fixed 45,000 / 5,000 split
        generator = torch.Generator().manual_seed(SEED)

        indices = torch.randperm(
            len(train_data),
            generator=generator
        ).tolist()

        train_indices = indices[:45000]
        validation_indices = indices[45000:]

        train_data = torch.utils.data.Subset(
            train_data,
            train_indices
        )

        validation_data = torch.utils.data.Subset(
            validation_data,
            validation_indices
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        print("Training images: ", len(train_data))
        print("Validation images: ", len(validation_data))

        # Select Device CPU or GPU

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {device}")

        if device.type == "cuda":
            print("GPU: ", torch.cuda.get_device_name(0))

        vgg6NeuralNetwork = VGG6(config.activation).to(device)

        #Select optimizer
        if config.optimizer == "Nesterov-SGD":
            optimizer = optim.SGD(
                vgg6NeuralNetwork.parameters(),
                lr=learningRate,
                momentum=momentumRate,
                nesterov=True
            )

        elif config.optimizer == "SGD":
            optimizer = optim.SGD(
                vgg6NeuralNetwork.parameters(),
                lr=learningRate,
                momentum=momentumRate
            )

        else:
            opt_class = getattr(optim, config.optimizer)
            optimizer = opt_class(
                vgg6NeuralNetwork.parameters(),
                lr=learningRate
            )

        print("Activation Function: ", config.activation," Optimization: ",config.optimizer)

        loss_function = nn.CrossEntropyLoss()

        # Top mdoel init -----------------

        best_val_acc = 0.0
        best_epoch = 0

        os.makedirs("Models", exist_ok=True)

        best_model_path = (f"Models/vgg6_cifar10_act_{config.activation}_"f"opt_{config.optimizer}_best.pth")

        #Train------------
        for epoch in range(config.epochs):
            vgg6NeuralNetwork.train()
            print(f"\nTraining epoch {epoch + 1}/{config.epochs} started...")

            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = vgg6NeuralNetwork(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(train_loader)
            train_acc = (100.0 * train_correct / train_total)

            # VALIDATION
            print("Validation Started...")
            vgg6NeuralNetwork.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = vgg6NeuralNetwork(inputs)
                    loss = loss_function(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            val_loss = (running_val_loss / len(validation_loader))

            val_acc = (100.0 * val_correct / val_total)

            # print result

            print(
                f"Epoch [{epoch + 1}/{config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            # -------------------- WANDB LOGGING --------------------

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_accuracy": val_acc
            })

            # -------------------- SAVE BEST MODEL --------------------

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1

                torch.save(vgg6NeuralNetwork.state_dict(),best_model_path)
                print(f"Best model saved at epoch {best_epoch}")
                print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        print("\nTraining Completed.")
        print("Best Epoch: ", best_epoch)
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        # Load best checkpoint
        vgg6NeuralNetwork.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
                weights_only=True
            )
        )

        # Test dataset: original 10,000 CIFAR-10 test images
        test_data = torchvision.datasets.CIFAR10(
            root=dataFilePathLocation,
            train=False,
            transform=validationTransform,
            download=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        vgg6NeuralNetwork.eval()

        running_test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = vgg6NeuralNetwork(inputs)
                loss = loss_function(outputs, labels)
                running_test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss = running_test_loss / len(test_loader)

        test_acc = (100.0 * test_correct / test_total)

        print("\n========== FINAL TEST RESULTS ==========")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print("========================================")

        # TEST WANDB LOGGING

        wandb.log({
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        print("\nTraining and Testing End.")
        print("Best model saved at: ", best_model_path)
        return {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_model_path": best_model_path
        }

#To get model config data from user input
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


#To train model based on config_params
def runForBestValidationAccurcytrain(batch_size, mean, std):
    SEED = 22
    print("Best Validaiton Accuracy Model Function Called....")

    # -------------------- TRANSFORMS --------------------
    # Validation and testing transform without augmentation
    validationTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    #Add transform
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    validation_data = torchvision.datasets.CIFAR10(
        root=dataFilePathLocation,
        train=True,
        transform=validationTransform,
        download=True
    )

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(validation_data),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]
    validation_indices = indices[45000:]

    validation_data = torch.utils.data.Subset(
        validation_data,
        validation_indices
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("Validation images: ", len(validation_data))
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print("GPU: ", torch.cuda.get_device_name(0))

    vgg6NeuralNetwork = VGG6("GELU").to(device)

    print("Activation Function: GELU", " Optimization: NAdam")

    loss_function = nn.CrossEntropyLoss()

    best_model_path = ("Models/vgg6_cifar10_act_GELU_opt_NAdam_best.pth")

    # Load best checkpoint
    vgg6NeuralNetwork.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )
    vgg6NeuralNetwork.eval()
    running_test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vgg6NeuralNetwork(inputs)
            loss = loss_function(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss = running_test_loss / len(validation_loader)

    test_acc = (100.0 * test_correct / test_total)

    print("\n===== FINAL VALIDATION RESULTS =======")
    print(f"Validation Loss: {test_loss:.4f}")
    print(f"Validation Accuracy: {test_acc:.2f}%")
    print("========================================")

if __name__ == "__main__":
    print('|---------------------------------------------------------------------------|')
    print('|-------------------------Deep Learning Assignment 1------------------------|')
    print('|-----------------------------Prateek Maheshwari----------------------------|')
    print('|----------------------------------CS25M525---------------------------------|')
    print('|---------------------------------------------------------------------------|')

    dataFilePathLocation = '.'
    # Calculate CIFAR-10 normalization values
    MEAN, STD = calculateMeanStd()
    print("Required: VGG6.py,"
          " testModel.py,"
          " Images(folder to store images and few sample images are already present),"
          " Models(to store models),"
          " cifar-10-batces-py (for cifar10 dataset)")
    print("")

    while True:
        print("1. Evaluate Model For Best Validation Accuracy")
        print("2. Test either for other images Train Model")
        print("3. Train Validate and Test Model your choice")
        print("4. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            print("Starting for Best Validation Accuracy")
            runForBestValidationAccurcytrain(128, MEAN, STD)
            print("Best Validation Accuracy Done.")

        elif choice == 2:
            print('To try GELU NAdam Model put all images in \"Images\" folder (few sample images are already present) ')
            image_path = input("and press any key to continue..")
            images_folder = r"Images"
            loadAndRunModel("Models/vgg6_cifar10_act_GELU_opt_NAdam_best.pth", images_folder)

        elif choice == 3:
            # Run the configuration menu
            config_params = get_model_config()
            print('Selected Model Configuration: ',end="")
            for key in ["activation", "optimizer", "batch_size", "lr", "epochs"]:
                print(f"{key}: {config_params[key]} ", end="")
            print("Started.")
            train(config_params, MEAN, STD)
            print("Done.")

        elif choice == 4:
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")

    exit()
    # ##################
    #
    #
    #
    #For Question 2c Learning Rate Variation and its plot generation and model training
    # lr_List = [0.001, 0.005, 0.01]
    # for e in lr_List:
    #     config_params = {
    #         "activation": 'SiLU',
    #         "optimizer": 'NAdam',
    #         "lr": e,
    #         "epochs": 20,
    #         "momentum": 0.9,
    #         "seed": 22,
    #         "batch_size": 128
    #     }
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
    #     train(config_params, MEAN, STD)
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    #For Question 2c Epoch Variation and its plot generation and model training
    # epoch_List = [10,20,25,30,50]
    # for e in epoch_List:
    #     config_params = {
    #         "activation": 'SiLU',
    #         "optimizer": 'NAdam',
    #         "lr": 0.001,
    #         "epochs": e,
    #         "momentum": 0.9,
    #         "seed": 22,
    #         "batch_size": 128
    #     }
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
    #     train(config_params, MEAN, STD)
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    #For Question 2c Batch Size Variation and its plot generation and model training
    # bs_List = [32, 64, 128]
    # for e in bs_List:
    #     config_params = {
    #         "activation": 'SiLU',
    #         "optimizer": 'NAdam',
    #         "lr": 0.001,
    #         "epochs": 20,
    #         "momentum": 0.9,
    #         "seed": 22,
    #         "batch_size": e
    #     }
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Starting")
    #     train(config_params, MEAN, STD)
    #     print(
    #         f"Epochs {config_params['epochs']}, Batch size {config_params['batch_size']}, Learning rate {config_params['lr']} Done")


    # #For Question 2 a and b plot generation and model training
    # #for activations and optimizers combination
    # #For offline code executing
    # # Define local grid search parameters
    # activations = ['GELU']
    # optimizers = ['Adam']
    #
    # # Generate all combinations locally without W&B server
    # grid = list(itertools.product(activations, optimizers))
    #
    # for act, opt in grid:
    #     config_params = {
    #         "activation": act,
    #         "optimizer": opt,
    #         "lr": 0.001,
    #         "epochs": 50,
    #         "momentum": 0.9,
    #         "batch_size": 128
    #     }
    #     print(f"\n---> Starting Offline Run: Activation={act}, Optimizer={opt}")
    #     train(config_params, MEAN, STD)
        #trainAll(config_params)