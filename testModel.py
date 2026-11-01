# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from VGG6 import VGG6
# import os
#
#
#
# def loadAndRunModel(model_path, image_folder_path):
#     # Import or define your VGG6 model class before running this code.
#     # from model import VGG6
#     print("Loading model... ", model_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # CIFAR-10 class names
#     class_names = [
#         "airplane", "automobile", "bird", "cat", "deer",
#         "dog", "frog", "horse", "ship", "truck"
#     ]
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. Create the model using the SAME architecture and activation
#     model = VGG6(activationFunction='GELU')
#
#     # 2. Load trained weights
#
#     state_dict = torch.load(
#         model_path,
#         map_location=device,
#         weights_only=True
#     )
#
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#
#     print("Trained model loaded successfully!")
#
#     # 3. Image preprocessing
#     # Use the SAME normalization as in your training pipeline.
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             (0.4914, 0.4822, 0.4465),
#             (0.2470, 0.2435, 0.2616)
#         )
#     ])
#
# # 4. Predict any image
# #     def predict_image(image_path):
#
#     # Supported image formats
#     image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
#
#     # Fetch all image names
#     image_names = [
#         filename
#         for filename in os.listdir(image_folder_path)
#         if filename.lower().endswith(image_extensions)
#     ]
#     for image_name in sorted(image_names):
#         image_path = os.path.join(image_folder_path, image_name)
#         #print(image_path)
#
#
#         image = Image.open(image_path).convert("RGB")
#
#         # Prepare image for the model
#         input_tensor = transform(image).unsqueeze(0).to(device)
#
#         # Make prediction
#         with torch.no_grad():
#             outputs = model(input_tensor)
#             probabilities = torch.softmax(outputs, dim=1)
#             confidence, predicted_index = probabilities.max(dim=1)
#
#         predicted_class = class_names[predicted_index.item()]
#         confidence_value = confidence.item() * 100
#
#         print(
#             f"  Image: {image_name} | "
#             f"Prediction: {predicted_class} | "
#             f"Confidence: {confidence_value:.2f}%"
#         )
#
#


import sys
import platform
from importlib import metadata

# pip package names used in your project
packages = ["torch", "torchvision", "wandb", "numpy"]

print("=" * 40)
print("ENVIRONMENT DETAILS")
print("=" * 40)
print("Python  :", sys.version.split()[0])
print("OS      :", platform.system(), platform.release())
print("Machine :", platform.machine())

print("\n" + "=" * 40)
print("DEPENDENCY VERSIONS")
print("=" * 40)
requirements = []
for pkg in packages:
    try:
        version = metadata.version(pkg)
        print(f"{pkg:<12}: {version}")
        requirements.append(f"{pkg}=={version}")
    except metadata.PackageNotFoundError:
        print(f"{pkg:<12}: NOT INSTALLED")

# GPU / CUDA details
try:
    import torch
    print("\n" + "=" * 40)
    print("GPU / CUDA DETAILS")
    print("=" * 40)
    print("CUDA available :", torch.cuda.is_available())
    print("CUDA version   :", torch.version.cuda)
    print("cuDNN version  :", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
    else:
        print("GPU            : CPU only")
except ImportError:
    pass

# Write requirements.txt automatically
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements) + "\n")
print("\nrequirements.txt created with:")
print("\n".join(requirements))