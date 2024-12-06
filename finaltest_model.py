import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from train_model import DṛṣṭiGyanCNN  

# Set the number of threads for CPU
torch.set_num_threads(4)  # Adjust based on CPU cores

# Device setup
device = torch.device("cpu")

# Load the trained model
model = DṛṣṭiGyanCNN()
model.load_state_dict(torch.load('./DṛṣṭiGyan v1.pth', map_location=device))
model.to(device) 
model.eval()

# Define the transform for the test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-100 test set
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

# Evaluate the model
correct = 0
total = 0
confusion_matrix = np.zeros((100, 100))  # CIFAR-100 has 100 classes

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for true, pred in zip(labels, predicted):
            confusion_matrix[true.item(), pred.item()] += 1

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Function to plot the images and their predictions
def plot_errors(images, labels, predictions, num_errors=5):
    incorrect_indices = np.where(labels != predictions)[0]
    if len(incorrect_indices) < num_errors:
        num_errors = len(incorrect_indices)

    plt.figure(figsize=(15, 3))
    for i, index in enumerate(incorrect_indices[:num_errors]):
        image = images[index].cpu().numpy().transpose((1, 2, 0))
        label = labels[index].item()
        prediction = predictions[index].item()

        plt.subplot(1, num_errors, i + 1)
        plt.imshow(image)
        plt.title(f'True: {label}\nPred: {prediction}')
        plt.axis('off')
    plt.show()

# Collect images, labels, and predictions for error analysis
all_images = []
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_images.extend(images.cpu())
        all_labels.extend(labels.cpu())
        all_predictions.extend(predicted.cpu())

# Convert lists to tensors for easy processing
all_images = torch.stack(all_images)
all_labels = torch.tensor(all_labels)
all_predictions = torch.tensor(all_predictions)

# Plot some errors
plot_errors(all_images, all_labels, all_predictions, num_errors=5)

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Print some of the predicted and true labels
        if i == 0:  # Only print for the first batch to avoid clutter
            for img, label, pred in zip(images[:5], labels[:5], predicted[:5]):
                print(f"True Label: {label.item()}, Predicted: {pred.item()}")
                plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
                plt.show()
        break

