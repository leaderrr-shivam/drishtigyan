import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from train_model import DṛṣṭiGyanCNN

# Set the number of threads for CPU
torch.set_num_threads(4)  # Adjust based on CPU cores

# Device setup
device = torch.device("cpu")

# Define test dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model evaluation for each fold
def evaluate_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

# Plot confusion matrix
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Plot misclassified images
def plot_misclassified_images(images, labels, preds, class_names):
    misclassified_indices = [i for i in range(len(labels)) if labels[i] != preds[i]]
    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(misclassified_indices[:16]):  # Show only the first 16 misclassified images
        image = images[idx].permute(1, 2, 0).numpy()
        plt.subplot(4, 4, i + 1)
        plt.imshow((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1))  # De-normalize
        plt.title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")
        plt.axis("off")

    plt.suptitle("Misclassified Images", fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize accuracy across folds
def plot_accuracies(fold_accuracies):
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(fold_accuracies) + 1), fold_accuracies, color='skyblue')
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Folds")
    plt.xticks(range(1, len(fold_accuracies) + 1))
    plt.ylim(0, 1)
    plt.show()

# Main function to evaluate and perform error analysis
def find_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DṛṣṭiGyanCNN(num_classes=100).to(device)

    fold_paths = [
        "./DṛṣṭiGyan_best_model_fold_1.pth",
        "./DṛṣṭiGyan_best_model_fold_2.pth",
        "./DṛṣṭiGyan_best_model_fold_3.pth",
        "./DṛṣṭiGyan_best_model_fold_4.pth",
        "./DṛṣṭiGyan_best_model_fold_5.pth"
    ]

    class_names = test_dataset.classes
    fold_accuracies = []
    best_accuracy = 0
    best_fold = None
    best_model_path = None

    for fold_idx, path in enumerate(fold_paths, start=1):
        accuracy, preds, labels = evaluate_model(path, model, device)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold_idx}: Accuracy = {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_fold = fold_idx
            best_model_path = path

    # Visualize fold accuracies
    plot_accuracies(fold_accuracies)

    # Load and analyze the best model
    accuracy, preds, labels = evaluate_model(best_model_path, model, device)
    print(f"\nBest Fold: {best_fold} with Accuracy = {best_accuracy * 100:.2f}%")
    print(f"Saving Best Model from Fold {best_fold} as 'DṛṣṭiGyan.pth'")
    torch.save(torch.load(best_model_path, map_location=device), "./DṛṣṭiGyan.pth")

    print("\nClassification Report for Best Model:")
    report = classification_report(labels, preds, target_names=class_names)
    print(report)

    # Confusion matrix
    plot_confusion_matrix(labels, preds, class_names)

    # Misclassified images
    images, _ = next(iter(test_loader))
    plot_misclassified_images(images, labels, preds, class_names)

if __name__ == "__main__":
    find_best_model()
