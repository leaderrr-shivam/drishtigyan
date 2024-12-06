import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_and_preprocess_cifar100(batch_size=32, val_split=0.2):
    # Define data augmentation and preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match model input size
        transforms.RandomHorizontalFlip(),  # Horizontal flip for data augmentation
        transforms.RandomCrop(224, padding=4),  # Random crop with padding
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load CIFAR-100 dataset with augmentation
    train_dataset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

    # Debug: print dataset length and sample data
    print("Training dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("First training sample:", train_dataset[0])

    # Splitting the training dataset into training and validation sets (80%-20% split)
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Debug: print sizes of the split datasets
    print("Training set size after split:", len(train_dataset))
    print("Validation set size after split:", len(val_dataset))

    # Create DataLoaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("CIFAR-100 dataset loaded, augmented, and split successfully!")

    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_and_preprocess_cifar100(batch_size=32)
