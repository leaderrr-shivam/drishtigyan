import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import copy
from tqdm import tqdm  # For progress bars

# Set the number of threads for CPU
torch.set_num_threads(4)  # Adjust based on CPU cores

# Device setup
device = torch.device("cpu")

# Define data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-100 dataset
from torchvision.datasets import CIFAR100

dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)

# K-Fold Cross-Validation setup
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define a custom CNN model (DṛṣṭiGyan)
class DṛṣṭiGyanCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(DṛṣṭiGyanCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted for an output of (128, 4, 4)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()  # Add this line to define the ReLU activation

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Model setup
def create_model():
    # Initialize the custom CNN model with the name DṛṣṭiGyan
    model = DṛṣṭiGyanCNN(num_classes=100)
    return model.to(device)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, early_stopping_patience=5, fold=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_counter = 0
    min_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        # Training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} Training") as pbar:
            start_time = time.time()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_corrects += (preds == labels).sum().item()
                train_total += labels.size(0)

                pbar.update(1)

        train_acc = train_corrects / train_total
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} Validation") as pbar:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_corrects += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar.update(1)

        val_acc = val_corrects / val_total
        val_loss /= len(val_loader)

        # Calculate additional metrics
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")

        # Early stopping logic
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stopping_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            if fold is not None:
                torch.save(model.state_dict(), f'DṛṣṭiGyan_best_model_fold_{fold + 1}.pth')
                print(f"Model saved with validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        epoch_time = time.time() - start_time
        print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    model.load_state_dict(best_model_wts)
    return model

# Main function
def main():
    # Run k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nStarting fold {fold + 1}/{num_folds}")
        train_subset = data.Subset(dataset, train_idx)
        val_subset = data.Subset(dataset, val_idx)

        train_loader = data.DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = data.DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

        model = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        print(f"Training fold {fold + 1}...")
        model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, fold=fold)

    print("Training completed for all folds.")

# Ensures the script runs only when it is executed directly
if __name__ == '__main__':
    main()
