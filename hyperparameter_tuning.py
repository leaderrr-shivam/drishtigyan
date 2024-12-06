import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import optuna
import os
from train_model import DṛṣṭiGyanCNN  # Import the model architecture from your custom module
from tqdm import tqdm  # Import tqdm for progress bars

# Set the number of threads for CPU
torch.set_num_threads(4)  # Adjust based on CPU cores

# Device setup
device = torch.device("cpu")

# Load the pre-trained model
model_path = './DṛṣṭiGyan.pth'
model = DṛṣṭiGyanCNN()  # Replace with your model class
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Define a function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # Adding tqdm for the training loop
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    optimizer_choice = trial.suggest_categorical('optimizer', ['AdamW'])
    
    # Update the data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=2)  #Less epochs for quick training

    # Evaluate on the validation set
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Create an Optuna study and optimize with tqdm progress
study = optuna.create_study(direction='maximize')
with tqdm(total=10, desc='Trials') as pbar:
    def callback(study, trial):
        pbar.update(1)
    study.optimize(objective, n_trials=10, callbacks=[callback])

# Save the best model
best_trial = study.best_trial
print(f"Best Trial:")
print(f"  Value (accuracy): {best_trial.value:.4f}")
print(f"  Parameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save the optimized model
model_path_v1 = './DṛṣṭiGyan v1.pth'
torch.save(model.state_dict(), model_path_v1)
print(f"Model saved as {model_path_v1}")
