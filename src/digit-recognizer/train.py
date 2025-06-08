import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset class with augmentation
class MNISTDataset(Dataset):
    def __init__(self, data, labels, is_train=True):
        self.data = torch.FloatTensor(data) / 255.0  # Normalize to [0, 1]
        self.labels = torch.LongTensor(labels)
        self.is_train = is_train
        
        # Define augmentations for training
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(15),  # Increased rotation
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),  # Increased translation
                    scale=(0.85, 1.15),  # Random scaling
                    shear=10  # Added shear
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.3),
                transforms.RandomApply([
                    transforms.RandomInvert(p=1.0)  # Random inversion
                ], p=0.1),
                transforms.RandomApply([
                    transforms.RandomAdjustSharpness(sharpness_factor=2)
                ], p=0.3),
                transforms.RandomApply([
                    transforms.RandomAutocontrast()
                ], p=0.3),
                transforms.RandomApply([
                    transforms.RandomEqualize()
                ], p=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Random erasing
            ])
        else:
            self.transform = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx].view(28, 28)
        if self.is_train and self.transform:
            img = self.transform(img)
        else:
            img = img.view(1, 28, 28)  # Add channel dimension
        return img, self.labels[idx]

# Enhanced CNN Model with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.in_channels = 1
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual blocks
        self.layer1 = self.make_layer(32, 32, 2, stride=1)
        self.layer2 = self.make_layer(32, 64, 2, stride=2)
        self.layer3 = self.make_layer(64, 128, 2, stride=2)
        
        # Global average pooling and fully connected
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10  # Increased patience
    patience_counter = 0
    min_delta = 0.001  # Minimum change in validation loss to be considered as improvement
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model with minimum delta check
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(f'New best model saved! (Validation Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return train_losses, val_losses

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('../datasets/digit-recognizer/train.csv')
    
    # Split features and labels
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = MNISTDataset(X_train, y_train, is_train=True)
    val_dataset = MNISTDataset(X_val, y_val, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model = DigitRecognizer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Calculate total steps for OneCycleLR with longer warmup
    total_steps = len(train_loader) * 25  # 100 epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        total_steps=total_steps,
        pct_start=0.4,  # Longer warmup
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = initial_lr/1e4
    )
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device=device)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main()
