# Training Instructions for Deepfake Detection Models

## Overview
This document provides instructions for training the deep learning models used in the Hybrid Deepfake Detection System.

## Dataset Requirements

### 1. Image Dataset
- **Recommended Dataset**: FaceForensics++, Celeb-DF, DFDC
- **Structure**:
```
  dataset/
  ├── train/
  │   ├── real/
  │   └── fake/
  └── val/
      ├── real/
      └── fake/
```
- **Size**: Minimum 10,000 images per class
- **Format**: JPG/PNG, 224x224 or higher resolution

### 2. Video Dataset
- Uses same frame extraction as inference
- Videos stored in same structure as images
- Recommended: 1000+ videos per class

### 3. Audio Dataset
- **Recommended**: ASVspoof, WaveFake, FakeAVCeleb
- **Structure**: Same as image dataset
- **Format**: WAV/MP3, 16kHz+ sample rate
- **Duration**: 3-10 seconds per clip

## Training Scripts

### Image Model Training
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.image_model import ImageDeepfakeDetector
from config.model_config import Config

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD)
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Initialize model
model = ImageDeepfakeDetector(num_classes=Config.NUM_CLASSES, pretrained=True)
model = model.to(Config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Training loop
for epoch in range(Config.NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {accuracy:.2f}%')
    
    scheduler.step(val_loss)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, f'models/checkpoints/image_model_epoch_{epoch}.pth')

# Save final model
torch.save(model.state_dict(), 'models/checkpoints/image_model.pth')
```

### Audio Model Training
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.audio_model import AudioDeepfakeDetector
from utils.preprocessing import AudioPreprocessor
from config.model_config import Config
import os

class AudioDataset(Dataset):
    def __init__(self, root_dir, preprocessor):
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.samples = []
        
        # Load real audio
        real_dir = os.path.join(root_dir, 'real')
        for filename in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, filename), 0))
        
        # Load fake audio
        fake_dir = os.path.join(root_dir, 'fake')
        for filename in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, filename), 1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        mel_spec = self.preprocessor.audio_to_melspectrogram(audio_path)
        return mel_spec.squeeze(0), label

# Initialize preprocessor
preprocessor = AudioPreprocessor()

# Load datasets
train_dataset = AudioDataset('dataset/train', preprocessor)
val_dataset = AudioDataset('dataset/val', preprocessor)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

# Initialize model
model = AudioDeepfakeDetector(num_classes=Config.NUM_CLASSES)
model = model.to(Config.DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

# Training loop (similar to image model)
# ... [same structure as image training]

# Save final model
torch.save(model.state_dict(), 'models/checkpoints/audio_model.pth')
```

## Training Tips

1. **Data Augmentation**: Essential for preventing overfitting
2. **Learning Rate**: Start with 0.001, use scheduler
3. **Batch Size**: 32-64 depending on GPU memory
4. **Epochs**: 30-50 epochs typically sufficient
5. **Validation**: Monitor validation loss to prevent overfitting
6. **Early Stopping**: Stop if validation loss doesn't improve for 5 epochs

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Model Checkpoints

Save checkpoints in `models/checkpoints/`:
- `image_model.pth`: Final image model
- `audio_model.pth`: Final audio model
- `image_model_epoch_X.pth`: Intermediate checkpoints
- `audio_model_epoch_X.pth`: Intermediate checkpoints

## Pre-trained Weights

For demonstration without training:
```python
# Create dummy checkpoint for testing
import torch
from models.image_model import ImageDeepfakeDetector

model = ImageDeepfakeDetector(pretrained=True)
torch.save(model.state_dict(), 'models/checkpoints/image_model.pth')
```

## References

1. FaceForensics++: https://github.com/ondyari/FaceForensics
2. EfficientNet Paper: https://arxiv.org/abs/1905.11946
3. ASVspoof Dataset: https://www.asvspoof.org/