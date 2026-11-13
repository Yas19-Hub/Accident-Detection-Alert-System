import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np

# ------------------------
# Config
# ------------------------
data_root = "dataset_split"   # contains train/, val/, test/
batch_size = 64
num_epochs = 50
early_stopping_patience = 10   # stop if val acc doesn't improve for N epochs
num_workers = 4                # tune depending on CPU cores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # speed up on fixed-size inputs

# ------------------------
# Data transforms
# ------------------------
# Stronger augmentations for train to reduce overfitting
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------
# Datasets & Dataloaders
# ------------------------
train_dir = os.path.join(data_root, "train")
val_dir   = os.path.join(data_root, "val")
test_dir  = os.path.join(data_root, "test")

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=val_test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

pin_mem = True if device.type == "cuda" else False

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
)

# ------------------------
# Class weights (to handle imbalance)
# ------------------------
targets = np.array(train_dataset.targets)
class_counts = np.bincount(targets)
print("Train class counts:", class_counts)

# inverse-frequency weighting, normalized
class_weights = 1.0 / (class_counts + 1e-6)   # avoid div by zero
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights.cpu().numpy())

# ------------------------
# Model: ResNet18 (pretrained)
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# ------------------------
# Loss, Optimizer, Scheduler
# ------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Reduce LR when val_acc plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ------------------------
# Training + Validation Loop with Early Stopping
# ------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{running_corrects / total:.4f}",
        })

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device, phase="Val"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=phase, leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{running_corrects / total:.4f}",
            })

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        since = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device, phase="Validation"
        )

        # Step LR scheduler based on val_acc
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")

        elapsed = time.time() - since
        print(
            f"Epoch {epoch + 1} finished in {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Check for improvement
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve_epochs = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "models/accident_frame_resnet18.pth")
            print(f"✅ New best model saved (val_acc={best_val_acc:.4f})")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s)")

        # Early stopping condition
        if no_improve_epochs >= patience:
            print(
                f"\n⛔ Early stopping triggered at epoch {epoch + 1}. "
                f"Best epoch was {best_epoch} with val_acc={best_val_acc:.4f}"
            )
            break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model, best_val_acc


# ------------------------
# Run training
# ------------------------
model, best_val_acc = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=num_epochs,
    patience=early_stopping_patience,
)

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

# ------------------------
# Final test evaluation
# ------------------------
test_loss, test_acc = eval_one_epoch(
    model, test_loader, criterion, device, phase="Test"
)
print(f"\nTest Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
