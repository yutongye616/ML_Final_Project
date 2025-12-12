import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import VOCMultiLabelDataset, get_default_transforms
from model import build_model


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    CLASSES = ["person", "car", "dog", "cat", "bicycle"]
    num_classes = len(CLASSES)

    # Paths are relative to the project root: ML_Final_Project/
    VOC_ROOT = os.path.join("data", "VOC2012")
    IMG_DIR = os.path.join(VOC_ROOT, "JPEGImages")
    ANN_DIR = os.path.join(VOC_ROOT, "Annotations")

    print("Images dir:", IMG_DIR)
    print("Annotations dir:", ANN_DIR)

    transforms = get_default_transforms()

    dataset = VOCMultiLabelDataset(
        images_dir=IMG_DIR,
        annotations_dir=ANN_DIR,
        classes=CLASSES,
        transform=transforms,
    )

    print("Total images:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty – check your paths in train.py")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = build_model(num_classes=num_classes, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print("Training loss:", avg_loss)

    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)


if __name__ == "__main__":
    train()
