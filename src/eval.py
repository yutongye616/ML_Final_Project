%%writefile src/eval.py
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader

from dataset import VOCMultiLabelDataset, get_default_transforms
from model import build_model


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    CLASSES = ["person", "car", "dog", "cat", "bicycle"]
    num_classes = len(CLASSES)

    VOC_ROOT = os.path.join("data", "VOC2012")
    IMG_DIR = os.path.join(VOC_ROOT, "JPEGImages")
    ANN_DIR = os.path.join(VOC_ROOT, "Annotations")

    transforms = get_default_transforms()

    dataset = VOCMultiLabelDataset(
        images_dir=IMG_DIR,
        annotations_dir=ANN_DIR,
        classes=CLASSES,
        transform=transforms,
    )

    print("Total images:", len(dataset))

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # build model and load weights
    model = build_model(num_classes=num_classes, device=device)
    ckpt_path = os.path.join("results", "model_final.pth")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")


if __name__ == "__main__":
    evaluate()
