import os
import glob
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class VOCMultiLabelDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, classes, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self):
        xml_files = glob.glob(os.path.join(self.annotations_dir, "*.xml"))
        samples = []
        for xml_path in xml_files:
            filename = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = os.path.join(self.images_dir, filename + ".jpg")
            if os.path.exists(img_path):
                samples.append((img_path, xml_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = torch.zeros(len(self.classes), dtype=torch.float32)

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in self.class_to_idx:
                labels[self.class_to_idx[name]] = 1.0
        return labels

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        labels = self._parse_xml(xml_path)

        return img, labels


def get_default_transforms(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
