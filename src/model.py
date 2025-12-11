import torch.nn as nn
import torchvision.models as models

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes, pretrained=True, device="cpu"):
    model = MultiLabelResNet(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)
