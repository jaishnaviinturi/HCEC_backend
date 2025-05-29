import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class HybridHCECModel(nn.Module):
    def __init__(self):
        super(HybridHCECModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_fc = nn.Linear(768, 512)
        combined_feature_dim = 512 + 512
        self.regression_head = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        vit_features = self.vit_fc(vit_features)
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        regression_output = self.regression_head(combined_features)
        return regression_output