import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ASL_CNN(nn.Module):
    '''
    Initializes the CNN Model.
    Architecture:
      - Transfer learning from resnet50
      - Freeze all layers except the last residual block
      - Replace fully connected layer with
        - Linear -> ReLU -> Dropout -> Linear
      - Output has 28 classes (A-Z, DEL, SPACE)
    '''
    def __init__(self, num_classes=28):
        super(ASL_CNN, self).__init__()

        # Batch normalization as the first layer for input normalization
        self.batch_norm = nn.BatchNorm2d(3)

        self._base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in self._base_model.parameters():
            param.requires_grad = False

        for param in self._base_model.layer4.parameters():
            param.requires_grad = True

        num_features = self._base_model.fc.in_features # 2048
        self._base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    '''
    Executes a forward pass of the CNN model.

    Parameters
    x : Tensor
      The input image.

    Returns a tensor of the model's prediction.
    '''
    def forward(self, x):
        x = self.batch_norm(x)
        return self._base_model(x)


class ASL_CNN_FeatureExtractor(nn.Module):
    '''
    Feature extractor version of ASL_CNN, removing only the final Linear classification layer
    Extends pytorch's nn.Module.
    '''
    def __init__(self, trained_asl_cnn):
        super(ASL_CNN_FeatureExtractor, self).__init__()

        self.batch_norm = trained_asl_cnn.batch_norm

        self.resnet_layers = nn.Sequential(
            trained_asl_cnn._base_model.conv1,
            trained_asl_cnn._base_model.bn1,
            trained_asl_cnn._base_model.relu,
            trained_asl_cnn._base_model.maxpool,
            trained_asl_cnn._base_model.layer1,
            trained_asl_cnn._base_model.layer2,
            trained_asl_cnn._base_model.layer3,
            trained_asl_cnn._base_model.layer4,
            trained_asl_cnn._base_model.avgpool
        )

        self.fc_layers = nn.Sequential(
            trained_asl_cnn._base_model.fc[0],  # Linear(2048 -> 512)
            trained_asl_cnn._base_model.fc[1],  # ReLU
            trained_asl_cnn._base_model.fc[2]   # Dropout
        )
        # Skipping final Linear(512 -> 28) classifier

    '''
    Executes a forward pass of the CNN model.

    Parameters
    x : Tensor (batch, 3, 128, 128)
      The input image.

    Returns 512-dim features as a Tensor (batch, 512)
    '''
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

