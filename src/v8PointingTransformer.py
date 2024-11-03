import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO


# Define Fast R-CNN Backbone for Feature Extraction
'''
class FastRCNNBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(FastRCNNBackbone, self).__init__()
        # Load a pre-trained Faster R-CNN model and use its backbone (ResNet + FPN)
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Use the backbone part only (ResNet + FPN without region proposals)
        self.backbone = self.model.backbone

    def forward(self, image):
        # Forward pass through the Fast R-CNN backbone to get feature maps
        feature_maps = self.backbone(image)

        # Extract the final layer feature map, which will be a dictionary
        # You can concatenate all the multi-scale features or use one of them
        # Here we use the feature map from the first scale (['0'])
        return feature_maps['0']  # You can select other layers like ['1'], ['2'], etc.
'''


# YOLOv8 Backbone for Feature Extraction
class YOLOBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(YOLOBackbone, self).__init__()

        # Load YOLOv8 from PyTorch Hub (can also use YOLOv3 or YOLOv8)
        #if pretrained:
        #self.model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)
        #else:
        #    self.model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=False)
        self.model = YOLO('yolov8s.pt')  # Load a pre-trained model

        # Remove the detection layers, keep only the backbone (the first layers)
        #self.backbone = nn.Sequential(*list(self.model.model[:10]))  # The first 10 layers form the backbone

    def forward(self, image):
        # Pass the image through the YOLO backbone to get feature maps
        feature_maps = self.model.model.backbone(image)  # Output feature maps

        # Return the feature maps [batch_size, channels, height, width]
        return feature_maps


class PointingDeviceClassification(nn.Module):
    def __init__(self, num_classes, transformer_hidden_dim, num_transformer_layers):
        super(PointingDeviceClassification, self).__init__()

        # YOLO backbone for image feature extraction
        self.image_backbone = YOLOBackbone()  # Placeholder for YOLO or CNN feature extractor

        # Replace YOLO backbone with Fast R-CNN backbone for feature extraction
        #self.image_backbone = FastRCNNBackbone(pretrained=True)  # Fast R-CNN backbone

        # Pointing vector embedding (from DeepPoint)
        self.pointing_embedding = nn.Linear(3, transformer_hidden_dim)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=8),
            num_layers=num_transformer_layers
        )

        # Classification head
        self.fc = nn.Linear(transformer_hidden_dim, num_classes)

    def enable_training(self):
        # Ensure that only the transformer and your classification layers are set to train mode
        self.transformer_encoder.train()
        self.fc.train()
        #self.image_backbone.train()  # Use only if you want to update the backbone

    def forward(self, image, pointing_vector):
        # Pass the image through the YOLO backbone to get feature maps
        image_features = self.image_backbone(image)  # [batch_size, channels, h, w]

        # Flatten the feature maps to get image tokens
        batch_size, channels, h, w = image_features.shape
        image_tokens = image_features.view(batch_size, channels, h * w).permute(0, 2, 1)  # [batch_size, num_patches, hidden_dim]

        # Embed the 3D pointing direction vector (from DeepPoint)
        pointing_token = self.pointing_embedding(pointing_vector).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Concatenate the pointing token with the image tokens
        tokens = torch.cat((image_tokens, pointing_token), dim=1)  # [batch_size, num_patches + 1, hidden_dim]

        # Apply transformer encoder to the concatenated tokens
        transformer_output = self.transformer_encoder(
            tokens.permute(1, 0, 2))  # [num_patches + 1, batch_size, hidden_dim]

        # Take the output corresponding to the pointing token (last token)
        pointing_output = transformer_output[-1, :, :]  # [batch_size, hidden_dim]

        # Classify the output
        output = self.fc(pointing_output)  # [batch_size, num_classes]

        return output
