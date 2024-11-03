import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#jfrom ultralytics import YOLO


# Define Fast R-CNN Backbone for Feature Extraction
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


# YOLOv5 Backbone for Feature Extraction
class YOLOBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(YOLOBackbone, self).__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Access the YOLO backbone layers
        self.backbone = nn.Sequential(*list(self.model.model.children())[:10])  # Extract backbone

    def forward(self, image):
        # Pass the image through the YOLO backbone to get feature maps
        #feature_maps = self.model.model.backbone(image)  # Output feature maps
        print(f"image type = {type(image)}")
        print(f"image shape = {image.shape}")
        feature_maps = self.backbone(image)

        # Return the feature maps [batch_size, channels, height, width]
        return feature_maps


class PointingDeviceClassification(nn.Module):
    def __init__(self, num_classes, transformer_hidden_dim, num_transformer_layers):
        super(PointingDeviceClassification, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_hidden_dim = transformer_hidden_dim

        # YOLO backbone for image feature extraction
        self.image_backbone = YOLOBackbone()  # Placeholder for YOLO or CNN feature extractor

        # Replace YOLO backbone with Fast R-CNN backbone for feature extraction
        #self.image_backbone = FastRCNNBackbone(pretrained=True)  # Fast R-CNN backbone

        self.pointing_embedding = nn.Linear(3, transformer_hidden_dim)
        #self.pointing_projection = nn.Linear(transformer_hidden_dim, num_patches)  # To match image tokens' hidden dim

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=8), num_layers=num_transformer_layers)

        # Classification head
        self.fc = nn.Linear(transformer_hidden_dim, num_classes)

    def enable_training(self):
        # Ensure that only the transformer and your classification layers are set to train mode
        self.transformer_encoder.train()
        self.fc.train()
        #self.image_backbone.train()  # Use only if you want to update the backbone

    def pad_and_concat(self, tensor1, tensor2, dim):
        # Ensure tensor shapes match in all dimensions except the concatenation dimension
        pad_dim = [max(tensor1.size(i), tensor2.size(i)) for i in range(len(tensor1.shape))]

        # Pad tensor1 if needed
        pad1 = [(pad_dim[i] - tensor1.size(i)) if i != dim else 0 for i in range(len(tensor1.shape))]
        pad1 = [(0, pad) for pad in pad1[::-1]]  # reverse for correct padding order
        tensor1_padded = torch.nn.functional.pad(tensor1, sum(pad1, ()))

        # Pad tensor2 if needed
        pad2 = [(pad_dim[i] - tensor2.size(i)) if i != dim else 0 for i in range(len(tensor2.shape))]
        pad2 = [(0, pad) for pad in pad2[::-1]]  # reverse for correct padding order
        tensor2_padded = torch.nn.functional.pad(tensor2, sum(pad2, ()))

        # Concatenate along the specified dimension
        return torch.cat((tensor1_padded, tensor2_padded), dim=dim)

    def forward(self, image, pointing_vector):
        image = image.to(self.device)
        pointing_vector = pointing_vector.to(self.device)
        # Pass the image through the YOLO backbone to get feature maps
        image_features = self.image_backbone(image)  # [batch_size, channels, h, w]

        # Check if the output is a tuple (multiple feature maps)
        if isinstance(image_features, tuple):
            image_features = image_features[0]  # Pick the first feature map (you can change this based on your needs)

        # If the output is 3D (batch_size, channels, num_patches), handle accordingly
        if len(image_features.shape) == 3:
            batch_size, channels, num_patches = image_features.shape
            image_tokens = image_features.permute(0, 2, 1)  # [batch_size, num_patches, channels]
        elif len(image_features.shape) == 4:
            batch_size, channels, h, w = image_features.shape
            image_tokens = image_features.view(batch_size, channels, h * w).permute(0, 2, 1)  # [batch_size, num_patches, channels]

        hidden_dim = channels

        # Embed the 3D pointing direction vector (from DeepPoint)
        pointing_token = self.pointing_embedding(pointing_vector).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        transformer_hidden_dim = self.transformer_hidden_dim

        if hidden_dim != transformer_hidden_dim:
            # Project image tokens and pointing tokens to the transformer_hidden_dim
            image_tokens = nn.Linear(hidden_dim, transformer_hidden_dim).to(self.device)(
                image_tokens)  # [batch_size, num_patches, transformer_hidden_dim]
            # pointing_token = nn.Linear(hidden_dim, transformer_hidden_dim).to(self.device)(pointing_token)  # [batch_size, 1, transformer_hidden_dim]

        # Concatenate the image tokens with the pointing token
        tokens = torch.cat((image_tokens, pointing_token),
                           dim=1)  # [batch_size, num_patches + 1, transformer_hidden_dim]

        #pointing_token = self.pointing_projection(pointing_token)
        print(f"image_tokens type = {type(image_tokens)}")
        print(f"pointing_tokens type = {type(image_tokens)}")
        print(f"image_tokens shape = {image_tokens.shape}")
        print(f"pointing_tokens shape = {pointing_token.shape}")
        # Concatenate the pointing token with the image tokens
        #tokens = self.pad_and_concat(image_tokens, pointing_token, dim=1)  # [batch_size, num_patches + 1, hidden_dim]

        # Apply transformer encoder to the concatenated tokens
        transformer_output = self.transformer_encoder(
            tokens.permute(1, 0, 2))  # [num_patches + 1, batch_size, hidden_dim]

        # Take the output corresponding to the pointing token (last token)
        pointing_output = transformer_output[-1, :, :]  # [batch_size, hidden_dim]

        # Classify the output
        output = self.fc(pointing_output)  # [batch_size, num_classes]

        return output
