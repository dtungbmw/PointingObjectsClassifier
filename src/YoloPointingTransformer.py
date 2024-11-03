import torch
import torch.nn as nn
from ultralytics import YOLO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from praxis.pc_dataset import *

class TransformerPointingClassifier(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4):
        super(TransformerPointingClassifier, self).__init__()
        
        # YOLOv8 for object detection
        self.yolo = YOLO('yolov8s.pt')
        
        # Transformer with encoder and decoder layers
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers
        )
        
        # Linear layers to project YOLO features and pointing vector to embedding dimension
        self.yolo_feature_proj = nn.Linear(640, embed_dim)  # Assuming 640 is YOLOv8 feature dimension
        self.pointing_proj = nn.Linear(3, embed_dim)  # Pointing vector has x, y, z
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, image, pointing_vector):
        # Run YOLOv8 on the image to get detected object features
        yolo_results = self.yolo(image)
        boxes = yolo_results[0].boxes  # Detected boxes
        object_features = [box.feats for box in boxes]  # Extract YOLOv8 features
        
        # Project YOLO features to transformer input dimension and stack them
        object_embeddings = torch.stack([self.yolo_feature_proj(f) for f in object_features]).unsqueeze(1)
        
        # Project pointing vector to the same dimension
        pointing_embedding = self.pointing_proj(pointing_vector).unsqueeze(0).unsqueeze(1)
        
        # Encoder-Decoder Transformer
        transformer_output = self.transformer(
            src=object_embeddings,  # Object features as encoder input
            tgt=pointing_embedding  # Pointing vector as decoder input
        )
        
        # Classification head
        class_logits = self.classifier(transformer_output.squeeze(0).squeeze(0))
        
        return class_logits
    


class PointingDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: A list of tuples (image_path, pointing_vector, label) 
            transform: Transformations to apply to the image
        """
        self.data = data
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, pointing_vector, label = self.data[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Convert pointing vector and label to tensors
        pointing_vector = torch.tensor(pointing_vector, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, pointing_vector, label


class TransformerPointingPredictor:
    
    def predict_pointed_object(image_path, pointing_vector):
        # Load the trained model
        model = TransformerPointingClassifier(num_classes=10)  # Ensure num_classes matches training
        model.load_state_dict(torch.load("path/to/model_weights.pth"))  # Load saved model weights
        model.eval()  # Set model to evaluation mode

        # Move model to device (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Define the transform to resize and normalize the image
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Convert pointing vector to tensor and move to device
        pointing_vector = torch.tensor(pointing_vector, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(image, pointing_vector)
        
        # Get the predicted class label
        _, predicted_label = torch.max(output, 1)
        return predicted_label.item()
    
class TransformerPointingTrainer:
    
    def train(self):
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

        # Create dataset and dataloader
        dataset = PointingDataset(data, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
                # Initialize model, loss, and optimizer
        model = TransformerPointingClassifier(num_classes=10)  # Adjust `num_classes` as needed
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training loop
        num_epochs = 10  # Number of epochs

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (images, pointing_vectors, labels) in enumerate(dataloader):
                # Move data to device (e.g., GPU if available)
                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                pointing_vectors = pointing_vectors.to("cuda" if torch.cuda.is_available() else "cpu")
                labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images, pointing_vectors)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Accumulate loss for logging
                running_loss += loss.item()
                
                # Print loss every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                    running_loss = 0.0

        print("Training complete.")

# Example usage
##model = TransformerPointingClassifier(num_classes=10)
#image = torch.randn(1, 3, 640, 640)  # Example input image
#pointing_vector = torch.tensor([0.5, -0.2, 1.0])  # Example pointing vector

# Forward pass
#output = model(image, pointing_vector)
#print("Class logits:", output)
