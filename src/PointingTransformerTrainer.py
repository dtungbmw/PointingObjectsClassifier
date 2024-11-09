import torch
import torch.nn as nn
import hydra
import logging
import torch.optim as optim
import torch.nn.functional as F
from PointingTransformer import YOLOBackbone, PointingDeviceClassification
from omegaconf import DictConfig, OmegaConf
from POCTrainer import *


def string_to_ascii_tensor(s):
    # Convert each character in the string to its ASCII code
    ascii_values = [ord(c) for c in s]
    # Create a tensor from the list of ASCII values
    return torch.tensor(ascii_values, dtype=torch.long)


def transform_image(input_tensor):
        # Remove extra dimension
        input_tensor = input_tensor.squeeze(1)  # Shape now (1, 3, 2028, 2704)

        # Resize to (1, 3, 640, 640)
        input_tensor = F.interpolate(input_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        return input_tensor


@hydra.main(version_base=None, config_path="../conf", config_name="base")    
def main(cfg: DictConfig) -> None:
    # Define number of classes for your dataset
    num_classes = 3  # Example: 3 object categories
    transformer_hidden_dim = 256 #512
    num_transformer_layers = 6
    learning_rate = 1e-4

    # Instantiate the DeepPoint model and the combined model with YOLO backbone
    #deep_point_model = DeepPoint()
    pointing_classification_model = PointingDeviceClassification(num_classes, transformer_hidden_dim,
                                                                        num_transformer_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the correct device
    pointing_classification_model = pointing_classification_model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pointing_classification_model.parameters(), lr=learning_rate)

    # Define training data
    #train_loader = [(torch.rand(8, 3, 640, 640), torch.rand(8, 3), torch.randint(0, 3, (8,))) for _ in range(100)]  # Simulated data

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    trainer = POCTrainer()
    train_loader = trainer.setup_dataloader(cfg)
    print(cfg)    

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        pointing_classification_model.enable_training()
        running_loss = 0.0
        try:
            for i, (images, pointing_vectors, labels) in enumerate(train_loader):
                images = transform_image(images)
                images = images.to(device).float()
                pointing_vectors = pointing_vectors.to(device).float()
                # Forward pass
                outputs = pointing_classification_model(images, pointing_vectors)
                loss = criterion(outputs, torch.tensor(labels).to(device))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0
        except:
            print("ignore... cont...")
            continue

    # Save the entire model
    torch.save(pointing_classification_model, 'full_model.pth')
    print('Finished Training')



if __name__ == "__main__":
    main()