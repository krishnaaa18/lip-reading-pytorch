import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LipReadingDataset
from model import LipReadingModel
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 0.001

# Load dataset
label_map = {'d1.pt': 0, 'd2.pt': 1}  # Update as needed
dataset = LipReadingDataset(root_dir='D:/Github repoS/lip reading/processed', label_map=label_map)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Initialize model, loss, and optimizer
model = LipReadingModel(num_classes=10, hidden_size=64).to(device)
 # Adjust num_classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # ✅ Move model to GPU if available
print(f"Loaded {len(train_loader.dataset)} samples.")


# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    torch.cuda.empty_cache()


    model.train()
    print("Starting training...")

    for inputs, labels in train_loader:
        # Move tensors to the GPU if available
        inputs, labels = inputs.to(device), labels.to(device)


        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the model
torch.save(model.state_dict(), 'lip_reading_model.pth')
