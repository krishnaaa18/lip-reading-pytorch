import torch
from model import LipReadingModel
import torch.nn.functional as F 
from torchvision import transforms
from PIL import Image
import os
model = LipReadingModel(input_size=64, num_frames=763, hidden_size=256, num_classes=10)
 # Adjust based on your model
model.load_state_dict(torch.load('lip_reading_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), strict=False)

model.eval()
input_data = torch.randn(763, 1, 64, 64)
input_data = input_data.unsqueeze(0)
# output = model(input_data)
# probabilities = F.softmax(output, dim=1)
# predicted_class = torch.argmax(probabilities, dim=1)
# print(f"Predicted probabilities: {probabilities}")
# print(f"Predicted class: {predicted_class.item()}")
with torch.no_grad():
    output = model(input_data)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f"Predicted probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class.item()}")
