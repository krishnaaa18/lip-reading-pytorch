import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset import LipReadingDataset
from torch.utils.data import DataLoader

label_map = {'d1.pt': 0}  # You can add more later

dataset = LipReadingDataset(root_dir='D:/Github repoS/lip reading/processed', label_map=label_map)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for video, label in loader:
    print(f"Video tensor shape: {video.shape}")
    print(f"Label: {label}")
    break