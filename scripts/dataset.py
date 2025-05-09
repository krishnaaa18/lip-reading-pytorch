import os
import torch
from torch.utils.data import Dataset

class LipReadingDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        self.label_map = label_map  # e.g., {'d1.pt': 0, 'd2.pt': 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        tensor = torch.load(os.path.join(self.root_dir, file_name))  # Shape: [T, 1, 64, 64]
        label = self.label_map[file_name]  # Integer label
        return tensor, label


