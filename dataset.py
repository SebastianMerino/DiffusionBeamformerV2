import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transform
        # self.transform_input = transforms.Normalize([0,0],[0.1172,0.1172])
        self.transform_output = transforms.Lambda(lambda t: (t * 2) - 1)
        self.input_file_list = sorted(os.listdir(input_folder))
        self.output_file_list = sorted(os.listdir(output_folder))

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.input_folder, self.input_file_list[idx])
        x = np.load(file_path)
        x = torch.Tensor(x)
        x = x.permute(2, 0, 1)

        file_path = os.path.join(self.output_folder, self.output_file_list[idx])
        y = np.load(file_path)
        y = torch.Tensor(y)
        y = y.unsqueeze(0)

        if self.transform:
            # x = self.transform_input(x)
            y = self.transform_output(y)
        return x, y
