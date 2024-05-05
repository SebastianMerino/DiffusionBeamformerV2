import os
import torch
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
import torch.nn.functional as func
from torch.utils.data import DataLoader

from guided_diffusion import *
from dataset import CustomDataset
from model import UNETv13


def main():
    # Parsing args
    args = get_args()
    n_epoch = args.epochs
    data_folder = Path(args.data_folder)

    # Managing folders
    input_folder = data_folder / 'input_testing'
    output_folder = data_folder / 'target_testing'
    gen_folder = data_folder / 'generated_diffusion_GN'
    gen_folder.mkdir(parents=True, exist_ok=True)
    save_dir = Path(os.getcwd()) / 'weightsGN'

    data = CustomDataset(input_folder, output_folder, transform=True) # retrained without group_norm
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

    diffusion = create_gaussian_diffusion()
    model13A = UNETv13(residual=False, attention_res=[], group_norm=False).to(device)
    model13A.load_state_dict(torch.load(save_dir/f"model_{n_epoch}.pth", map_location=device))
    model13A.eval()

    BATCH_SIZE = 16
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    batch_number = 0
    for x, y in dataloader:  # x: images
        x = x.to(device)
        y_gen = diffusion.p_sample_loop(model13A, y.shape, x, progress=False, clip_denoised=True)
        print(f'Saving batch {batch_number:03d}, {datetime.now()}')
        for id, sample in enumerate(y_gen):
            name = (data.input_file_list[batch_number * BATCH_SIZE + id])
            np.save(gen_folder / name, sample.cpu().detach().numpy())
        batch_number += 1


def get_args():
    parser = argparse.ArgumentParser(description='Samples diffusion model')
    parser.add_argument('data_folder', type=str, help='parent folder where data is stored')
    parser.add_argument('epochs', type=int, help='number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
