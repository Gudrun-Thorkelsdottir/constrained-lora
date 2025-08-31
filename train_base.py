import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm


from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from models.unet import UNet




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()




def main():

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True


    model = UNet(
            input_channels=3,
            input_height=96,
            ch=128,
            ch_mult=(1, 2, 2, 4),
            num_res_blocks=2,
            attn_resolutions=(16,),
            resamp_with_conv=True,
            dropout=0,
            )
    model.to(device)



    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    dataset = get_dataset(dataset="camelyon17")
    train_data = dataset.get_subset(
        "train",
        transform=transforms.ToTensor(),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)


    





main()
