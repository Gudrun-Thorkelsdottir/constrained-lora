'''
Copied and modified from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/compute_fid.py
'''



# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys
import Path
from tqdm import tqdm

import torch
from absl import flags
from cleanfid import fid, features
from cleanfid.inception_torchscript import InceptionV3W


from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import argparse
from torchvision import transforms

from unet import UNetModelWrapper

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="otcfm", help="flow matching model type")
    parser.add_argument('--dataset', type=str, default="camelyon17", help="WILDS dataset to use for computing FID")
    parser.add_argument('--size', type=int, default=96, help="size of dataset samples")
    parser.add_argument('--num_channel', type=int, default=128, help="base channel of UNet")
    parser.add_argument('--checkpoint_dir', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/", help="checkpoint directory")
    parser.add_argument('--integration_steps', type=int, default=100, help="number of inference steps")
    parser.add_argument('--integration_method', type=str, default="dopri5", help="integration method to use")
    parser.add_argument('--step', type=int, default=400500, help="training steps")
    parser.add_argument('--num_gen', type=int, default=50000, help="number of samples to generate")
    parser.add_argument('--tol', type=float, default=1e-5, help="Integrator tolerance (absolute and relative)")
    parser.add_argument('--batch_size_fid', type=int, default=1024, help="batch size to compute FID")

    return parser.parse_args()



# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
args = get_args()

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=args.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
PATH = f"{args.checkpoint_dir}{args.model}_{args.dataset}_weights_step_{args.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
#state_dict = checkpoint["ema_model"]
state_dict = checkpoint["net_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


#create dataloader
dataset = get_dataset(dataset=args.dataset)
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size_fid, drop_last=True)

def activations_from_dataloader(dataloader, model):

    all_acts = []
    for batch in tqdm(dataloader, desc="Computing activations (real)"):
        imgs = batch[0].to(device)
        with torch.no_grad():
            feats = fid.get_batch_features(imgs, model_name='inception', device=device)
        all_acts.append(feats)

    return torch.cat(all_acts, dim=0)

def activations_from_generator(generate, num_samples, batch_size):

    all_acts = []
    total = 0
    while total < num_samples:
        imgs = generate(batch_size).to(device)
        with torch.no_grad():
            feats = fid.get_batch_features(imgs, model_name='inception', device=device)
        all_acts.append(feats)
        total += imgs.shape[0]

    return torch.cat(all_acts, dim=0)[:num_samples]



# Define the integration method if euler is used
if args.integration_method == "euler":
    node = NeuralODE(new_net, solver=args.integration_method)


def generate(batch_size):
    with torch.no_grad():
        x = torch.randn(batch_size, 3, args.size, args.size, device=device)
        if args.integration_method == "euler":
            print("Use method: ", args.integration_method)
            t_span = torch.linspace(0, 1, args.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", args.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                new_net,
                x,
                t_span,
                rtol=args.tol,
                atol=args.tol,
                method=args.integration_method,
            )
    traj = traj[-1, :]  # .view([-1, 3, args.size, args.size]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    return img


print("Start computing FID")

real_dir = Path('fid_images/real')
fake_dir = Path('fid_images/fake')
real_dir.mkdir(parents=True, exist_ok=True)
fake_dir.mkdir(parents=True, exist_ok=True)



feat_model = features.build_feature_extractor("legacy_tensorflow", device)

real_feats = activations_from_dataloader(train_loader, feat_model)
gen_feats = activations_from_generator(generate, args.num_gen, args.batch_size_fid)

fid = fid_from_feats(real_feats, gen_feats)

print()
print("FID has been computed")
# print()
# print("Total NFE: ", new_net.nfe)
print()
print("FID: ", fid)
