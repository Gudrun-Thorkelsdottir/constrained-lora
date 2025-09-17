'''

Copied and modified from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/train_cifar10.py

'''


# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
#from absl import app, flags
import argparse
from torchvision import datasets, transforms
from tqdm import trange
from utils import ema, generate_samples, infiniteloop

from conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from unet import UNetModelWrapper

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="otcfm", help="flow matching model type")
    parser.add_argument('--dataset', type=str, default="camelyon17", help="WILDS dataset to use for training")
    parser.add_argument('--output_dir', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/out/ot-cfm/", help="output_directory")
    parser.add_argument('--num_channel', type=int, default=128, help="base channel of UNet")

    parser.add_argument('--lr', type=float, default=1e-4, help="target learning rate")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="gradient norm clipping")
    parser.add_argument('--total_steps', type=int, default=400501, help="total training steps")  # Lipman et al uses 400k but double batch size
    parser.add_argument('--warmup', type=int, default=5000, help="learning rate warmup")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")  # Lipman et al uses 128
    parser.add_argument('--num_workers', type=int, default=4, help="workers of Dataloader")
    parser.add_argument('--ema_decay', type=float, default=0.9999, help="ema decay rate")
    parser.add_argument('--parallel', type=bool, default=False, help="multi gpu training")
    parser.add_argument('--save_step', type=int, default=10, help="frequency of saving checkpoints, 0 to disable during training")
    parser.add_argument('--save_dir', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/")

    parser.add_argument('--checkpoint', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/otcfm_cifar10_weights_step_400000.pt", help="pretrained checkpoint to be finetune")

    return parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
args = get_args()


def warmup_lr(step):
    return min(step, args.warmup) / args.warmup


def train():

    #args = get_args()
    print(
        "lr, total_steps, ema decay, save_step:",
        args.lr,
        args.total_steps,
        args.ema_decay,
        args.save_step,
    )

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
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    datalooper = infiniteloop(train_loader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)

    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    net_model.load_state_dict(checkpoint["net_model"])
    ema_model.load_state_dict(checkpoint["ema_model"])

    optim.load_state_dict(checkpoint["optim"])
    sched.load_state_dict(checkpoint["sched"])
    step = checkpoint["step"]

    print("loaded checkpoint")
    print("starting from step " + str(step))


    if args.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if args.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif args.model == "icfm":
        print("Not supported")
        #FM = ConditionalFlowMatcher(sigma=sigma)
    elif args.model == "fm":
        print("Not supported")
        #FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif args.model == "si":
        print("Not supported")
        #FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = args.save_dir 
    os.makedirs(savedir, exist_ok=True)

    with trange(step, args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, args.ema_decay)  # new

            # sample and Saving the weights
            if args.save_step > 0 and step % args.save_step == 0:
                generate_samples(net_model, args.parallel, args.output_dir, step, x1.shape[2], net_="normal")
                generate_samples(ema_model, args.parallel, args.output_dir, step, x1.shape[2], net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{args.model}_{args.dataset}_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    train()
