import torch 
import numpy as np
import argparse

from unet import UNetModelWrapper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default="otcfm", help="flow matching model type")
    parser.add_argument('--dataset', type=str, default="camelyon17", help="WILDS dataset to use for computing FID")
    parser.add_argument('--size', type=int, default=96, help="size of dataset samples")
    parser.add_argument('--num_channel', type=int, default=128, help="base channel of UNet")
    parser.add_argument('--checkpoint', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/otcfm_camelyon17_weights_step_400500.pt", help="checkpoint")
    parser.add_argument('--discrete_steps', type=int, default=100, help="number of inference steps")
    parser.add_argument('--integration_method', type=str, default="dopri5", help="integration method to use")
    parser.add_argument('--batch_size', type=int, default=50, help="batch size to compute FID")

    return parser.parse_args()



@torch.no_grad()
def estimate_path_length(u, x, n_steps):
    """
    Estimates average path length over a vector field trajectory.

    Args:
        u: Callable, the vector field function u_theta(x, t)
        x: Tensor of shape [batch_size, dim], initial samples
        n_steps: Integer, number of Euler steps

    Returns:
        L: Scalar tensor, average path length across batch
    """


    delta_t = 1.0 / n_steps
    t = torch.tensor(0.0, device=x.device)
    length = 0.0


    for _ in range(n_steps):

        v = u(t, x)
        x_next = x + delta_t * v

        length += torch.norm(x_next - x, dim=1).sum()

        x = x_next
        t += delta_t


    L = length / x.shape[0]
    return L





def main():

    args = get_args()


    #load trained model
    u = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    u.load_state_dict(checkpoint["net_model"])
    u.eval()


    #draw (batch_size) samples from the prior
    x = torch.randn(args.batch_size, 3, args.size, args.size, device=device) 

    #estimate path length
    L = estimate_path_length(u, x, args.discrete_steps)

    print("Estimated average length: " + str(L))



main()
