import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
import numpy as np
import argparse

from torchdyn.core import NeuralODE
from unet import UNetModelWrapper

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size = 96  #camelyon17


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="otcfm", help="flow matching model type")
    parser.add_argument('--dataset', type=str, default="camelyon17", help="WILDS dataset to use for computing FID")
    parser.add_argument('--size', type=int, default=96, help="size of dataset samples")
    parser.add_argument('--num_channel', type=int, default=128, help="base channel of UNet")
    parser.add_argument('--checkpoint', type=str, default="/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/otcfm_camelyon17_weights_step_400500.pt", help="checkpoint")
    parser.add_argument('--integration_steps', type=int, default=100, help="number of inference steps")
    parser.add_argument('--integration_method', type=str, default="dopri5", help="integration method to use")
    parser.add_argument('--step', type=int, default=400500, help="training steps")
    parser.add_argument('--num_gen', type=int, default=50000, help="number of samples to generate")
    parser.add_argument('--tol', type=float, default=1e-5, help="Integrator tolerance (absolute and relative)")
    parser.add_argument('--batch_size_fid', type=int, default=128, help="batch size to compute FID")

    return parser.parse_args()


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        #inception = models.inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        inception = models.inception_v3(weights=weights, aux_logits=True)
        inception.fc = nn.Identity()
        self.inception = inception.eval()
        #self.features = nn.Sequential(*list(inception.children())[:-1])  # Remove last FC layer
        #self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.inception(x)
        #return x.view(x.size(0), -1)
        return x



def get_real_activations(loader, feature_extractor, num_gen):    #, transform):

    features = []
    count = 0

    for imgs in loader:
        imgs = imgs[0].to(device)

        #imgs = transform(imgs).to(device)

        with torch.no_grad():
            feats = feature_extractor(imgs).cpu().numpy()
            features.append(feats)


        count += imgs.shape[0]
        if count >= num_gen:
            break


    features = np.concatenate(features, axis=0)[:num_gen]
    return features




def get_gen_activations(generate, feature_extractor, num_gen, batch_size):   #, transform):

    features = []
    count = 0

    while count < num_gen:
        imgs = generate(batch_size).to(device)
        with torch.no_grad():
            feats = feature_extractor(imgs).cpu().numpy()
            features.append(feats)
        count += imgs.shape[0]

    features = np.concatenate(features, axis=0)[:num_gen]
    return features




def generate(net, batch_size, integration_steps):

    node = NeuralODE(net, solver="euler")

    with torch.no_grad():
        x = torch.randn(batch_size, 3, size, size, device=device)
        t_span = torch.linspace(0, 1, integration_steps + 1, device=device)
        traj = node.trajectory(x, t_span=t_span)

    traj = traj[-1, :]
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img



def calculate_fid(real_activations, gen_activations, eps=1e-6):

    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_activations, axis=0), np.cov(gen_activations, rowvar=False)

    ssdiff = np.sum((mu_real - mu_gen) ** 2)
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

    


def main():

    args = get_args()

    feature_extractor = InceptionV3().to(device)


    #make WILDS dataloader
    dataset = get_dataset(dataset=args.dataset)
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [
                transforms.Resize((299, 299)),   #to work with pretrained Inception network
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size_fid, drop_last=True)
    print("got dataset")

    #load trained model
    trained_net = UNetModelWrapper(
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
    trained_net.load_state_dict(checkpoint["net_model"])
    trained_net.eval()
    print("got model")

    #create generator function
    def generator_fn(batch_size):
        return generate(trained_net, batch_size, args.integration_steps)

    #calculate activations
    print("calculating activations...")
    real_acts = get_real_activations(train_loader, feature_extractor, args.num_gen)
    print("got real activations")
    gen_acts = get_gen_activations(generator_fn, feature_extractor, args.num_gen, args.batch_size_fid)
    print("got generated activations")

    fid = calculate_fid(real_acts, gen_acts)
    print(fid)



main()
