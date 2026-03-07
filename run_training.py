import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.auto_encoder import AE
from models.beta_vae import BetaVAE
from models.vae import VAE
from utils.train_model import train
from utils.visualize_recon import visualize_recon

if __name__ == "__main__":
    # --- Parameters ---
    batch_size = 64
    lr = 1e-3
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1] for tanh
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize Models ---
    modelAE = AE(
        latent_dim=128,
        hidden_dims=[128, 256],
        in_shape=[1, 28, 28],
        conv_params = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        }
    ).to(device)

    modelVAE = VAE(
        latent_dim=128,
        hidden_dims=[128, 256],
        in_shape=[1, 28, 28],
        conv_params = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        }
    ).to(device)

    modelBetaVAE = BetaVAE(
        latent_dim=128,
        beta=2.0,
        hidden_dims=[128, 256],
        in_shape=[1, 28, 28],
        conv_params = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        }
    ).to(device)

    models = [modelAE, modelVAE, modelBetaVAE]
    models_names = ["AE", "VAE", "BetaVAE"]

    # --- Prepare Optimizers ---
    optimizerAE = optim.Adam(modelAE.parameters(), lr=lr)
    optimizerVAE = optim.Adam(modelVAE.parameters(), lr=lr)
    optimizerBetaVAE = optim.Adam(modelBetaVAE.parameters(), lr=lr)
    optimizers = [optimizerAE, optimizerVAE, optimizerBetaVAE]

    # --- Train ---
    train(models, epochs, optimizers, train_loader, test_loader, device)

    #-- Visualize Reconstructions ---
    visualize_recon(models, models_names, test_loader, device)


