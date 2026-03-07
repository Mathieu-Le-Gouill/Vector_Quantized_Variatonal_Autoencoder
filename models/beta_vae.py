
from typing import List
import torch.nn.functional as F

from models.vae import VAE

class BetaVAE(VAE):
    def __init__(self,
                 latent_dim: int,
                 beta: float = 1.0,
                 hidden_dims: List=None,
                 in_shape: List=None,
                 conv_params: dict=None
                 ):
        """
        Beta Variational Auto-Encoder architecture for 2D/3D data.

        Args:
            latent_dim: dimension of the latent space (L)
            beta: beta parameter to modulate the weight of the KL divergence term in the loss function
            hidden_dims: list of hidden dimensions for the encoder and decoder
            in_shape: shape of the input tensor (C, H, W, ...)
            conv_params: dictionary of convolution parameters (kernel_size, stride, padding, output_padding)
        """
        super().__init__(latent_dim, hidden_dims, in_shape, conv_params)
        self.beta = beta


    def compute_loss(self, x):
        """
        Compute the VAE loss function (reconstruction + KL divergence).
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            total loss (scalar)
        """
        batch_size = x.size(0)
        recon, mu, logvar = self.forward(x)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kl_loss = kl.sum(dim=1).mean()
        recon_loss = F.mse_loss(recon, x, reduction='sum') / batch_size

        return recon_loss + self.beta * kl_loss