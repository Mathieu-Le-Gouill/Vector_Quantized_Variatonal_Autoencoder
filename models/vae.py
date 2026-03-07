
from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch import Tensor, nn
from models.components.encoder import Encoder
from models.components.decoder import Decoder
import math


class VAE(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List=None,
                 in_shape: List=None,
                 conv_params: dict=None,
                 ):
        """
        Variational Auto-encoder architecture for 2D/3D data.

        Args:
            latent_dim: dimension of the latent space (L)
            hidden_dims: list of hidden dimensions for the encoder and decoder
            in_shape: shape of the input tensor (C, H, W, ...)
            conv_params: dictionary of convolution parameters (kernel_size, stride, padding, output_padding)
        """
        super().__init__()

        assert latent_dim > 0, "latent_dim must be positive"
        assert in_shape is not None, "in_shape cannot be None"
        assert len(in_shape) >= 3, "in_shape must be at least 3D (B, C, H, W, ...)"

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        kernel_size, stride, padding, output_padding = self._extract_conv_params(conv_params)
        in_channels = in_shape[0]

        # --- Encoder ---
        self.enc_layer = Encoder(in_channels, hidden_dims, kernel_size, stride, padding)

        enc_flat_dim = self._compute_latent_shape(in_shape)

        self.mu_fc = nn.Linear(enc_flat_dim, latent_dim)
        self.logvar_fc = nn.Linear(enc_flat_dim, latent_dim)

        # --- Decoder ---
        self.dec_fc = nn.Linear(latent_dim, enc_flat_dim)

        self.dec_layer = Decoder(in_channels, hidden_dims, kernel_size, stride, padding, output_padding)


    def encode(self, x: Tensor) -> Tensor:
        """
        Encode the input tensor into the latent space.
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            sampled tensor of shape (B, L), mean tensor of shape (B, L), log variance tensor of shape (B, L)
        """
        enc = self.enc_layer(x) # (B, C, H, W, ...)
        enc_flat = torch.flatten(enc, start_dim=1) # (B, latent_dim)

        z, mu, log_var = self._bottleneck(enc_flat)

        return z, mu, log_var
    
    
    def _bottleneck(self, x: Tensor) -> Tuple:
        """
        Bottleneck layer that samples from the latent distribution.
        Args:
            x: latent tensor of shape (B, L)
        Returns:
            sampled tensor of shape (B, L), mean tensor of shape (B, L), log variance tensor of shape (B, L)
        """
        mu = self.mu_fc(x)
        log_var = self.logvar_fc(x) # log(std**2)

        std = torch.exp(0.5 * log_var) # sqrt(exp(log_var))
        epsilon = torch.randn_like(std, device=std.device) # reparameterization trick

        z = mu + std * epsilon

        return z, mu, log_var
    
    
    def decode(self, x: Tensor) -> Tensor:
        """
        Decode the latent tensor into the original space.
        Args:
            x: latent tensor of shape (B, L)
        Returns:
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        dec_flat = self.dec_fc(x)# (B, latent_dim)
        dec = dec_flat.view(-1, *self.enc_shape)  # (B, C, H, W, ...)
        recon = self.dec_layer(dec)

        return recon

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...), mean tensor of shape (B, L), log variance tensor of shape (B, L)
        """
        z, mu, log_var = self.encode(x)
        recon = self.decode(z)

        return recon, mu, log_var
    

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

        return recon_loss + kl_loss
    
    
    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input x, returns the reconstructed x as recon
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        recon, _, _ = self.forward(x)

        return recon
    
    
    def _extract_conv_params(self, conv_params):
        """
        Extract convolution parameters with default values.
        Args:
            conv_params (dict or None): dictionary of convolution parameters.
        Returns:
            tuple: (kernel_size, stride, padding, output_padding)
        """
        conv_params = conv_params or {}
        return (
            conv_params.get("kernel_size", 3),
            conv_params.get("stride", 2),
            conv_params.get("padding", 1),
            conv_params.get("output_padding", 1),
        )
                
    
    def _compute_latent_shape(self, in_shape):
        """
        Compute the encoder output shape per sample and its flattened size.
        Args:
            in_shape: shape of the input tensor (C, H, W, ...)
        Returns:
            enc_flat_dim: int, flattened size per sample
        """
        assert self.enc_layer is not None

        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_shape)
            enc = self.enc_layer(dummy_input) # (1, C, H, W,...)
            self.enc_shape = enc.shape[1:] # (C, H, W,...)
            enc_flat_dim = math.prod(self.enc_shape)  #(C * H * W * ...)

        return enc_flat_dim

