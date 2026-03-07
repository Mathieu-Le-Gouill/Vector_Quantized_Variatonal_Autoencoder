
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import Tensor, nn
from models.components.encoder import Encoder
from models.components.decoder import Decoder


class AE(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List=None,
                 in_shape: List=None,
                 conv_params: dict=None,
                 ):
        """
        Auto-encoder architecture for 2D/3D data.

        Args:
            latent_dim: dimension of the latent space (L)
            hidden_dims: list of hidden dimensions for the encoder and decoder
            in_shape: shape of the input tensor (C, H, W, ...)
             conv_params: dict of convolution parameters:
                         kernel_size, stride, padding, output_padding
        """
        super().__init__()

        assert latent_dim > 0, "latent_dim must be positive"
        assert in_shape is not None, "in_shape cannot be None"
        assert len(in_shape) >= 2, "in_shape must be at least 3D (C, H, W, ...)"

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        kernel_size, stride, padding, output_padding = self._extract_conv_params(conv_params)
        in_channels = in_shape[0]

        # --- Encoder ---
        self.enc_layer = Encoder(in_channels, hidden_dims, kernel_size, stride, padding)

        enc_flat_dim = self._compute_latent_shape(in_shape)

        self.enc_fc = nn.Linear(enc_flat_dim, latent_dim)

        # --- Decoder ---
        self.dec_fc = nn.Linear(latent_dim, enc_flat_dim)

        self.dec_layer = Decoder(in_channels, hidden_dims, kernel_size, stride, padding, output_padding)
        

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode the input tensor into the latent space.
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            latent tensor of shape (B, L)
        """
        enc = self.enc_layer(x) # (B, C, H, W, ...)
        enc_flat = torch.flatten(enc, start_dim=1) # (B, L)
        z = self.enc_fc(enc_flat)
    
        return z
    

    def decode(self, x: Tensor) -> Tensor:
        """
        Decode the latent tensor into the original space.
        Args:
            x: latent tensor of shape (B, L)
        Returns:
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        dec_flat = self.dec_fc(x) # (B, L)
        dec = dec_flat.view(-1, *self.enc_shape)  # (B, C, H, W, ...)
        recon = self.dec_layer(dec)

        return recon


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        z = self.encode(x)
        recon = self.decode(z)

        return recon
    

    def compute_loss(self, x):
        """
        Compute the loss function for the autoencoder.
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        """
        recon = self.forward(x)

        return F.mse_loss(recon, x)
    

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input x, returns the reconstructed x as recon
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...)
        """

        return self.forward(x)
    
    
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



