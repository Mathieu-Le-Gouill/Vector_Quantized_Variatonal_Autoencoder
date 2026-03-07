
from typing import List
import torch.nn.functional as F
import torch
from torch import Tensor, nn
from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.components.vector_quantizer import VectorQuantizer
import math


class VQVAE(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 voc_size: int,
                 beta: float=0.25,
                 hidden_dims: List=None,
                 in_shape: List=None,
                 conv_params: dict=None,
                 ):
        """
        Vector Quantized Variational Auto-encoder architecture for 2D/3D data.

        Args:
            latent_dim: dimension of the latent / embedding space (L)
            voc_size: size of the vocabulary in the embedding
            beta: beta parameter to controls how much we want to weigh commitment loss compared to other components
            hidden_dims: list of hidden dimensions for the encoder and decoder
            in_shape: shape of the input tensor (C, H, W, ...)
            conv_params: dictionary of convolution parameters (kernel_size, stride, padding, output_padding)
        """
        super().__init__()

        self.beta = beta
        self.voc_size = voc_size

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

        self.enc_fc = nn.Linear(enc_flat_dim, latent_dim)

        self.vq_layer = VectorQuantizer(voc_size, latent_dim, beta)

        # --- Decoder ---
        self.dec_fc = nn.Linear(latent_dim, enc_flat_dim)

        self.dec_layer = Decoder(in_channels, hidden_dims, kernel_size, stride, padding, output_padding)


    def encode(self, x: Tensor) -> Tensor:
        """
        Encode the input tensor into the latent space.
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            quantized latent tensor of shape (B, L, H, W, ...), vector quantized loss (scalar)
        """
        enc = self.enc_layer(x) # (B, C, H, W, ...)
        enc_flat = torch.flatten(enc, start_dim=1) 

        z = self.enc_fc(enc_flat) # (B, L)
        vq_loss, quantized_latents = self.vq_layer(z)

        return quantized_latents, vq_loss
    
    
    def decode(self, x: Tensor) -> Tensor:
        """
        Decode the latent tensor into the original space.
        Args:
            x: latent tensor of shape (B, L)
        Returns:
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        dec_flat = self.dec_fc(x)# (B, L)
        dec = dec_flat.view(-1, *self.enc_shape)  # (B, C, H, W, ...)
        recon = self.dec_layer(dec)

        return recon

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...), vector quantized loss (scalar)
        """
        quantized_latents, vq_loss = self.encode(x)
        recon = self.decode(quantized_latents)

        return recon, vq_loss
    

    def compute_loss(self, x):
        """
        Compute the VAE loss function (reconstruction + KL divergence).
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            total loss (scalar)
        """
        recon, vq_loss = self.forward(x)

        recon_loss = F.mse_loss(recon, x)

        return recon_loss + vq_loss
    
    
    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input x, returns the reconstructed x as recon
        Args:                
            x: input tensor of shape (B, C, H, W, ...)
        Returns:                
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        recon, _ = self.forward(x)

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
            enc_flat_dim = math.prod(self.enc_shape)

        return enc_flat_dim

