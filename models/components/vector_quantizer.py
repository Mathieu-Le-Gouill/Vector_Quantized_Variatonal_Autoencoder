from typing import List
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 voc_size: int,
                 latent_dim: int,
                 beta: float=0.25
                ):
        """
        Vector Quantization layer.

        Args:
            voc_size: number of embeddings in the codebook (K)
            latent_dim: dimension of each embedding vector (L)
            beta: commitment loss weight
        """
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim
        self.voc_size = voc_size

        self.latent_emb = nn.Embedding(voc_size, latent_dim)
        self.latent_emb.weight.data.uniform_(-1/latent_dim, 1/latent_dim)

    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: latent tensor of shape (B, L, H, W, ...)
        Returns:
            vector quantized loss (scalar), quantized latent tensor of shape (B, L, H, W, ...)
        """
        latents = x.permute(0, *range(2, x.ndim), 1).contiguous() # (B, H, W, ..., L)
        flat_latents = latents.view(-1, self.latent_dim)  # (B * H * W * ..., L)

        v_emb = self.latent_emb.weight # (K, L)
        
        # Compute L2 distance between latents and embeddings
        dist = torch.sum(flat_latents**2, dim=1, keepdim=True) \
             + torch.sum(v_emb**2, dim=1) \
             - 2 * torch.matmul(flat_latents, v_emb.t()) # (B * H * W * ..., K)

        enc_idxs = torch.argmin(dist, dim=1) # (B * H * W * ...)
        
        # Get the closest vector from embeddings dictionnary
        quantized_latents = v_emb[enc_idxs]
        quantized_latents = quantized_latents.view(latents.shape)  # (B, H, W, ..., L)

        # Compute the losses
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)

        vq_loss = embedding_loss + self.beta * commitment_loss

        # Backward pass Straight-Through Estimator trick to have : d quantized_x / d x = 1
        quantized_latents = latents + (quantized_latents - latents).detach() # (B, H, W, ..., L)
        quantized_latents = latents = x.permute(0, -1, *range(1, latents.ndim-1)).contiguous() # (B, L, H, W, ...)

        return vq_loss, quantized_latents