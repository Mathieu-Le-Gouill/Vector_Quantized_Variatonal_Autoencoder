from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dims,
                 kernel_size,
                 stride,
                 padding
                ):
        """
        Encoder part of a classic autoencoder architecture.
        Args:
            in_channels: number of input channels
            hidden_dims: list of hidden dimensions for the encoder
            kernel_size: kernel size for the convolutional layers
            stride: stride for the convolutional layers
            padding: padding for the convolutional layers
        """
        super().__init__()

        enc_layers = []
        in_ch = in_channels

        for h_dim in hidden_dims:
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, h_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_ch = h_dim
        
        self.encoder = nn.Sequential(*enc_layers)

    def forward(self, x):
        """
        Encode the input tensor into the latent space.
        Args:
            x: input tensor of shape (B, C, H, W, ...)
        Returns:
            latent tensor of shape (B, L)
        """
        enc = self.encoder(x)

        return enc
        