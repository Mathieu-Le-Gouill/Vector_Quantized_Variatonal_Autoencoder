from torch import nn

class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dims,
                 kernel_size,
                 stride,
                 padding,
                 output_padding
                ):
        """
        Decoder part of a classic autoencoder architecture.
        Args:
            in_channels: number of input channels
            hidden_dims: list of hidden dimensions for the encoder
            kernel_size: kernel size for the convolutional layers
            stride: stride for the convolutional layers
            padding: padding for the convolutional layers
            output_padding: output padding for the transposed convolutional layers
        """
        super().__init__()

        decoder_dims = hidden_dims[::-1]
    
        # Create pairs of in/out channels for each convtranspose layers
        channel_pairs = list(zip(decoder_dims, decoder_dims[1:])) + [(decoder_dims[-1], decoder_dims[-1])]

        dec_layers = [
            nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            for in_ch, out_ch in channel_pairs
        ]

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[0], in_channels, kernel_size=kernel_size, padding=padding),
            nn.Tanh()
        )

        self.dec_layer = nn.Sequential(
            *dec_layers, 
            *self.final_layer
        )

    def forward(self, x):
        """
        Decode the latent tensor into the original space.
        Args:
            x: latent tensor of shape (B, L)
        Returns:
            reconstructed tensor of shape (B, C, H, W, ...)
        """
        dec = self.dec_layer(x)

        return dec
