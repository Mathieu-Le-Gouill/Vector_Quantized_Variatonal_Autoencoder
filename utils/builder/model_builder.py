from models.auto_encoder import AE
from models.beta_vae import BetaVAE
from models.vae import VAE
from models.vq_vae import VQVAE

def build_model(config, device):

    if config.model.type == "AE":
        return AE(
            latent_dim = config.model.latent_dim,
            hidden_dims = config.model.hidden_dims,
            in_shape = config.model.in_shape,
            conv_params = config.model.conv_params
        ).to(device)

    elif config.model.type == "VAE":
        return VAE(
            latent_dim = config.model.latent_dim,
            hidden_dims = config.model.hidden_dims,
            in_shape = config.model.in_shape,
            conv_params = config.model.conv_params
        ).to(device)
    
    elif config.model.type == "BetaVAE":
        return BetaVAE(
            latent_dim = config.model.latent_dim,
            beta = config.model.beta,
            hidden_dims = config.model.hidden_dims,
            in_shape = config.model.in_shape,
            conv_params = config.model.conv_params
        ).to(device)
    
    elif config.model.type == "VQVAE":
        return VQVAE(
            latent_dim = config.model.latent_dim,
            voc_size = config.model.voc_size,
            beta = config.model.beta,
            hidden_dims = config.model.hidden_dims,
            in_shape = config.model.in_shape,
            conv_params = config.model.conv_params
        ).to(device)
    else:
        raise ValueError(f"Unknown model {config.model.type}")