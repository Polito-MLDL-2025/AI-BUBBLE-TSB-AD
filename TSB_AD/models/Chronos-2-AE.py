import torch
from torch.nn import nn
from chronos import BaseChronosPipeline

class VAEHead(nn.Module):
    """
    A Variational Autoencoder (VAE) head designed to replace the standard Quantile Head in a Chronos pipeline.
    It processes transformer embeddings, mapping them to a latent space from which it attempts
    to reconstruct the original input, providing a probabilistic interpretation of the data.

    Args:
        input_dim (int): The dimensionality of the input embeddings from the transformer.
        latent_dim (int): The dimensionality of the latent space. Defaults to 32.
        output_dim (int): The dimensionality of the reconstructed output. Defaults to 1.
    """
    def __init__(self, input_dim, latent_dim=32, output_dim=1):
        super().__init__()
        
        # Encoder
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim) 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class AEHead(nn.Module):
    """
    A standard Autoencoder (AE) head with increased complexity, designed to process transformer embeddings.
    It learns a compressed, latent representation of the input and then reconstructs it,
    aiming to capture essential features for anomaly detection.

    Args:
        input_dim (int): The dimensionality of the input embeddings from the transformer.
        latent_dim (int): The dimensionality of the bottleneck (latent) layer. Defaults to 64.
        output_dim (int): The dimensionality of the reconstructed output. Defaults to 1.
    """
    def __init__(self, input_dim, latent_dim=64, output_dim=1): # Increased default latent_dim
        super().__init__()
        
        # Intermediate hidden dimensions
        hidden_dim1 = input_dim # Use input_dim for first hidden layer
        hidden_dim2 = input_dim // 2 
        
        # Encoder: input_dim -> hidden_dim1 -> hidden_dim2 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU() # Activation before the latent space
        )
        
        # Decoder: latent_dim -> hidden_dim2 -> hidden_dim1 -> output_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim) # Output layer no activation for reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        # Return None for mu and logvar to maintain interface compatibility
        return recon, None, None



class ChronosAnomalyModel(nn.Module):
    """
    Integrates a pre-trained Chronos pipeline with either a VAEHead or AEHead for anomaly detection.
    The Chronos model's backbone is frozen, and its embeddings are passed to the
    specified head for reconstruction-based anomaly scoring.

    Args:
        pipeline (BaseChronosPipeline): The pre-trained Chronos pipeline model.
        head_type (str): The type of anomaly detection head to use ('vae' or 'ae'). Defaults to 'vae'.
        latent_dim (int): The dimensionality of the latent space for the VAE/AE head. Defaults to 64.
    """
    def __init__(self, pipeline: BaseChronosPipeline, head_type='vae', latent_dim=64):
        super().__init__()
        self.model = pipeline.model
        
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get dimensions
        backbone_dim = self.model.config.d_model
        
        # Initialize Head
        if head_type == 'vae':
            self.head = VAEHead(input_dim=backbone_dim + 2, latent_dim=latent_dim, output_dim=backbone_dim + 2)
        elif head_type == 'ae':
            self.head = AEHead(input_dim=backbone_dim + 2, latent_dim=latent_dim, output_dim=backbone_dim + 2)

    def forward(self, context_tensor):
        """
        context_tensor: [Batch, N_Vars, Seq_Len]
        """
        # Handle Input Shapes
        # We expect a 3D tensor: [Batch, Variables, Time]
        if context_tensor.dim() != 3:
            raise ValueError(f"Expected 3D input [Batch, Vars, Time], got {context_tensor.shape}")
            
        B, V, T = context_tensor.shape
        device = context_tensor.device

        # Create Group IDs
        # We want variates of the same sample to share information.
        # IDs will look like: [0, 0, 0, 1, 1, 1, ..., B-1, B-1]
        group_ids = torch.arange(B, device=device).repeat_interleave(V)

        # Flatten for Encoder
        # Chronos expects [Total_Batch, Seq_Len]
        context_flat = context_tensor.reshape(B * V, T)

        # Get Embeddings
        with torch.no_grad():
            encoder_outputs, loc_scale, _, _ = self.model.encode(
                context=context_flat,
                group_ids=group_ids 
            )

            last_hidden_state = encoder_outputs[0]  # Shape: [B*V, Num_Patches+Tokens, Dim]
            # 5. Filter [REG] Token
            if getattr(self.model.chronos_config, "use_reg_token", False):
                last_hidden_state = last_hidden_state[:, :-1, :]
            
            loc, scale = loc_scale

            # Reshape loc/scale to match sequence length
            # We need to repeat them for every patch so we can concat them
            # loc shape: [Batch, 1] -> [Batch, Patches, 1]
            num_patches = last_hidden_state.shape[1]
            loc_seq = loc.unsqueeze(1).repeat(1, num_patches, 1)
            scale_seq = scale.unsqueeze(1).repeat(1, num_patches, 1)
            
            # Concatenate
            # new shape: [Batch, Patches, Dim + 2]
            combined_input = torch.cat([last_hidden_state, loc_seq, scale_seq], dim=-1) 
            
            
        # Pass through Head
        # The head processes each patch embedding independently
        recon, mu, logvar = self.head(combined_input)  # recon shape: [B*V, Patches, Dim]
        
        return recon, mu, logvar, combined_input