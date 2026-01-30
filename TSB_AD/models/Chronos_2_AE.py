import numpy as np
import torchinfo
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from chronos import BaseChronosPipeline

from .base import BaseDetector
from ..utils.dataset import TSDataset
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import get_gpu, EarlyStoppingTorch

class VAEHead(nn.Module):
    """
    A Variational Autoencoder (VAE) head designed to replace the standard Quantile Head in a Chronos pipeline.
    It processes transformer embeddings, mapping them to a latent space from which it attempts
    to reconstruct the original input, providing a probabilistic interpretation of the data.

    Args:
        input_dim (int): The dimensionality of the input embeddings from the transformer.
        latent_dim (int): The dimensionality of the latent space. Defaults to 32.
        output_dim (int): The dimensionality of the reconstructed output. Defaults to 1.

    References:
        - Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
            Implementation from: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial.git
        - PyTorch VAE Library: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
    """
    def __init__(self, input_dim, latent_dim=32, output_dim=1):
        super().__init__()
        
        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(input_dim // 2, latent_dim)
        self.fc_var = nn.Linear(input_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim) 
        )

    def reparameterize(self, mu, logvar):
        # ! Are we sure? Double check later via debugging
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x_shared = self.encoder_shared(x)
        mu = self.fc_mu(x_shared)
        logvar = self.fc_var(x_shared)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class AEHead(nn.Module):
    """
    A standard Autoencoder (AE) head with increased complexity, designed to process transformer embeddings.
    It learns a compressed, latent representation of the input and then reconstructs it.

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
    def __init__(self, pipeline, head_type='vae', latent_dim=32, device=None):
        super().__init__()
        self.pipeline = pipeline
        self.model = pipeline.model
        
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get dimensions
        backbone_dim = self.model.config.d_model
        
        # Initialize head
        if head_type == 'vae':
            self.head = VAEHead(input_dim=backbone_dim + 2, latent_dim=latent_dim, output_dim=backbone_dim + 2)
        elif head_type == 'ae':
            self.head = AEHead(input_dim=backbone_dim + 2, latent_dim=latent_dim, output_dim=backbone_dim + 2)

    def forward(self, context_tensor):
        """
        Forward pass through the Chronos model and the anomaly detection head.
        
        Args:
            context_tensor: [Batch, N_Vars, Seq_Len]

        Returns:
            recon: Reconstructed embeddings from the head.
            mu: Latent mean (for VAE) or None (for AE).
            logvar: Latent log-variance (for VAE) or None (for AE).
            combined_input: Original embeddings concatenated with loc and scale.
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
            encoder_outputs, loc_scale, _, num_context_patches = self.model.encode(
                context=context_flat,
                group_ids=group_ids 
            )

            last_hidden_state = encoder_outputs[0]  # Shape: [B*V, Num_Patches+Tokens, Dim]
            
            # Filter [REG] Token and future patches
            last_hidden_state = last_hidden_state[:, :num_context_patches, :]
            
            # Prepare loc and scale
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

class Chronos2AE(BaseDetector):
    def __init__(self, 
                 slidingWindow=100,
                 head_type='vae',
                 latent_dim=32,
                 batch_size=32,
                 lr=1e-3,
                 beta=1.0,
                 epochs=50,
                 validation_size=0.2,
                 patience=5,
                 stride=1):
        super().__init__()
        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.window_size = slidingWindow
        self.head_type = head_type
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.epochs = epochs
        self.validation_size = validation_size
        self.patience = patience
        self.stride = stride
        self.__anomaly_score = None
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

        # Download Chronos-2 pipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=self.device
        )

    def fit(self, data, y=None):
        split_idx = int((1-self.validation_size)*len(data)) # Using 80% training, 20% validation
        tsTrain = data[:split_idx]
        tsValid = data[split_idx:]

        # Initialize Datasets and DataLoaders
        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.window_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=True
        )        
        val_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.window_size, stride=self.stride),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Build the model
        self.model = ChronosAnomalyModel(
            pipeline=self.pipeline,
            head_type=self.head_type, 
            latent_dim=self.latent_dim,
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.head.parameters(), lr=self.lr)
        
        # Training Loop
        self.model.train()
        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            total_loss = 0
            for batch, _ in train_loader:
                batch = batch.permute(0, 2, 1).to(self.device) # [Batch, Time, Vars] -> [Batch, Vars, Time]
                
                # Forward
                recon, mu, logvar, target = self.model(batch)
                
                # Loss Calculation
                # Loss was high because of how it was summed for each element in the batch.
                loss = self.criterion(recon, target, mu, logvar)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation every epoch
            # ? To speed up it could be good to do this only every 5 epochs, tho it is no the way is done in TSBAD afaik
            # ! Uncomment later, keep this commented for quick debugging only
            self.model.eval()
            val_loss = 0
            if len(val_loader) > 0:
                with torch.no_grad():
                    for batch, _ in val_loader:
                        batch = batch.permute(0, 2, 1).to(self.device)
                        recon, mu, logvar, original_embeddings = self.model(batch)
                        loss = self.criterion(recon, original_embeddings, mu, logvar)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
            else:
                avg_val_loss = avg_loss

            self.model.train()

            # implement early stopping
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("\tEarly stopping")
                break
            
            epoch_pbar.set_description(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | Early stopping: {self.early_stopping.counter}")

        return self

    def decision_function(self, X):
        dataset = ReconstructDataset(X, window_size=self.window_size, stride=self.stride) # By default: [window_size = 100, stride = 1]
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        window_scores = []
        
        with torch.no_grad():
            for batch, _ in tqdm(loader, desc="Computing score", unit="batch"):
                # original shape: [Batch, Window_Size, Vars], must be permuted for Chronos 2
                batch = batch.permute(0, 2, 1).to(self.device) # shape: [Batch, Vars, Window_Size]
                recon, _, _, target = self.model(batch)
                
                # recon and target shape: [Batch * Vars, Num_Patches, Backbone_Dim + 2]
                
                # Compute MSE per windows
                mse = F.mse_loss(recon, target, reduction='none') # shape: [Batch * Vars, Num_Patches, Backbone_Dim + 2]
                
                # Average over Patches and Dimensions to get one score per window
                # score shape: [Batch * Vars]
                score = mse.mean(dim=[1, 2]).cpu().numpy()
                window_scores.append(score)
        
        # window_scores contains scores for each window/variable combination
        window_scores = np.concatenate(window_scores)

        # Reshape to [Num_Windows, Num_Vars]
        # We know that for each window in the batch, we processed V variables sequentially
        # So the flat structure is [W0_V0, W0_V1, ..., W0_Vm, W1_V0, ...]
        if X.ndim > 1 and X.shape[1] > 1:
            num_vars = X.shape[1]
            window_scores = window_scores.reshape(-1, num_vars)
            # Aggregate across variables to get one score per window
            # In the rest of the codebase they usally use mean and sum:
            #   USAD: Averages the reconstruction error across all features and time points within a window
            #   TranAD: Averages the reconstruction error over the feature dimension
            #   Chronos: Processes each channel independently and then averages the final scores
            window_scores = window_scores.mean(axis=1)

        # Map window-level scores back to original time series points.
        # Since we use a sliding window with stride=1, each point is covered by multiple windows,
        # we aggregate these scores by averaging them
        final_scores = np.zeros(len(X))
        counts = np.zeros(len(X))
        
        for i, score in enumerate(window_scores):
            # Window i covers [i * stride : i * stride + window_size]
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            final_scores[start_idx: end_idx] += score
            counts[start_idx: end_idx] += 1
        counts[counts == 0] = 1 # Avoid division by 0
        
        # Store final anomaly scores
        self.__anomaly_score = final_scores / counts
        return self.__anomaly_score
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    # ! May need to try to tune beta
    def criterion(self, recon_x, x, mu=None, logvar=None, beta=None):
        # x, recon_x shaope: [Batch, Seq_Len, Dim] -- NOTE: Seq_Len here is actually Num_Patches
        # e.g. [32, 4, 770] for Batch=32, Num_Patches=4, Backbone_Dim(Chronos 2 default)=768 + 2 (loc and scale)

        # Compute Reconstruction Loss
        mse = F.mse_loss(recon_x, x, reduction='none') # mse shape: [32, 4, 770]
        recon_loss = torch.sum(mse, dim=2).mean()

        # If mu and logvar are None, computes AE loss (Recon only).
        if mu is None or logvar is None:
            return recon_loss
        
        # Analytical KLD: -0.5 * sum(1 + log(var) - mu^2 - var)
        # We calculate it element-wise first
        kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        #! Are we sure about this? Double check later with debugging
        #! Also read the VAE papers again to be sure
        # Sum over Latent Dimension (dim=-1), Mean over Batch and Sequence (dim=0, 1)
        # This gives the average "Total Divergence per Vector" in the batch
        kld_per_vector = torch.sum(kld_element, dim=-1) 
        
        # Normalize KLD to match the scale of MSE.
        # We take the mean over all dimensions (Batch, Patches, Latent_Dim).
        # This makes KLD "average divergence per latent unit", comparable to 
        # MSE's "average error per input unit".
        kld_loss = torch.mean(kld_per_vector)

        if beta is None:
            beta = self.beta

        return recon_loss + (beta * kld_loss)