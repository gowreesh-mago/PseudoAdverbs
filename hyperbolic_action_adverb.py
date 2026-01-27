import argparse
from torch.utils.data import Dataset, DataLoader
from hyperbolic_helpers.hierarchy import Hierarchy
import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch
from hypll.tensors import TangentTensor, ManifoldTensor
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
import torch.nn.functional as F
from hypll.optim import RiemannianAdam
import time
from tqdm import tqdm
import wandb

'''
The original paper also improves the text embeddings using an attention layer but I am skipping this for now and keeping it simple here.
'''

class PoincareOperations:
    """Wrapper for Poincare ball operations using hypll"""

    def __init__(self, curvature=1.0, dim=512, learnable_curvature=False, device='cuda', debug=False):
        self.c = Curvature(curvature, requires_grad=learnable_curvature)
        self.manifold = PoincareBall(c=self.c)
        self.dim = dim
        self.device = device
        self.debug = debug

        # Origin point on manifold (needed for logmap)
        self.origin = torch.zeros(dim)

    def exp_map(self, v):
        """
        Exponential map at origin: T_0 P^n -> P^n
        Projects Euclidean features to Poincare ball

        Args:
            v: Euclidean vector (B, D)
        Returns:
            Poincare ball point (ManifoldTensor)
        """
        if self.debug:
            print(f"[exp_map] Input shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
            print(f"[exp_map] Input norm: {v.norm(dim=-1).mean():.4f}")

        # Convert regular tensor to TangentTensor, then map to manifold
        tangent_vec = TangentTensor(data=v, man_dim=1, manifold=self.manifold)
        result = self.manifold.expmap(tangent_vec)

        if self.debug:
            print(f"[exp_map] Output type: {type(result)}, shape: {result.shape}")
            print(f"[exp_map] Output norm: {result.tensor.norm(dim=-1).mean():.4f}")

        return result

    def log_map(self, x):
        """
        Logarithmic map to origin: P^n -> T_0 P^n

        Args:
            x: Point on Poincare ball (ManifoldTensor or Tensor)
        Returns:
            Tangent vector (TangentTensor that acts like regular tensor)
        """
        if self.debug:
            print(f"[log_map] Input type: {type(x)}, shape: {x.shape if isinstance(x, ManifoldTensor) else x.shape}")

        # Ensure x is ManifoldTensor
        if not isinstance(x, ManifoldTensor):
            x = ManifoldTensor(data=x, man_dim=1, manifold=self.manifold)

        # Get device from underlying tensor
        device = x.tensor.device

        if self.debug:
            print(f"[log_map] Input norm: {x.tensor.norm(dim=-1).mean():.4f}")
            print(f"[log_map] Creating origin with dim={self.dim}, device={device}")

        # Create origin as a single point (D,) on the manifold
        # This will broadcast to match x's batch/temporal dimensions
        origin_data = torch.zeros(self.dim, device=device, requires_grad=False)
        origin_manifold = ManifoldTensor(data=origin_data, man_dim=0, manifold=self.manifold)

        if self.debug:
            print(f"[log_map] Origin shape: {origin_data.shape}, x shape: {x.tensor.shape}")

        tangent_vec = self.manifold.logmap(origin_manifold, x)

        if self.debug:
            print(f"[log_map] Output type: {type(tangent_vec)}, shape: {tangent_vec.shape}")

        return tangent_vec

    def distance(self, x, y):
        """
        Geodesic distance on Poincare ball

        Args:
            x, y: Points on Poincare ball (B, D) or (B, N, D)
        Returns:
            Distance (B,) or (B, N)
        """
        return self.manifold.dist(x, y)

    def mobius_add(self, x, y):
        """
        Mobius addition on Poincare ball (for residual connections)

        Args:
            x, y: Points on Poincare ball
        Returns:
            x ⊕ y on Poincare ball
        """
        return self.manifold.mobius_add(x, y)

    def mobius_scalar_mul(self, r, x):
        """
        Mobius scalar multiplication (custom implementation)
        r ⊗ x in Poincare ball

        Args:
            r: Scalar or tensor of scalars
            x: Point on Poincare ball
        Returns:
            r ⊗ x on Poincare ball
        """
        # Implementation based on: https://arxiv.org/abs/1805.09112
        # r ⊗ x = tanh(r * arctanh(||x||)) * x / ||x||
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
        # Ensure we're within valid range for arctanh
        x_norm_clamped = x_norm.clamp(max=1.0 - 1e-5)

        arctanh_x_norm = torch.atanh(x_norm_clamped)
        new_norm = torch.tanh(r * arctanh_x_norm)

        result = new_norm * x / x_norm
        return self.project(result)

    def project(self, x, eps=-1.0):
        """Project points onto Poincare ball (ensure constraints)"""
        if isinstance(x, ManifoldTensor):
            return self.manifold.project(x, eps=eps)
        else:
            # Convert to ManifoldTensor first
            x_manifold = ManifoldTensor(data=x, man_dim=1, manifold=self.manifold)
            return self.manifold.project(x_manifold, eps=eps)

class ActionAdverbDataset(Dataset):
    def __init__(self, data_dir, features_dir, hierarchy, split='train'):
        '''
        For each batch we want the features, action ,adverb, parent action, and parent adverb.
        '''
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.hierarchy = hierarchy
        self.csv_path = os.path.join(data_dir, f'{split}.csv')
        # Load the dataset
        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data item at index idx
        clip_id = self.data.iloc[idx]['clip_id']

        # Load from .npz files
        with np.load(os.path.join(self.features_dir, f'{clip_id}_flow.npz'), allow_pickle=True) as flow_data:
            # Try common key names, then fallback to first available array
            if 'features' in flow_data:
                flow_features = flow_data['features']
            elif 'arr_0' in flow_data:
                flow_features = flow_data['arr_0']
            else:
                flow_features = flow_data[flow_data.files[0]]
            # Make a copy to ensure we can use it after closing the file
            flow_features = np.array(flow_features, dtype=np.float32)

        with np.load(os.path.join(self.features_dir, f'{clip_id}_rgb.npz'), allow_pickle=True) as rgb_data:
            if 'features' in rgb_data:
                rgb_features = rgb_data['features']
            elif 'arr_0' in rgb_data:
                rgb_features = rgb_data['arr_0']
            else:
                rgb_features = rgb_data[rgb_data.files[0]]
            # Make a copy to ensure we can use it after closing the file
            rgb_features = np.array(rgb_features, dtype=np.float32)

        action = self.data.iloc[idx]['clustered_action']
        adverb = self.data.iloc[idx]['clustered_adverb']
        action_parent = self.hierarchy.get_action_parent(action)
        adverb_parent = self.hierarchy.get_adverb_parent(adverb)
        return {
            'clip_id': clip_id,
            'flow_features': flow_features,
            'rgb_features': rgb_features,
            'action': action,
            'adverb': adverb,
            'action_parent': action_parent,
            'adverb_parent': adverb_parent
        }


def collate_fn(batch):
    """Custom collate function for ActionAdverbDataset

    Handles variable-length temporal features by applying mean pooling
    over the temporal dimension to get fixed-size representations.
    """
    # Helper function to pool temporal features
    def pool_features(features):
        """Apply mean pooling over temporal dimension if needed"""
        if features.ndim == 2:
            # Shape: (T, D) -> take mean over time
            return np.mean(features, axis=0)
        elif features.ndim == 1:
            # Already a 1D vector
            return features
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

    # Pool all features in the batch
    flow_features_pooled = [pool_features(item['flow_features']) for item in batch]
    rgb_features_pooled = [pool_features(item['rgb_features']) for item in batch]

    return {
        'clip_id': [item['clip_id'] for item in batch],
        'flow_features': flow_features_pooled,
        'rgb_features': rgb_features_pooled,
        'action': [item['action'] for item in batch],
        'adverb': [item['adverb'] for item in batch],
        'action_parent': [item['action_parent'] for item in batch],
        'adverb_parent': [item['adverb_parent'] for item in batch]
    }
    
class Model(nn.Module):
    def __init__(self, config, hierarchy: Hierarchy, init_glove=True):
        super(Model, self).__init__()
        self.config = config
        self.action_vocab = hierarchy.get_all_actions()
        self.action_parent_vocab = hierarchy.get_all_parent_actions()
        self.adverb_vocab = hierarchy.get_all_adverbs()
        self.adverb_parent_vocab = hierarchy.get_all_parent_adverbs()

        self.video_encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
        self.action_encoder_video = nn.Sequential(
            nn.Linear(config['output_dim'], config['bottleneck_dim']),
            nn.ReLU(),
            nn.Linear(config['bottleneck_dim'], config['output_dim'])
        )
        self.adverb_encoder_video = nn.Sequential(
            nn.Linear(config['output_dim'], config['bottleneck_dim']),
            nn.ReLU(),
            nn.Linear(config['bottleneck_dim'], config['output_dim'])
        )

        
        # Create embeddings with GloVe dimension
        self.action_embeddings = nn.Embedding(len(self.action_vocab), config['glove_dim'])
        self.adverb_embeddings = nn.Embedding(len(self.adverb_vocab), config['glove_dim'])
        self.action_parent_embeddings = nn.Embedding(len(self.action_parent_vocab), config['glove_dim'])
        self.adverb_parent_embeddings = nn.Embedding(len(self.adverb_parent_vocab), config['glove_dim'])

        # Projection layers from GloVe dim to output dim
        self.action_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.adverb_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.action_parent_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.adverb_parent_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])

        if config['combination_method'] == 'concat':
            self.action_adverb_combined_proj = nn.Linear(2 * config['glove_dim'], config['output_dim'])
        else:
            self.action_adverb_combined_proj = nn.Linear(config['glove_dim'], config['output_dim'])

        if init_glove:
            self.init_glove_embeddings()

    def init_glove_embeddings(self, freeze=False):
        """Initialize all embedding layers with GloVe vectors"""
        
        # Load and assign for each embedding layer
        action_weights, self.action_2_idx = self.load_glove_embeddings(self.action_vocab)
        self.action_embeddings.weight.data.copy_(action_weights)

        adverb_weights, self.adverb_2_idx = self.load_glove_embeddings(self.adverb_vocab)
        self.adverb_embeddings.weight.data.copy_(adverb_weights)

        action_parent_weights, self.action_parent_2_idx = self.load_glove_embeddings(self.action_parent_vocab)
        self.action_parent_embeddings.weight.data.copy_(action_parent_weights)

        adverb_parent_weights, self.adverb_parent_2_idx = self.load_glove_embeddings(self.adverb_parent_vocab)
        self.adverb_parent_embeddings.weight.data.copy_(adverb_parent_weights)

        if freeze:
            self.action_embeddings.weight.requires_grad = False
            self.adverb_embeddings.weight.requires_grad = False
            self.action_parent_embeddings.weight.requires_grad = False
            self.adverb_parent_embeddings.weight.requires_grad = False

    def load_glove_embeddings(self, vocab, embedding_dim=300):
        """
        Load GloVe embeddings for vocabulary

        Args:
            glove_path: Path to GloVe file (e.g., glove.6B.300d.txt)
            vocab: List of words
            embedding_dim: GloVe dimension

        Returns:
            Embedding matrix (vocab_size, embedding_dim)
        """
        vocab_lower = [w.lower() for w in vocab]
        vocab_lower.sort()
        vocab_to_idx = {w: i for i, w in enumerate(vocab_lower)}

        # Initialize with random for OOV words
        embeddings = np.random.randn(len(vocab), embedding_dim).astype(np.float32) * 0.01

        found = set()
        with open(self.config['glove_path'], 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                word = tokens[0]
                if word in vocab_to_idx:
                    vec = np.array(tokens[1:], dtype=np.float32)
                    embeddings[vocab_to_idx[word]] = vec
                    found.add(word)

        print(f"Loaded GloVe embeddings: {len(found)}/{len(vocab)} words found")
        not_found = list(set(vocab_lower) - found)
        print(f"Not found words: {not_found}")
        return torch.FloatTensor(embeddings), vocab_to_idx



    def get_action_embeddings(self, indices):
        """Get projected action embeddings"""
        return self.action_text_proj(self.action_embeddings(indices))

    def get_adverb_embeddings(self, indices):
        """Get projected adverb embeddings"""
        return self.adverb_text_proj(self.adverb_embeddings(indices))
    
    def get_action_parent_embeddings(self, indices):
        """Get projected action parent embeddings"""
        return self.action_parent_text_proj(self.action_parent_embeddings(indices))
    
    def get_adverb_parent_embeddings(self, indices):
        """Get projected adverb parent embeddings"""
        return self.adverb_parent_text_proj(self.adverb_parent_embeddings(indices))
    
    def combine_action_adverb_embeddings(self, action_embeds, adverb_embeds):
        """Combine action and adverb embeddings based on the specified method"""
        if self.config['combination_method'] == 'concat':
            combined = torch.cat((action_embeds, adverb_embeds), dim=-1)
        elif self.config['combination_method'] == 'mean':
            combined = (action_embeds + adverb_embeds)/2.0
        else:
            raise ValueError(f"Unknown combination method: {self.config['combination_method']}")
        return self.action_adverb_combined_proj(combined)

    def get_video_features(self, flow, rgb):
        video_input = torch.cat((flow, rgb), dim=-1)
        video_features = self.video_encoder(video_input)
        action_features_video = self.action_encoder_video(video_features)
        adverb_features_video = self.adverb_encoder_video(video_features)
        return video_features, action_features_video, adverb_features_video

    def get_text_features(self, actions, adverbs, action_parents, adverb_parents):
        action_indices = torch.LongTensor([self.action_2_idx[a.lower()] for a in actions]).to(self.config['device'])
        adverb_indices = torch.LongTensor([self.adverb_2_idx[a.lower()] for a in adverbs]).to(self.config['device'])
        action_parent_indices = torch.LongTensor([self.action_parent_2_idx[a.lower()] for a in action_parents]).to(self.config['device'])
        adverb_parent_indices = torch.LongTensor([self.adverb_parent_2_idx[a.lower()] for a in adverb_parents]).to(self.config['device'])

        action_embeds = self.get_action_embeddings(action_indices)
        adverb_embeds = self.get_adverb_embeddings(adverb_indices)
        action_parent_embeds = self.get_action_parent_embeddings(action_parent_indices)
        adverb_parent_embeds = self.get_adverb_parent_embeddings(adverb_parent_indices)

        return action_embeds, adverb_embeds, action_parent_embeds, adverb_parent_embeds
    
    def forward(self, batch, mode='train'):
        if mode == 'train':
            video_features, action_features_video, adverb_features_video = self.get_video_features(batch['flow'], batch['rgb'])
            action_embeds, adverb_embeds, action_parent_embeds, adverb_parent_embeds = self.get_text_features(
                batch['actions'], batch['adverbs'], batch['action_parents'], batch['adverb_parents'])

            combined_action_adverb = self.combine_action_adverb_embeddings(action_embeds, adverb_embeds)

            return {
                'video_features': video_features,
                'action_parent_embeds': action_parent_embeds,
                'adverb_parent_embeds': adverb_parent_embeds,
                'action_features_video': action_features_video,
                'adverb_features_video': adverb_features_video,
                'combined_action_adverb': combined_action_adverb,
            }
        else:
            # For evaluation, get video features and all possible text embeddings
            video_features, action_features_video, adverb_features_video = self.get_video_features(batch['flow'], batch['rgb'])

            # Get all action and adverb embeddings
            all_action_indices = torch.arange(len(self.action_vocab)).to(self.config['device'])
            all_adverb_indices = torch.arange(len(self.adverb_vocab)).to(self.config['device'])

            # Get embeddings for all actions and adverbs (num_actions, output_dim) and (num_adverbs, output_dim)
            all_action_embeds = self.get_action_embeddings(all_action_indices)
            all_adverb_embeds = self.get_adverb_embeddings(all_adverb_indices)

            # Create all possible action-adverb combinations
            # all_action_embeds: (num_actions, output_dim)
            # all_adverb_embeds: (num_adverbs, output_dim)
            # We need to create (num_actions * num_adverbs, output_dim) combinations
            num_actions = len(self.action_vocab)
            num_adverbs = len(self.adverb_vocab)

            # Expand to create all combinations
            # action_embeds_expanded: (num_actions, num_adverbs, output_dim)
            action_embeds_expanded = all_action_embeds.unsqueeze(1).expand(num_actions, num_adverbs, -1)
            # adverb_embeds_expanded: (num_actions, num_adverbs, output_dim)
            adverb_embeds_expanded = all_adverb_embeds.unsqueeze(0).expand(num_actions, num_adverbs, -1)

            # Flatten to (num_actions * num_adverbs, output_dim)
            action_embeds_flat = action_embeds_expanded.reshape(-1, self.config['output_dim'])
            adverb_embeds_flat = adverb_embeds_expanded.reshape(-1, self.config['output_dim'])

            # Combine all action-adverb pairs
            all_combined_action_adverb = self.combine_action_adverb_embeddings(action_embeds_flat, adverb_embeds_flat)

            return {
                'video_features': video_features,  # (batch_size, output_dim)
                'action_features_video': action_features_video,  # (batch_size, output_dim)
                'adverb_features_video': adverb_features_video,  # (batch_size, output_dim)
                'all_action_embeds': all_action_embeds,  # (num_actions, output_dim)
                'all_adverb_embeds': all_adverb_embeds,  # (num_adverbs, output_dim)
                'all_combined_action_adverb': all_combined_action_adverb,  # (num_actions * num_adverbs, output_dim)
            }


class HyperbolicTrainerModule(nn.Module):
    def __init__(self, model: Model, config):
        super(HyperbolicTrainerModule, self).__init__()
        self.model = model

        curvature = Curvature(config['curvature'], requires_grad=config.get('learnable_curvature', False))
        self.manifold = PoincareBall(c=curvature)
        self.poincare_ops = PoincareOperations(
            curvature=config['curvature'],
            dim=config['output_dim'],
            learnable_curvature=config.get('learnable_curvature', False),
            device='cuda',  # Will be updated in forward
            debug=config.get('debug', False)
        )
        # Define loss functions and other training components here
        pass
    
    def forward(self, batch, mode='train'):

        if mode == 'train':
            outputs_euclidean = self.model(batch)
            # Map features to hyperbolic space
            video_features_hyp = self.poincare_ops.exp_map(outputs_euclidean['video_features'])
            action_parent_embeds_hyp = self.poincare_ops.exp_map(outputs_euclidean['action_parent_embeds'])
            adverb_parent_embeds_hyp = self.poincare_ops.exp_map(outputs_euclidean['adverb_parent_embeds'])
            action_features_video_hyp = self.poincare_ops.exp_map(outputs_euclidean['action_features_video'])
            adverb_features_video_hyp = self.poincare_ops.exp_map(outputs_euclidean['adverb_features_video'])
            combined_action_adverb_hyp = self.poincare_ops.exp_map(outputs_euclidean['combined_action_adverb'])
            return {
                'video_features_hyp': video_features_hyp,
                'action_parent_embeds_hyp': action_parent_embeds_hyp,
                'adverb_parent_embeds_hyp': adverb_parent_embeds_hyp,
                'action_features_video_hyp': action_features_video_hyp,
                'adverb_features_video_hyp': adverb_features_video_hyp,
                'combined_action_adverb_hyp': combined_action_adverb_hyp,
            }
        else:
            # Evaluation mode: compute similarity scores
            outputs_euclidean = self.model(batch, mode='eval')

            # Map all features to hyperbolic space
            action_features_video_hyp = self.poincare_ops.exp_map(outputs_euclidean['action_features_video'])  # (batch_size, output_dim)
            adverb_features_video_hyp = self.poincare_ops.exp_map(outputs_euclidean['adverb_features_video'])  # (batch_size, output_dim)
            video_features_hyp = self.poincare_ops.exp_map(outputs_euclidean['video_features'])  # (batch_size, output_dim)

            all_action_embeds_hyp = self.poincare_ops.exp_map(outputs_euclidean['all_action_embeds'])  # (num_actions, output_dim)
            all_adverb_embeds_hyp = self.poincare_ops.exp_map(outputs_euclidean['all_adverb_embeds'])  # (num_adverbs, output_dim)
            all_combined_hyp = self.poincare_ops.exp_map(outputs_euclidean['all_combined_action_adverb'])  # (num_actions * num_adverbs, output_dim)

            # Compute action probabilities: similarity between action_features_video and all_action_embeds
            # action_features_video_hyp: (batch_size, output_dim)
            # all_action_embeds_hyp: (num_actions, output_dim)
            # We need to compute distance for each batch item against all actions
            action_sim = self.compute_batch_similarity(action_features_video_hyp, all_action_embeds_hyp)  # (batch_size, num_actions)
            action_probs = torch.softmax(action_sim, dim=1)  # (batch_size, num_actions)

            # Compute adverb probabilities: similarity between adverb_features_video and all_adverb_embeds
            adverb_sim = self.compute_batch_similarity(adverb_features_video_hyp, all_adverb_embeds_hyp)  # (batch_size, num_adverbs)
            adverb_probs = torch.softmax(adverb_sim, dim=1)  # (batch_size, num_adverbs)

            # Compute composition probabilities: similarity between video_features and all_combined
            composition_sim = self.compute_batch_similarity(video_features_hyp, all_combined_hyp)  # (batch_size, num_actions * num_adverbs)
            composition_probs = torch.softmax(composition_sim, dim=1)  # (batch_size, num_actions * num_adverbs)

            return {
                'action_probs': action_probs,  # (batch_size, num_actions)
                'adverb_probs': adverb_probs,  # (batch_size, num_adverbs)
                'composition_probs': composition_probs,  # (batch_size, num_actions * num_adverbs)
                'action_sim': action_sim,  # (batch_size, num_actions)
                'adverb_sim': adverb_sim,  # (batch_size, num_adverbs)
                'composition_sim': composition_sim,  # (batch_size, num_actions * num_adverbs)
            }

    def compute_batch_similarity(self, video_features, text_features):
        """
        Compute similarity between video features and all text features in a batched manner.

        Args:
            video_features: (batch_size, output_dim) - ManifoldTensor
            text_features: (num_classes, output_dim) - ManifoldTensor

        Returns:
            similarity: (batch_size, num_classes) - negative distances
        """
        batch_size = video_features.shape[0]
        num_classes = text_features.shape[0]

        # Expand dimensions for broadcasting
        # video_features: (batch_size, 1, output_dim)
        video_expanded = video_features.tensor.unsqueeze(1)  # Get underlying tensor
        # text_features: (1, num_classes, output_dim)
        text_expanded = text_features.tensor.unsqueeze(0)  # Get underlying tensor

        # Repeat to create (batch_size, num_classes, output_dim)
        video_repeated = video_expanded.expand(batch_size, num_classes, -1)
        text_repeated = text_expanded.expand(batch_size, num_classes, -1)

        # Reshape to (batch_size * num_classes, output_dim) for distance computation
        video_flat = video_repeated.reshape(-1, video_features.shape[-1])
        text_flat = text_repeated.reshape(-1, text_features.shape[-1])

        # Convert to ManifoldTensor for distance computation
        video_flat_manifold = ManifoldTensor(data=video_flat, man_dim=1, manifold=self.manifold)
        text_flat_manifold = ManifoldTensor(data=text_flat, man_dim=1, manifold=self.manifold)

        # Compute distances
        distances = self.manifold.dist(video_flat_manifold, text_flat_manifold)  # (batch_size * num_classes,)

        # Reshape back to (batch_size, num_classes)
        distances = distances.reshape(batch_size, num_classes)

        # Return negative distance as similarity
        return -distances

    def predict(self, batch, action_vocab, adverb_vocab):
        """
        Predict action and adverb using the Troika formula from the paper.

        Formula from paper (Equation 1):
        p̃(c_{i,j}|x) = p(c_{i,j}|x) + p(action_i|x) * p(adverb_j|x)

        Args:
            batch: Input batch with video features
            action_vocab: List of action names
            adverb_vocab: List of adverb names

        Returns:
            Dictionary containing:
                - predicted_actions: (batch_size,) tensor of action indices
                - predicted_adverbs: (batch_size,) tensor of adverb indices
                - predicted_action_names: List of predicted action names
                - predicted_adverb_names: List of predicted adverb names
                - combined_probs: (batch_size, num_actions * num_adverbs) combined probabilities
        """
        # Get eval outputs
        outputs = self.forward(batch, mode='eval')

        action_probs = outputs['action_probs']  # (batch_size, num_actions)
        adverb_probs = outputs['adverb_probs']  # (batch_size, num_adverbs)
        composition_probs = outputs['composition_probs']  # (batch_size, num_actions * num_adverbs)

        batch_size = action_probs.shape[0]
        num_actions = action_probs.shape[1]
        num_adverbs = adverb_probs.shape[1]

        # Compute the product p(action_i|x) * p(adverb_j|x) for all combinations
        # action_probs: (batch_size, num_actions) -> (batch_size, num_actions, 1)
        # adverb_probs: (batch_size, num_adverbs) -> (batch_size, 1, num_adverbs)
        action_probs_expanded = action_probs.unsqueeze(2)  # (batch_size, num_actions, 1)
        adverb_probs_expanded = adverb_probs.unsqueeze(1)  # (batch_size, 1, num_adverbs)

        # Outer product: (batch_size, num_actions, num_adverbs)
        action_adverb_product = action_probs_expanded * adverb_probs_expanded

        # Flatten to (batch_size, num_actions * num_adverbs)
        action_adverb_product_flat = action_adverb_product.reshape(batch_size, -1)

        # Apply Troika formula: p̃(c_{i,j}|x) = p(c_{i,j}|x) + p(action_i|x) * p(adverb_j|x)
        combined_probs = composition_probs + action_adverb_product_flat

        # Get predictions: argmax over all compositions
        predicted_composition_indices = torch.argmax(combined_probs, dim=1)  # (batch_size,)

        # Convert flat composition indices to (action_idx, adverb_idx)
        predicted_action_indices = predicted_composition_indices // num_adverbs
        predicted_adverb_indices = predicted_composition_indices % num_adverbs

        # Convert to action and adverb names
        predicted_action_names = [action_vocab[idx.item()] for idx in predicted_action_indices]
        predicted_adverb_names = [adverb_vocab[idx.item()] for idx in predicted_adverb_indices]

        return {
            'predicted_actions': predicted_action_indices,  # (batch_size,)
            'predicted_adverbs': predicted_adverb_indices,  # (batch_size,)
            'predicted_action_names': predicted_action_names,  # List of strings
            'predicted_adverb_names': predicted_adverb_names,  # List of strings
            'combined_probs': combined_probs,  # (batch_size, num_actions * num_adverbs)
            'action_probs': action_probs,  # (batch_size, num_actions)
            'adverb_probs': adverb_probs,  # (batch_size, num_adverbs)
        }
    
    def compute_similarity(self, v_emb, t_emb):
        """Compute similarity as negative distance in Poincare space"""
        dist = self.manifold.dist(v_emb, t_emb)
        return -dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ============================================================================
# Loss Functions
# ============================================================================

class TaxonomicEntailmentLoss(nn.Module):
    """
    Enforces hierarchical structure using entailment cones in Poincare space
    Implements Eq. 9-10 from H²EM paper
    Adapted for Poincare ball (originally Lorentz model)
    """

    def __init__(self, manifold, gamma=0.1, curvature_value=1.0):
        super().__init__()
        self.manifold = manifold
        self.gamma = gamma  # Boundary condition parameter
        # Store curvature value directly
        self.c_val = curvature_value

    def entailment_cone_loss(self, parent, child):
        """
        Compute entailment loss for parent-child relationship
        Child should lie within parent's entailment cone

        Args:
            parent, child: Points on Poincare ball (B, D) - ManifoldTensor
        Returns:
            Loss (B,)
        """
        # Extract underlying tensors if ManifoldTensor
        if isinstance(parent, ManifoldTensor):
            parent_t = parent.tensor
        else:
            parent_t = parent

        if isinstance(child, ManifoldTensor):
            child_t = child.tensor
        else:
            child_t = child

        # Compute angle between parent and child from origin
        # In Poincare ball, we use the norm and distance
        parent_norm = torch.norm(parent_t, dim=-1, keepdim=True).clamp(min=1e-7)
        child_norm = torch.norm(child_t, dim=-1, keepdim=True).clamp(min=1e-7)

        # Cosine similarity in ambient space (approximation for small curvature)
        cos_angle = (parent_t * child_t).sum(dim=-1) / (parent_norm.squeeze() * child_norm.squeeze() + 1e-7)
        # Clamp to slightly tighter range to avoid numerical issues with acos
        cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Exterior angle to the cone
        angle = torch.acos(cos_angle)

        # Aperture of the cone (depends on parent's distance from origin)
        # Closer to origin = more general = wider cone
        c_sqrt = torch.sqrt(torch.tensor(self.c_val, device=parent_t.device))
        sin_arg = 2 * self.gamma / (c_sqrt * parent_norm.squeeze() + 1e-7)
        sin_arg = sin_arg.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        aperture = torch.arcsin(sin_arg)
        aperture = aperture.clamp(0, np.pi / 2)

        # Loss is positive if child is outside the cone
        loss = F.relu(angle - aperture)

        return loss

    def forward(self, v_action, v_adverb, v_comp, t_action, t_adverb, t_comp,
                t_action_parent=None, t_adverb_parent=None):
        """
        Compute full taxonomic entailment loss

        Conceptual Hierarchy:
            - composition ⊂ action
            - composition ⊂ adverb

        Semantic Hierarchy (if parent categories provided):
            - action ⊂ parent_action
            - adverb ⊂ parent_adverb
        """
        loss = 0.0

        # Conceptual hierarchy for visual modality
        loss += self.entailment_cone_loss(v_action, v_comp).mean()
        loss += self.entailment_cone_loss(v_adverb, v_comp).mean()

        # Conceptual hierarchy for text modality
        loss += self.entailment_cone_loss(t_action, t_comp).mean()
        loss += self.entailment_cone_loss(t_adverb, t_comp).mean()

        # Semantic hierarchy (if parent categories available)
        if t_action_parent is not None:
            loss += self.entailment_cone_loss(t_action_parent, t_action).mean()

        if t_adverb_parent is not None:
            loss += self.entailment_cone_loss(t_adverb_parent, t_adverb).mean()

        return loss


class DiscriminativeAlignmentLoss(nn.Module):
    """
    Contrastive loss with hard negative mining in hyperbolic space
    Implements Eq. 11 from H²EM paper
    """

    def __init__(self, manifold, temperature=0.07, hard_neg_weight=3.0):
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight

    def forward(self, v_comp, t_comp, action_labels, adverb_labels):
        """
        Args:
            v_comp: Visual composition embeddings (B, D)
            t_comp: Text composition embeddings (B, D)
            action_labels: Action labels (B,)
            adverb_labels: Adverb labels (B,)
        Returns:
            Contrastive loss
        """
        batch_size = v_comp.size(0)

        # Compute pairwise distances
        distances = self.manifold.dist(v_comp.unsqueeze(1), t_comp.unsqueeze(0))  # (B, B)

        # Convert to similarities (negative distance)
        logits = -distances / self.temperature

        # Create hard negative mask
        # Hard negatives: same action OR same adverb, but different composition
        action_match = action_labels.unsqueeze(1) == action_labels.unsqueeze(0)  # (B, B)
        adverb_match = adverb_labels.unsqueeze(1) == adverb_labels.unsqueeze(0)  # (B, B)
        hard_negatives = (action_match | adverb_match) & ~torch.eye(batch_size, device=logits.device).bool()

        # Positive mask (diagonal)
        positives = torch.eye(batch_size, device=logits.device).bool()

        # Weighted InfoNCE loss
        exp_logits = torch.exp(logits)

        # Weight hard negatives more
        weights = torch.ones_like(logits)
        weights[hard_negatives] = self.hard_neg_weight
        weighted_exp = exp_logits * weights

        # Compute loss
        log_prob = logits[positives] - torch.log(weighted_exp.sum(dim=1) + 1e-7)
        loss = -log_prob.mean()

        return loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, output_dir, resume_from=None, wandb_enabled=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = os.path.join(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.wandb_enabled = wandb_enabled

        # Use GPU if available and not disabled
        use_gpu = config.get('gpu', True) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Loss functions
        self.entailment_loss = TaxonomicEntailmentLoss(
            manifold=model.manifold,
            gamma=config.get('gamma', 0.1),
            curvature_value=config['curvature']
        )
        self.alignment_loss = DiscriminativeAlignmentLoss(
            manifold=model.manifold,
            temperature=config.get('temperature', 0.07),
            hard_neg_weight=config.get('hard_neg_weight', 3.0)
        )

        # Optimizer (Riemannian for hyperbolic parameters)
        self.optimizer = RiemannianAdam(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )

        # Loss weights
        self.beta1 = config.get('beta1', 1.0)  # Alignment loss
        self.beta2 = config.get('beta2', 0.1)  # Entailment loss
        self.beta3 = config.get('beta3', 0.5)  # Primitive loss

        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_action_acc = 0.0
        self.best_adverb_acc = 0.0

        # Watch model with wandb
        if self.wandb_enabled:
            wandb.watch(model, log='all', log_freq=100, log_graph=True)

        if resume_from:
            self.load_checkpoint(resume_from)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_align_loss = 0.0
        epoch_entail_loss = 0.0
        epoch_action_loss = 0.0
        epoch_adverb_loss = 0.0

        # Timing meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        grad_norm = AverageMeter()
        start = time.time()

        # Global step counter for more granular logging
        global_step = self.current_epoch * len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch_dict in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - start)

            # Convert batch dict from ActionAdverbDataset format to model format
            # Dataset returns: flow_features, rgb_features, action, adverb, action_parent, adverb_parent
            flow = torch.from_numpy(np.stack([batch_dict['flow_features'][i] for i in range(len(batch_dict['flow_features']))])).float().to(self.device)
            rgb = torch.from_numpy(np.stack([batch_dict['rgb_features'][i] for i in range(len(batch_dict['rgb_features']))])).float().to(self.device)
            actions = batch_dict['action']  # List of action strings
            adverbs = batch_dict['adverb']  # List of adverb strings
            action_parents = batch_dict['action_parent']  # List of action parent strings
            adverb_parents = batch_dict['adverb_parent']  # List of adverb parent strings

            if self.config.get('debug', False):
                print(f"\n[Batch]")
                print(f"  Flow: {flow.shape}, RGB: {rgb.shape}")
                print(f"  Actions: {actions}")
                print(f"  Adverbs: {adverbs}")

            # Create batch dict for model
            model_batch = {
                'flow': flow,
                'rgb': rgb,
                'actions': actions,
                'adverbs': adverbs,
                'action_parents': action_parents,
                'adverb_parents': adverb_parents
            }

            # Forward pass
            outputs = self.model(model_batch, mode='train')

            # For loss computation, we need action and adverb labels as indices
            # Convert string labels to indices
            action_labels = torch.LongTensor([self.model.model.action_2_idx[a.lower()] for a in actions]).to(self.device)
            adverb_labels = torch.LongTensor([self.model.model.adverb_2_idx[a.lower()] for a in adverbs]).to(self.device)

            # Compute losses
            # 1. Discriminative Alignment Loss
            loss_align = self.alignment_loss(
                outputs['combined_action_adverb_hyp'],
                outputs['combined_action_adverb_hyp'],  # Same embedding for now
                action_labels,
                adverb_labels
            )

            # 2. Taxonomic Entailment Loss
            loss_entail = self.entailment_loss(
                outputs['action_features_video_hyp'],
                outputs['adverb_features_video_hyp'],
                outputs['video_features_hyp'],
                outputs['action_parent_embeds_hyp'],
                outputs['adverb_parent_embeds_hyp'],
                outputs['combined_action_adverb_hyp'],
                outputs['action_parent_embeds_hyp'],
                outputs['adverb_parent_embeds_hyp']
            )

            # 3. Primitive Auxiliary Losses
            # Action: encourage video action to match text action parent (diagonal)
            action_dist = self.model.manifold.dist(outputs['action_features_video_hyp'], outputs['action_parent_embeds_hyp'])
            loss_action = action_dist.mean()

            # Adverb: encourage video adverb to match text adverb parent (diagonal)
            adverb_dist = self.model.manifold.dist(outputs['adverb_features_video_hyp'], outputs['adverb_parent_embeds_hyp'])
            loss_adverb = adverb_dist.mean()

            # Total loss
            loss = (self.beta1 * loss_align +
                   self.beta2 * loss_entail +
                   self.beta3 * (loss_action + loss_adverb))

            if self.config.get('debug', False):
                print(f"  Losses:")
                print(f"    Total: {loss.item():.4f}")
                print(f"    Align: {loss_align.item():.4f}")
                print(f"    Entail: {loss_entail.item():.4f}")
                print(f"    Action: {loss_action.item():.4f}")
                print(f"    Adverb: {loss_adverb.item():.4f}")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Calculate gradient norm before clipping
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm.update(total_norm)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_align_loss += loss_align.item()
            epoch_entail_loss += loss_entail.item()
            epoch_action_loss += loss_action.item()
            epoch_adverb_loss += loss_adverb.item()

            # Measure batch processing time
            batch_time.update(time.time() - start - data_time.val)

            pbar.set_postfix({
                'loss': loss.item(),
                'l_align': loss_align.item(),
                'l_entail': loss_entail.item(),
                'grad': total_norm,
            })

            # Log step-level metrics to wandb
            if self.wandb_enabled and batch_idx % 10 == 0:  # Log every 10 batches
                step_metrics = {
                    'step': global_step + batch_idx,
                    'step/loss_total': loss.item(),
                    'step/loss_action': loss_action.item(),
                    'step/loss_adverb': loss_adverb.item(),
                    'step/loss_align': loss_align.item(),
                    'step/loss_entail': loss_entail.item(),
                    'step/grad_norm': total_norm,
                    'step/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step/batch_time': batch_time.val,
                    'step/data_time': data_time.val,
                }

                # Log curvature if learnable
                if self.config.get('learnable_curvature', False):
                    step_metrics['step/curvature'] = self.model.poincare_ops.c.c.item()

                wandb.log(step_metrics)

            start = time.time()

        # Calculate average losses
        avg_loss = epoch_loss / len(self.train_loader)
        avg_align_loss = epoch_align_loss / len(self.train_loader)
        avg_entail_loss = epoch_entail_loss / len(self.train_loader)
        avg_action_loss = epoch_action_loss / len(self.train_loader)
        avg_adverb_loss = epoch_adverb_loss / len(self.train_loader)

        self.train_losses.append(avg_loss)

        # Log epoch-level training metrics to wandb
        if self.wandb_enabled:
            epoch_metrics = {
                'epoch': self.current_epoch,
                'train/loss_total': avg_loss,
                'train/loss_action': avg_action_loss,
                'train/loss_adverb': avg_adverb_loss,
                'train/loss_align': avg_align_loss,
                'train/loss_entail': avg_entail_loss,
                'train/batch_time': batch_time.avg,
                'train/data_time': data_time.avg,
                'train/grad_norm': grad_norm.avg,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
            }

            # Log curvature if learnable
            if self.config.get('learnable_curvature', False):
                epoch_metrics['train/curvature'] = self.model.poincare_ops.c.c.item()

            # Log loss component ratios
            epoch_metrics['train/align_ratio'] = avg_align_loss / (avg_loss + 1e-8)
            epoch_metrics['train/entail_ratio'] = avg_entail_loss / (avg_loss + 1e-8)
            epoch_metrics['train/primitive_ratio'] = (avg_action_loss + avg_adverb_loss) / (avg_loss + 1e-8)

            wandb.log(epoch_metrics)

        print('E: %d | L: %.2E | L_align: %.2E | L_entail: %.2E | L_act: %.2E | L_adv: %.2E | Grad: %.2E | LR: %.2E' %
              (self.current_epoch, avg_loss, avg_align_loss, avg_entail_loss,
               avg_action_loss, avg_adverb_loss, grad_norm.avg, self.optimizer.param_groups[0]['lr']))

        return avg_loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_align_loss = 0.0
        val_entail_loss = 0.0
        val_action_loss = 0.0
        val_adverb_loss = 0.0

        # For accuracy computation
        all_action_preds = []
        all_adverb_preds = []
        all_action_gts = []
        all_adverb_gts = []
        all_action_probs = []
        all_adverb_probs = []

        # Get all possible actions and adverbs for evaluation
        action_vocab = self.config['action_vocab']
        adverb_vocab = self.config['adverb_vocab']

        with torch.no_grad():
            for batch_dict in tqdm(self.val_loader, desc='Validation'):
                # Convert batch dict from ActionAdverbDataset format to model format
                flow = torch.from_numpy(np.stack([batch_dict['flow_features'][i] for i in range(len(batch_dict['flow_features']))])).float().to(self.device)
                rgb = torch.from_numpy(np.stack([batch_dict['rgb_features'][i] for i in range(len(batch_dict['rgb_features']))])).float().to(self.device)
                actions = batch_dict['action']
                adverbs = batch_dict['adverb']
                action_parents = batch_dict['action_parent']
                adverb_parents = batch_dict['adverb_parent']

                # Create batch dict for model (train mode for loss computation)
                model_batch = {
                    'flow': flow,
                    'rgb': rgb,
                    'actions': actions,
                    'adverbs': adverbs,
                    'action_parents': action_parents,
                    'adverb_parents': adverb_parents
                }

                # Forward pass for loss computation
                outputs_train = self.model(model_batch, mode='train')

                # Convert string labels to indices
                action_labels = torch.LongTensor([self.model.model.action_2_idx[a.lower()] for a in actions]).to(self.device)
                adverb_labels = torch.LongTensor([self.model.model.adverb_2_idx[a.lower()] for a in adverbs]).to(self.device)

                # Compute losses
                loss_align = self.alignment_loss(
                    outputs_train['combined_action_adverb_hyp'],
                    outputs_train['combined_action_adverb_hyp'],
                    action_labels,
                    adverb_labels
                )

                loss_entail = self.entailment_loss(
                    outputs_train['action_features_video_hyp'],
                    outputs_train['adverb_features_video_hyp'],
                    outputs_train['video_features_hyp'],
                    outputs_train['action_parent_embeds_hyp'],
                    outputs_train['adverb_parent_embeds_hyp'],
                    outputs_train['combined_action_adverb_hyp'],
                    outputs_train['action_parent_embeds_hyp'],
                    outputs_train['adverb_parent_embeds_hyp']
                )

                # Primitive losses
                action_dist = self.model.manifold.dist(outputs_train['action_features_video_hyp'], outputs_train['action_parent_embeds_hyp'])
                loss_action = action_dist.mean()
                adverb_dist = self.model.manifold.dist(outputs_train['adverb_features_video_hyp'], outputs_train['adverb_parent_embeds_hyp'])
                loss_adverb = adverb_dist.mean()

                loss = (self.beta1 * loss_align + self.beta2 * loss_entail +
                        self.beta3 * (loss_action + loss_adverb))
                val_loss += loss.item()
                val_align_loss += loss_align.item()
                val_entail_loss += loss_entail.item()
                val_action_loss += loss_action.item()
                val_adverb_loss += loss_adverb.item()

                # Use predict method to get action and adverb predictions
                predictions = self.model.predict(model_batch, action_vocab, adverb_vocab)

                # Store predictions and ground truth
                action_preds = predictions['predicted_actions']
                adverb_preds = predictions['predicted_adverbs']
                action_probs = predictions['action_probs']
                adverb_probs = predictions['adverb_probs']

                all_action_preds.extend(action_preds.cpu().numpy())
                all_adverb_preds.extend(adverb_preds.cpu().numpy())
                all_action_gts.extend(action_labels.cpu().numpy())
                all_adverb_gts.extend(adverb_labels.cpu().numpy())
                all_action_probs.append(action_probs.cpu().numpy())
                all_adverb_probs.append(adverb_probs.cpu().numpy())

        # Calculate metrics
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_align_loss = val_align_loss / len(self.val_loader)
        avg_val_entail_loss = val_entail_loss / len(self.val_loader)
        avg_val_action_loss = val_action_loss / len(self.val_loader)
        avg_val_adverb_loss = val_adverb_loss / len(self.val_loader)

        # Calculate accuracies
        all_action_preds = np.array(all_action_preds)
        all_action_gts = np.array(all_action_gts)
        all_adverb_preds = np.array(all_adverb_preds)
        all_adverb_gts = np.array(all_adverb_gts)
        all_action_probs = np.concatenate(all_action_probs, axis=0)  # (N, num_actions)
        all_adverb_probs = np.concatenate(all_adverb_probs, axis=0)  # (N, num_adverbs)

        # Top-1 accuracy
        action_acc = (all_action_preds == all_action_gts).mean()
        adverb_acc = (all_adverb_preds == all_adverb_gts).mean()

        # Top-5 accuracy
        action_top5_preds = np.argsort(all_action_probs, axis=1)[:, -5:]
        adverb_top5_preds = np.argsort(all_adverb_probs, axis=1)[:, -5:]
        action_top5_acc = np.mean([gt in preds for gt, preds in zip(all_action_gts, action_top5_preds)])
        adverb_top5_acc = np.mean([gt in preds for gt, preds in zip(all_adverb_gts, adverb_top5_preds)])

        # Per-class accuracies (mean P@1)
        action_per_class_accs = []
        for action_idx in np.unique(all_action_gts):
            mask = all_action_gts == action_idx
            if mask.sum() > 0:
                acc = (all_action_preds[mask] == all_action_gts[mask]).mean()
                action_per_class_accs.append(acc)
        action_mean_acc = np.mean(action_per_class_accs) if action_per_class_accs else 0.0

        adverb_per_class_accs = []
        for adverb_idx in np.unique(all_adverb_gts):
            mask = all_adverb_gts == adverb_idx
            if mask.sum() > 0:
                acc = (all_adverb_preds[mask] == all_adverb_gts[mask]).mean()
                adverb_per_class_accs.append(acc)
        adverb_mean_acc = np.mean(adverb_per_class_accs) if adverb_per_class_accs else 0.0

        # Calculate confusion statistics
        action_correct_per_class = {}
        action_total_per_class = {}
        for gt, pred in zip(all_action_gts, all_action_preds):
            if gt not in action_total_per_class:
                action_total_per_class[gt] = 0
                action_correct_per_class[gt] = 0
            action_total_per_class[gt] += 1
            if gt == pred:
                action_correct_per_class[gt] += 1

        adverb_correct_per_class = {}
        adverb_total_per_class = {}
        for gt, pred in zip(all_adverb_gts, all_adverb_preds):
            if gt not in adverb_total_per_class:
                adverb_total_per_class[gt] = 0
                adverb_correct_per_class[gt] = 0
            adverb_total_per_class[gt] += 1
            if gt == pred:
                adverb_correct_per_class[gt] += 1

        self.val_losses.append(avg_val_loss)

        # Print validation results
        print('E %d | Val Loss: %.4f | Action P@1: %.3f (Top5: %.3f) | Adverb P@1: %.3f (Top5: %.3f) | Action Mean: %.3f | Adverb Mean: %.3f' %
              (self.current_epoch, avg_val_loss, action_acc, action_top5_acc, adverb_acc, adverb_top5_acc,
               action_mean_acc, adverb_mean_acc))

        # Log validation metrics to wandb
        if self.wandb_enabled:
            val_metrics = {
                'epoch': self.current_epoch,
                'test/loss_total': avg_val_loss,
                'test/loss_align': avg_val_align_loss,
                'test/loss_entail': avg_val_entail_loss,
                'test/loss_action': avg_val_action_loss,
                'test/loss_adverb': avg_val_adverb_loss,
                'test/action_acc': action_acc,
                'test/adverb_acc': adverb_acc,
                'test/action_top5_acc': action_top5_acc,
                'test/adverb_top5_acc': adverb_top5_acc,
                'test/action_mean_acc': action_mean_acc,
                'test/adverb_mean_acc': adverb_mean_acc,
                'test/combined_acc': (action_acc + adverb_acc) / 2,
            }

            # Log per-class statistics
            val_metrics['test/action_num_classes'] = len(action_per_class_accs)
            val_metrics['test/adverb_num_classes'] = len(adverb_per_class_accs)
            val_metrics['test/action_acc_std'] = np.std(action_per_class_accs) if action_per_class_accs else 0.0
            val_metrics['test/adverb_acc_std'] = np.std(adverb_per_class_accs) if adverb_per_class_accs else 0.0

            # Log worst and best performing classes
            if action_per_class_accs:
                val_metrics['test/action_min_acc'] = np.min(action_per_class_accs)
                val_metrics['test/action_max_acc'] = np.max(action_per_class_accs)
            if adverb_per_class_accs:
                val_metrics['test/adverb_min_acc'] = np.min(adverb_per_class_accs)
                val_metrics['test/adverb_max_acc'] = np.max(adverb_per_class_accs)

            wandb.log(val_metrics)

            # Create and log confusion matrix as wandb Table (for first few epochs or best model)
            if self.current_epoch % 10 == 0 or action_acc > self.best_action_acc:
                # Create per-class accuracy table
                action_class_data = []
                for class_idx in sorted(action_total_per_class.keys()):
                    class_name = action_vocab[class_idx] if class_idx < len(action_vocab) else f"class_{class_idx}"
                    acc = action_correct_per_class[class_idx] / action_total_per_class[class_idx]
                    action_class_data.append([class_name, acc, action_total_per_class[class_idx]])

                wandb.log({
                    f"test/action_per_class_epoch_{self.current_epoch}": wandb.Table(
                        columns=["Action", "Accuracy", "Count"],
                        data=action_class_data
                    )
                })

                adverb_class_data = []
                for class_idx in sorted(adverb_total_per_class.keys()):
                    class_name = adverb_vocab[class_idx] if class_idx < len(adverb_vocab) else f"class_{class_idx}"
                    acc = adverb_correct_per_class[class_idx] / adverb_total_per_class[class_idx]
                    adverb_class_data.append([class_name, acc, adverb_total_per_class[class_idx]])

                wandb.log({
                    f"test/adverb_per_class_epoch_{self.current_epoch}": wandb.Table(
                        columns=["Adverb", "Accuracy", "Count"],
                        data=adverb_class_data
                    )
                })

                # Log embedding norms (hyperbolic distance from origin)
                with torch.no_grad():
                    # Get all text embeddings
                    all_action_indices = torch.arange(len(action_vocab)).to(self.device)
                    all_adverb_indices = torch.arange(len(adverb_vocab)).to(self.device)

                    # Get embeddings in hyperbolic space
                    all_action_embeds = self.model.model.get_action_embeddings(all_action_indices)
                    all_adverb_embeds = self.model.model.get_adverb_embeddings(all_adverb_indices)

                    # Map to hyperbolic space
                    all_action_embeds_hyp = self.model.poincare_ops.exp_map(all_action_embeds)
                    all_adverb_embeds_hyp = self.model.poincare_ops.exp_map(all_adverb_embeds)

                    # Calculate norms (distance from origin in Poincare ball)
                    action_norms = torch.norm(all_action_embeds_hyp.tensor, dim=-1).cpu().numpy()
                    adverb_norms = torch.norm(all_adverb_embeds_hyp.tensor, dim=-1).cpu().numpy()

                    wandb.log({
                        'embeddings/action_norm_mean': action_norms.mean(),
                        'embeddings/action_norm_std': action_norms.std(),
                        'embeddings/action_norm_max': action_norms.max(),
                        'embeddings/adverb_norm_mean': adverb_norms.mean(),
                        'embeddings/adverb_norm_std': adverb_norms.std(),
                        'embeddings/adverb_norm_max': adverb_norms.max(),
                        'embeddings/action_norm_hist': wandb.Histogram(action_norms),
                        'embeddings/adverb_norm_hist': wandb.Histogram(adverb_norms),
                    })

        # Update best metrics
        if action_acc > self.best_action_acc:
            self.best_action_acc = action_acc
        if adverb_acc > self.best_adverb_acc:
            self.best_adverb_acc = adverb_acc

        return avg_val_loss

    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_one_epoch()

            # Validate every epoch for comprehensive tracking
            val_loss = self.validate()

            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            print(f'\n{"="*80}')
            print(f'Epoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.current_epoch})')
            print(f'  Best Action Acc: {self.best_action_acc:.4f} | Best Adverb Acc: {self.best_adverb_acc:.4f}')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            print(f'{"="*80}\n')

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')

                # Log best model to wandb
                if self.wandb_enabled:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_action_acc'] = self.best_action_acc
                    wandb.run.summary['best_adverb_acc'] = self.best_adverb_acc

            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_freq', 100) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

            # Log epoch timing
            if self.wandb_enabled:
                wandb.log({
                    'epoch': epoch,
                    'timing/epoch_time': epoch_time,
                    'timing/samples_per_second': len(self.train_loader.dataset) / epoch_time,
                })

            # Step scheduler
            self.scheduler.step()

        # Final summary
        print(f'\n{"="*80}')
        print(f'Training Complete!')
        print(f'  Best Val Loss: {self.best_val_loss:.4f}')
        print(f'  Best Action Acc: {self.best_action_acc:.4f}')
        print(f'  Best Adverb Acc: {self.best_adverb_acc:.4f}')
        print(f'{"="*80}\n')

        if self.wandb_enabled:
            wandb.run.summary['training_complete'] = True
            wandb.run.summary['total_epochs'] = num_epochs

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_action_acc': self.best_action_acc,
            'best_adverb_acc': self.best_adverb_acc,
            'config': self.config
        }
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint: {filename}')

        # Save to wandb as artifact (only for best model to save space)
        if self.wandb_enabled and filename == 'best_model.pth':
            artifact = wandb.Artifact(
                name=f'model-{wandb.run.id}',
                type='model',
                description=f'Best model at epoch {self.current_epoch} with val_loss={self.best_val_loss:.4f}, '
                           f'action_acc={self.best_action_acc:.4f}, adverb_acc={self.best_adverb_acc:.4f}',
                metadata={
                    'epoch': self.current_epoch,
                    'val_loss': self.best_val_loss,
                    'action_acc': self.best_action_acc,
                    'adverb_acc': self.best_adverb_acc,
                }
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f'Logged best model to WandB as artifact')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_action_acc = checkpoint.get('best_action_acc', 0.0)
        self.best_adverb_acc = checkpoint.get('best_adverb_acc', 0.0)
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
        print(f'  Best Val Loss: {self.best_val_loss:.4f}')
        print(f'  Best Action Acc: {self.best_action_acc:.4f}')
        print(f'  Best Adverb Acc: {self.best_adverb_acc:.4f}')



def main():
    '''
    Load the hierarchy
    Load the dataset
    Create data loaders
    define the model architecture
    define the losses
    train the model
    add metrics
    add in logging
    '''
    # Create argument parser
    parser = argparse.ArgumentParser(description='Hyperbolic Action-Adverb Training')

    # Data arguments
    parser.add_argument('--hierarchy_path', type=str, default='datasets/VATEX_Adverbs/action_adverb_hierarchy.json',
                        help='Path to the action-adverb hierarchy JSON file')
    parser.add_argument('--data-dir', type=str, default='datasets/VATEX_Adverbs',
                        help='Directory containing train.csv and val.csv')
    parser.add_argument('--features-dir', type=str, default='datasets/VATEX_Adverbs/features/',
                        help='Directory containing feature files')
    parser.add_argument('--glove-path', type=str, default='datasets/glove.6B.300d.txt',
                        help='Path to GloVe embeddings file')

    # Model architecture arguments
    parser.add_argument('--input-dim', type=int, default=4096,
                        help='Input dimension (flow_dim + rgb_dim)')
    parser.add_argument('--flow-dim', type=int, default=1024,
                        help='Flow feature dimension')
    parser.add_argument('--rgb-dim', type=int, default=1024,
                        help='RGB feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='Hidden layer dimension')
    parser.add_argument('--bottleneck-dim', type=int, default=512,
                        help='Bottleneck dimension for action/adverb encoders')
    parser.add_argument('--output-dim', type=int, default=300,
                        help='Output embedding dimension')
    parser.add_argument('--glove-dim', type=int, default=300,
                        help='GloVe embedding dimension')
    parser.add_argument('--combination-method', type=str, default='concat', choices=['concat', 'mean'],
                        help='Method to combine action and adverb embeddings')

    # Hyperbolic space arguments
    parser.add_argument('--curvature', type=float, default=1.0,
                        help='Curvature of Poincare ball')
    parser.add_argument('--learnable-curvature', action='store_true',
                        help='Make curvature a learnable parameter')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')

    # Loss weight arguments
    parser.add_argument('--beta1', type=float, default=1.0,
                        help='Weight for alignment loss')
    parser.add_argument('--beta2', type=float, default=0.1,
                        help='Weight for entailment loss')
    parser.add_argument('--beta3', type=float, default=0.5,
                        help='Weight for primitive loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Boundary condition parameter for entailment cones')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--hard-neg-weight', type=float, default=3.0,
                        help='Weight for hard negatives in contrastive loss')

    # Evaluation and checkpointing
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='Evaluate every N epochs')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save checkpoint every N epochs')

    # System arguments
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                        help='Force CPU usage')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader workers (0=no multiprocessing, recommended to avoid pickle errors)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Wandb arguments
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='hyperbolic-action-adverb',
                        help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Wandb run name')

    args = parser.parse_args()

    # Create config dict from args
    config = {
        'input_dim': args.input_dim,
        'flow_dim': args.flow_dim,
        'rgb_dim': args.rgb_dim,
        'hidden_dim': args.hidden_dim,
        'bottleneck_dim': args.bottleneck_dim,
        'output_dim': args.output_dim,
        'glove_dim': args.glove_dim,
        'glove_path': args.glove_path,
        'combination_method': args.combination_method,
        'curvature': args.curvature,
        'learnable_curvature': args.learnable_curvature,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'beta3': args.beta3,
        'gamma': args.gamma,
        'temperature': args.temperature,
        'hard_neg_weight': args.hard_neg_weight,
        'eval_interval': args.eval_interval,
        'save_freq': args.save_freq,
        'gpu': args.gpu,
        'device': args.device if not args.gpu else 'cuda',
        'debug': args.debug,
    }

    # Load Hierarchy first (needed for dataset and wandb metadata)
    hierarchy = Hierarchy(args.hierarchy_path)

    # Load train and test datasets
    train_dataset = ActionAdverbDataset(args.data_dir, args.features_dir, hierarchy, split='train')
    val_dataset = ActionAdverbDataset(args.data_dir, args.features_dir, hierarchy, split='test')

    # Auto-detect feature dimensions from first sample
    print("Detecting feature dimensions from data...")
    sample = train_dataset[0]
    sample_flow = sample['flow_features']
    sample_rgb = sample['rgb_features']

    # Apply pooling if needed (same logic as collate_fn)
    if sample_flow.ndim == 2:
        sample_flow = np.mean(sample_flow, axis=0)
    if sample_rgb.ndim == 2:
        sample_rgb = np.mean(sample_rgb, axis=0)

    detected_flow_dim = sample_flow.shape[0]
    detected_rgb_dim = sample_rgb.shape[0]
    detected_input_dim = detected_flow_dim + detected_rgb_dim

    print(f"Detected dimensions: flow={detected_flow_dim}, rgb={detected_rgb_dim}, total={detected_input_dim}")

    # Update config with detected dimensions
    config['flow_dim'] = detected_flow_dim
    config['rgb_dim'] = detected_rgb_dim
    config['input_dim'] = detected_input_dim

    # Initialize wandb if enabled (after dimension detection)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )
        # Log additional metadata
        wandb.config.update({
            'num_actions': len(hierarchy.get_all_actions()),
            'num_adverbs': len(hierarchy.get_all_adverbs()),
            'num_action_parents': len(hierarchy.get_all_parent_actions()),
            'num_adverb_parents': len(hierarchy.get_all_parent_adverbs()),
        })

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Add vocab to config
    config['action_vocab'] = hierarchy.get_all_actions()
    config['adverb_vocab'] = hierarchy.get_all_adverbs()

    # Create model
    print("Initializing model...")
    model = Model(config, hierarchy, init_glove=True)

    # Wrap in hyperbolic trainer module
    hyperbolic_model = HyperbolicTrainerModule(model, config)

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=hyperbolic_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        wandb_enabled=args.wandb
    )

    # Log dataset statistics
    if args.wandb:
        wandb.log({
            'dataset/train_size': len(train_dataset),
            'dataset/test_size': len(val_dataset),
            'dataset/train_batches': len(train_loader),
            'dataset/test_batches': len(val_loader),
        })

    # Train
    print(f"Starting training for {args.epochs} epochs...")
    trainer.train(args.epochs)

    print("Training complete!")

    # Log final model statistics
    if args.wandb:
        # Count parameters
        total_params = sum(p.numel() for p in hyperbolic_model.parameters())
        trainable_params = sum(p.numel() for p in hyperbolic_model.parameters() if p.requires_grad)

        wandb.run.summary['model/total_params'] = total_params
        wandb.run.summary['model/trainable_params'] = trainable_params
        wandb.run.summary['model/non_trainable_params'] = total_params - trainable_params

        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

        wandb.finish()


if __name__ == '__main__':
    main()