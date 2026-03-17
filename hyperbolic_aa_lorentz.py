import torch
import torch.nn as nn
from hyperbolic_helpers import lorentz as L
from hyperbolic_helpers.hierarchy import Hierarchy
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
import pandas as pd
import wandb
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
from torchmetrics.classification import MulticlassAccuracy
import argparse
import random


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def embedding_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm of embedding vectors. Shape: (B, D) -> (B,)"""
    return torch.norm(x, dim=-1)


def check_tensor_health(tensor, name, location="", batch_idx=None, epoch=None,
                       debug=False, raise_on_nan=False, config=None):
    """
    Comprehensive tensor health check with detailed logging.

    Args:
        tensor: Tensor to check
        name: Descriptive name for logging
        location: Code location/context (e.g., "AAModel.forward() line 167")
        batch_idx: Optional batch index for context
        epoch: Optional epoch number for context
        debug: If True, always print stats. If False, only print on NaN/Inf
        raise_on_nan: If True, raise exception on NaN/Inf detection
        config: Optional config dict for accessing output_dir

    Returns:
        True if tensor is healthy (no NaN/Inf), False otherwise
    """
    if tensor is None:
        print(f"⚠️  WARNING: Tensor '{name}' is None at {location}")
        return False

    # Check for NaN and Inf
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    is_healthy = not (has_nan or has_inf)

    # Only print if debug=True or if unhealthy
    if debug or not is_healthy:
        print(f"\n{'='*70}")
        if not is_healthy:
            print(f"🚨 NaN/Inf DETECTED: {name}")
        else:
            print(f"✓ TENSOR CHECK: {name}")

        if location:
            print(f"Location: {location}")
        if batch_idx is not None:
            print(f"Batch: {batch_idx}", end="")
            if epoch is not None:
                print(f", Epoch: {epoch}")
            else:
                print()

        print(f"\nTENSOR HEALTH: {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}, device: {tensor.device}")

        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            total_elements = tensor.numel()
            print(f"  ❌ HAS NaN: True ({nan_count}/{total_elements} elements)")
        else:
            print(f"  ✓ No NaN")

        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            total_elements = tensor.numel()
            print(f"  ❌ HAS Inf: True ({inf_count}/{total_elements} elements)")
        else:
            print(f"  ✓ No Inf")

        # Compute statistics (handle NaN gracefully)
        try:
            if not has_nan:
                tensor_min = tensor.min().item()
                tensor_max = tensor.max().item()
                tensor_mean = tensor.mean().item()
                tensor_std = tensor.std().item()
                tensor_norm = torch.norm(tensor).item()

                print(f"  Stats: min={tensor_min:.6f}, max={tensor_max:.6f}, "
                      f"mean={tensor_mean:.6f}, std={tensor_std:.6f}")
                print(f"  Norm (L2): {tensor_norm:.6f}")
            else:
                # Try to compute stats excluding NaN
                mask = ~torch.isnan(tensor)
                if mask.any():
                    valid_tensor = tensor[mask]
                    tensor_min = valid_tensor.min().item()
                    tensor_max = valid_tensor.max().item()
                    tensor_mean = valid_tensor.mean().item()
                    tensor_std = valid_tensor.std().item()
                    print(f"  Stats (excluding NaN): min={tensor_min:.6f}, max={tensor_max:.6f}, "
                          f"mean={tensor_mean:.6f}, std={tensor_std:.6f}")
                    print(f"  Norm (L2): nan")
                else:
                    print(f"  Stats: All values are NaN")
        except Exception as e:
            print(f"  Could not compute stats: {e}")

        print(f"{'='*70}\n")

    # Save diagnostic info to file if NaN/Inf detected
    if not is_healthy and config is not None:
        try:
            output_dir = config.get('output_dir', 'checkpoints')
            run_name = config.get('run_name', 'default_run')
            diag_dir = os.path.join(output_dir, run_name)
            os.makedirs(diag_dir, exist_ok=True)

            diag_file = os.path.join(diag_dir,
                f"nan_diagnostic_epoch{epoch}_batch{batch_idx}_{name.replace(' ', '_')}.txt")

            with open(diag_file, 'w') as f:
                f.write(f"NaN/Inf Diagnostic Report\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"Tensor: {name}\n")
                f.write(f"Location: {location}\n")
                f.write(f"Batch: {batch_idx}, Epoch: {epoch}\n\n")
                f.write(f"Shape: {tensor.shape}\n")
                f.write(f"dtype: {tensor.dtype}, device: {tensor.device}\n")
                f.write(f"Has NaN: {has_nan}\n")
                f.write(f"Has Inf: {has_inf}\n\n")

                if has_nan:
                    nan_count = torch.isnan(tensor).sum().item()
                    f.write(f"NaN count: {nan_count}/{tensor.numel()}\n")
                if has_inf:
                    inf_count = torch.isinf(tensor).sum().item()
                    f.write(f"Inf count: {inf_count}/{tensor.numel()}\n")

                f.write(f"\nTensor data (first 100 elements):\n")
                f.write(str(tensor.flatten()[:100].cpu().detach().numpy()))

            print(f"📝 Diagnostic info saved to: {diag_file}")
        except Exception as e:
            print(f"⚠️  Could not save diagnostic file: {e}")

    if raise_on_nan and not is_healthy:
        raise ValueError(f"NaN or Inf detected in tensor '{name}' at {location}")

    return is_healthy


def collate_fn(batch):
    """Custom collate function for ActionAdverbDataset.

    Handles variable-length temporal features by applying mean pooling
    over the temporal dimension to get fixed-size representations.
    """
    def pool_features(features):
        if features.ndim == 2:
            return np.mean(features, axis=0)
        elif features.ndim == 1:
            return features
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

    flow_features_pooled = [pool_features(item['flow_features']) for item in batch]
    rgb_features_pooled = [pool_features(item['rgb_features']) for item in batch]

    return {
        'clip_id': [item['clip_id'] for item in batch],
        'flow_features': torch.FloatTensor(np.stack(flow_features_pooled)),
        'rgb_features': torch.FloatTensor(np.stack(rgb_features_pooled)),
        'action': [item['action'] for item in batch],
        'adverb': [item['adverb'] for item in batch],
    }


class ActionAdverbDataset(Dataset):
    def __init__(self, data_dir, features_dir, hierarchy, split='train'):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.hierarchy = hierarchy
        self.csv_path = os.path.join(data_dir, f'{split}.csv')
        self.data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_id = self.data.iloc[idx]['clip_id']

        with np.load(os.path.join(self.features_dir, f'{clip_id}_flow.npz'), allow_pickle=True) as flow_data:
            if 'features' in flow_data:
                flow_features = flow_data['features']
            elif 'arr_0' in flow_data:
                flow_features = flow_data['arr_0']
            else:
                flow_features = flow_data[flow_data.files[0]]
            flow_features = np.array(flow_features, dtype=np.float32)

        with np.load(os.path.join(self.features_dir, f'{clip_id}_rgb.npz'), allow_pickle=True) as rgb_data:
            if 'features' in rgb_data:
                rgb_features = rgb_data['features']
            elif 'arr_0' in rgb_data:
                rgb_features = rgb_data['arr_0']
            else:
                rgb_features = rgb_data[rgb_data.files[0]]
            rgb_features = np.array(rgb_features, dtype=np.float32)

        action = self.data.iloc[idx]['clustered_action']
        adverb = self.data.iloc[idx]['clustered_adverb']
        return {
            'clip_id': clip_id,
            'flow_features': flow_features,
            'rgb_features': rgb_features,
            'action': action,
            'adverb': adverb,
        }


class AAModel(nn.Module):
    def __init__(self, config: dict, init_glove=True):
        super(AAModel, self).__init__()
        self.config = config
        hierarchy = config.get('hierarchy', None)
        self.action_vocab = hierarchy.get_all_actions()
        self.adverb_vocab = hierarchy.get_all_adverbs()

        self.video_encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )

        self.action_embeddings = nn.Embedding(len(self.action_vocab), config['glove_dim'])
        self.adverb_embeddings = nn.Embedding(len(self.adverb_vocab), config['glove_dim'])

        self.action_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.adverb_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])

        if init_glove:
            self.init_glove_embeddings()

    def init_glove_embeddings(self, freeze=False):
        action_weights, self.action_2_idx = self.load_glove_embeddings(self.action_vocab)
        self.action_embeddings.weight.data.copy_(action_weights)

        adverb_weights, self.adverb_2_idx = self.load_glove_embeddings(self.adverb_vocab)
        self.adverb_embeddings.weight.data.copy_(adverb_weights)

        if freeze:
            self.action_embeddings.weight.requires_grad = False
            self.adverb_embeddings.weight.requires_grad = False

    def load_glove_embeddings(self, vocab, embedding_dim=300):
        vocab_lower = sorted([w.lower() for w in vocab])
        vocab_to_idx = {w: i for i, w in enumerate(vocab_lower)}

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
        if not_found:
            print(f"Not found words: {not_found}")
        return torch.FloatTensor(embeddings), vocab_to_idx

    def get_action_embeddings(self, indices):
        return self.action_text_proj(self.action_embeddings(indices))

    def get_adverb_embeddings(self, indices):
        return self.adverb_text_proj(self.adverb_embeddings(indices))

    def get_video_features(self, flow, rgb):
        video_input = torch.cat((flow, rgb), dim=-1)
        return self.video_encoder(video_input)

    def get_text_features(self, actions, adverbs):
        action_indices = torch.LongTensor([self.action_2_idx[a.lower()] for a in actions]).to(self.config['device'])
        adverb_indices = torch.LongTensor([self.adverb_2_idx[a.lower()] for a in adverbs]).to(self.config['device'])
        return self.get_action_embeddings(action_indices), self.get_adverb_embeddings(adverb_indices)

    def forward(self, batch):
        flow = batch['flow_features'].to(self.config['device'])
        rgb = batch['rgb_features'].to(self.config['device'])

        # Check 1: Concatenated input features
        video_input = torch.cat((flow, rgb), dim=-1)
        debug = self.config.get('debug', False)
        if self.config.get('nan_check', True):
            check_tensor_health(video_input, "video_input (flow+rgb)",
                              "AAModel.forward() after concat", debug=debug, config=self.config)

        # Check 2: Video encoder output
        video_features = self.get_video_features(flow, rgb)
        if self.config.get('nan_check', True):
            check_tensor_health(video_features, "video_features",
                              "AAModel.forward() after video_encoder", debug=debug, config=self.config)

        # Check 3 & 4: Action and adverb embeddings
        action_embeds, adverb_embeds = self.get_text_features(batch['action'], batch['adverb'])
        if self.config.get('nan_check', True):
            check_tensor_health(action_embeds, "action_embeds",
                              "AAModel.forward() after get_action_embeddings", debug=debug, config=self.config)
            check_tensor_health(adverb_embeds, "adverb_embeds",
                              "AAModel.forward() after get_adverb_embeddings", debug=debug, config=self.config)

        return {
            'video_features': video_features,
            'action_embeds': action_embeds,
            'adverb_embeds': adverb_embeds,
        }


class ProjectionModule(nn.Module):
    def __init__(self, curv_init: float = 1.0, learn_curv: bool = True, entail_weight: float = 0.2, config: dict = None):
        super(ProjectionModule, self).__init__()
        self.curv = nn.Parameter(torch.Tensor([curv_init]), requires_grad=learn_curv)
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight
        self.alpha = nn.Parameter(torch.tensor(config.get('output_dim', 512) ** -0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.config = config  # Store config for NaN checking

    def entailment_loss(self, parent_embeds, child_embeds):
        _angle = L.oxy_angle(parent_embeds, child_embeds, self.curv.exp())
        _aperture = L.half_aperture(parent_embeds, self.curv.exp())
        return self.entail_weight * torch.clamp(_angle - _aperture, min=0).mean()

    def project(self, features):
        debug = self.config.get('debug', False) if self.config else False
        nan_check = self.config.get('nan_check', True) if self.config else True

        # Check 1: Input features before scaling
        if nan_check:
            check_tensor_health(features, "features (before alpha scaling)",
                              "ProjectionModule.project() line 314", debug=debug, config=self.config)

        # Log parameters
        if debug or nan_check:
            alpha_raw = self.alpha.item()
            alpha_exp = self.alpha.exp().item()
            if debug:
                print(f"  [ProjectionModule.project] alpha (raw): {alpha_raw:.6f}, alpha.exp(): {alpha_exp:.6f}")

        # Check 2: After alpha scaling - CRITICAL checkpoint
        features_scaled = features * self.alpha.exp()
        if nan_check:
            check_tensor_health(features_scaled, "features_scaled (after alpha)",
                              "ProjectionModule.project() line 315 (CRITICAL)", debug=debug, config=self.config)

        # Check 3: After hyperbolic projection
        features_hyp = L.exp_map0(features_scaled, self.curv.exp())
        if nan_check:
            check_tensor_health(features_hyp, "features_hyperbolic",
                              "ProjectionModule.project() after exp_map0", debug=debug, config=self.config)

        return features_hyp

    def forward(self, batch):
        debug = self.config.get('debug', False) if self.config else False
        nan_check = self.config.get('nan_check', True) if self.config else True

        # Check 1: Log parameters at start
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=math.log(100))
        _scale = self.logit_scale.exp()

        if debug:
            print(f"\n[ProjectionModule.forward] PARAMETERS:")
            print(f"  curv (raw): {self.curv.item():.6f} (clamped to [{self._curv_minmax['min']:.3f}, {self._curv_minmax['max']:.3f}])")
            print(f"  curv (exp): {_curv.item():.6f}")
            print(f"  alpha (raw): {self.alpha.item():.6f}")
            print(f"  alpha (exp): {self.alpha.exp().item():.6f}")
            print(f"  logit_scale (raw): {self.logit_scale.item():.6f} (clamped to max={math.log(100):.3f})")
            print(f"  logit_scale (exp): {_scale.item():.6f}")

        # Check 2: After each projection call
        action_embeds_hyp = self.project(batch['action_embeds'])
        if nan_check:
            check_tensor_health(action_embeds_hyp, "action_embeds_hyp",
                              "ProjectionModule.forward() after project(action)", debug=debug, config=self.config)

        adverb_embeds_hyp = self.project(batch['adverb_embeds'])
        if nan_check:
            check_tensor_health(adverb_embeds_hyp, "adverb_embeds_hyp",
                              "ProjectionModule.forward() after project(adverb)", debug=debug, config=self.config)

        video_embeds_hyp = self.project(batch['video_features'])
        if nan_check:
            check_tensor_health(video_embeds_hyp, "video_embeds_hyp",
                              "ProjectionModule.forward() after project(video)", debug=debug, config=self.config)

        # Check 3: After pairwise distance calculations
        video_2_adverb_logits = -L.pairwise_dist(video_embeds_hyp, adverb_embeds_hyp, _curv)
        if nan_check:
            check_tensor_health(video_2_adverb_logits, "video_2_adverb_logits",
                              "ProjectionModule.forward() after pairwise_dist", debug=debug, config=self.config)

        adverb_2_video_logits = -L.pairwise_dist(adverb_embeds_hyp, video_embeds_hyp, _curv)
        if nan_check:
            check_tensor_health(adverb_2_video_logits, "adverb_2_video_logits",
                              "ProjectionModule.forward() after pairwise_dist", debug=debug, config=self.config)

        video_2_action_logits = -L.pairwise_dist(video_embeds_hyp, action_embeds_hyp, _curv)
        if nan_check:
            check_tensor_health(video_2_action_logits, "video_2_action_logits",
                              "ProjectionModule.forward() after pairwise_dist", debug=debug, config=self.config)

        action_2_video_logits = -L.pairwise_dist(action_embeds_hyp, video_embeds_hyp, _curv)
        if nan_check:
            check_tensor_health(action_2_video_logits, "action_2_video_logits",
                              "ProjectionModule.forward() after pairwise_dist", debug=debug, config=self.config)

        batch_size = video_embeds_hyp.size(0)
        labels = torch.arange(batch_size).to(video_embeds_hyp.device)

        # Check 4: Scaled logits before cross_entropy
        if debug:
            v2adv_scaled = video_2_adverb_logits * _scale
            print(f"\n[Scaled Logits Check]")
            print(f"  video_2_adverb * scale: min={v2adv_scaled.min().item():.2f}, max={v2adv_scaled.max().item():.2f}")
            if (v2adv_scaled.abs() > 100).any():
                print(f"  ⚠️  WARNING: Some scaled logits exceed ±100!")

        # Check 5: Individual loss components
        ce_v2adv = nn.functional.cross_entropy(video_2_adverb_logits * _scale, labels)
        if nan_check:
            check_tensor_health(ce_v2adv.unsqueeze(0), "cross_entropy(v2adv)",
                              "ProjectionModule.forward() CE loss 1", debug=debug, config=self.config)

        ce_adv2v = nn.functional.cross_entropy(adverb_2_video_logits * _scale, labels)
        if nan_check:
            check_tensor_health(ce_adv2v.unsqueeze(0), "cross_entropy(adv2v)",
                              "ProjectionModule.forward() CE loss 2", debug=debug, config=self.config)

        ce_v2act = nn.functional.cross_entropy(video_2_action_logits * _scale, labels)
        if nan_check:
            check_tensor_health(ce_v2act.unsqueeze(0), "cross_entropy(v2act)",
                              "ProjectionModule.forward() CE loss 3", debug=debug, config=self.config)

        ce_act2v = nn.functional.cross_entropy(action_2_video_logits * _scale, labels)
        if nan_check:
            check_tensor_health(ce_act2v.unsqueeze(0), "cross_entropy(act2v)",
                              "ProjectionModule.forward() CE loss 4", debug=debug, config=self.config)

        contrastive_loss = (ce_v2adv + ce_adv2v + ce_v2act + ce_act2v) / 4
        if nan_check:
            check_tensor_health(contrastive_loss.unsqueeze(0), "contrastive_loss",
                              "ProjectionModule.forward() contrastive loss", debug=debug, config=self.config)

        # Entailment loss components
        ent_loss_action = self.entailment_loss(action_embeds_hyp, video_embeds_hyp)
        if nan_check:
            check_tensor_health(ent_loss_action.unsqueeze(0), "entailment_loss(action)",
                              "ProjectionModule.forward() entailment 1", debug=debug, config=self.config)

        ent_loss_adverb = self.entailment_loss(adverb_embeds_hyp, video_embeds_hyp)
        if nan_check:
            check_tensor_health(ent_loss_adverb.unsqueeze(0), "entailment_loss(adverb)",
                              "ProjectionModule.forward() entailment 2", debug=debug, config=self.config)

        entailment_loss = ent_loss_action + ent_loss_adverb
        if nan_check:
            check_tensor_health(entailment_loss.unsqueeze(0), "entailment_loss_total",
                              "ProjectionModule.forward() entailment total", debug=debug, config=self.config)

        loss = contrastive_loss + entailment_loss
        if nan_check:
            check_tensor_health(loss.unsqueeze(0), "total_loss",
                              "ProjectionModule.forward() FINAL LOSS", debug=debug, config=self.config)

        return {
            "loss": loss,
            'action_embeds_hyp': action_embeds_hyp,
            'adverb_embeds_hyp': adverb_embeds_hyp,
            'video_embeds_hyp': video_embeds_hyp,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
                "video_2_action_logits": video_2_action_logits,
                "video_2_adverb_logits": video_2_adverb_logits,
            },
        }


class CombinedModel(nn.Module):
    def __init__(self, config):
        super(CombinedModel, self).__init__()
        self.aa_model = AAModel(config)
        self.projection_module = ProjectionModule(config=config)
        self.config = config

    def forward(self, batch):
        return self.projection_module(self.aa_model(batch))


class AverageMeter(object):
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


class Trainer:
    def __init__(self, model: CombinedModel, train_loader: DataLoader, val_loader: DataLoader, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = os.path.join(config.get('output_dir', 'outputs'), config.get('run_name', 'default_run'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.wandb_enabled = config.get('wandb', False)
        self.save_checkpoints = config.get('save_checkpoints', False)
        resume_from = config.get('resume_from', None)

        use_gpu = config.get('gpu', True) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        hierarchy = config.get('hierarchy')
        self.num_actions = len(hierarchy.get_all_actions())
        self.num_adverbs = len(hierarchy.get_all_adverbs())

        self.epoch_loss = AverageMeter()
        self.epoch_entail_loss = AverageMeter()
        self.epoch_contrastive_loss = AverageMeter()
        self.epoch_action_acc = AverageMeter()
        self.epoch_adverb_acc = AverageMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.grad_norm = AverageMeter()

        self.optimizer = AdamW(
            list(self.model.parameters()),
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 0.2),
            betas=config.get('betas', (0.9, 0.98))
        )
        steps_per_epoch = len(self.train_loader)
        total_steps = config.get('epochs', 100) * steps_per_epoch
        warmup_steps = self.config.get('warmup_steps', 2) * steps_per_epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_action_acc = 0.0
        self.best_adverb_acc = 0.0
        self.best_compositional_acc = 0.0

        if self.wandb_enabled:
            wandb.watch(model, log='all', log_freq=100, log_graph=True)

        if resume_from:
            self.load_checkpoint(resume_from)

    def train_one_epoch(self):
        self.model.train()
        self.epoch_loss.reset()
        self.epoch_contrastive_loss.reset()
        self.epoch_entail_loss.reset()
        self.epoch_action_acc.reset()
        self.epoch_adverb_acc.reset()
        self.batch_time.reset()
        self.data_time.reset()
        self.grad_norm.reset()

        start = time.time()
        global_step = self.current_epoch * len(self.train_loader)

        train_action_norms = []
        train_adverb_norms = []
        train_video_norms = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, batch_dict in enumerate(pbar):
            self.data_time.update(time.time() - start)

            if self.config.get('debug', False):
                print(f"\n[Batch]")
                print(f"  Flow: {batch_dict['flow_features'].shape}, RGB: {batch_dict['rgb_features'].shape}")
                print(f"  Actions: {batch_dict['action']}")
                print(f"  Adverbs: {batch_dict['adverb']}")

            outputs = self.model(batch_dict)
            loss = outputs['loss']
            loss_contrastive = outputs['logging']['contrastive_loss']
            loss_entail = outputs['logging']['entailment_loss']
            _scale = outputs['logging']['logit_scale']
            _curv = outputs['logging']['curv']

            # Check 1: Model outputs and embeddings
            debug = self.config.get('debug', False)
            nan_check = self.config.get('nan_check', True)

            if nan_check:
                # Check embeddings
                healthy = check_tensor_health(outputs['video_embeds_hyp'], "video_embeds_hyp",
                                            f"Trainer.train_one_epoch() batch {batch_idx}",
                                            batch_idx=batch_idx, epoch=self.current_epoch,
                                            debug=debug, config=self.config)
                if not healthy:
                    print(f"\n🚨 NaN DETECTED IN FORWARD PASS!")
                    print(f"Batch details: {batch_idx}, Epoch: {self.current_epoch}")
                    print(f"Clip IDs: {batch_dict['clip_id'][:5]}...")  # Show first 5
                    print(f"Actions: {batch_dict['action'][:5]}...")
                    print(f"Adverbs: {batch_dict['adverb'][:5]}...")
                    print(f"\n❌ Training terminated due to NaN in forward pass.")
                    return float('nan')

                check_tensor_health(outputs['action_embeds_hyp'], "action_embeds_hyp",
                                  f"Trainer.train_one_epoch() batch {batch_idx}",
                                  batch_idx=batch_idx, epoch=self.current_epoch,
                                  debug=debug, config=self.config)
                check_tensor_health(outputs['adverb_embeds_hyp'], "adverb_embeds_hyp",
                                  f"Trainer.train_one_epoch() batch {batch_idx}",
                                  batch_idx=batch_idx, epoch=self.current_epoch,
                                  debug=debug, config=self.config)

            # Check 2: Loss values
            if nan_check:
                healthy = check_tensor_health(loss.unsqueeze(0), "loss",
                                            f"Trainer.train_one_epoch() batch {batch_idx}",
                                            batch_idx=batch_idx, epoch=self.current_epoch,
                                            debug=debug, config=self.config)
                if not healthy:
                    print(f"\n🚨 NaN DETECTED IN LOSS!")
                    print(f"  Contrastive loss: {loss_contrastive.item()}")
                    print(f"  Entailment loss: {loss_entail.item()}")
                    print(f"  Logit scale: {_scale.item()}")
                    print(f"  Curvature: {_curv.item()}")
                    print(f"\n❌ Training terminated due to NaN in loss.")
                    return float('nan')

                check_tensor_health(loss_contrastive.unsqueeze(0), "loss_contrastive",
                                  f"Trainer.train_one_epoch() batch {batch_idx}",
                                  batch_idx=batch_idx, epoch=self.current_epoch,
                                  debug=debug, config=self.config)
                check_tensor_health(loss_entail.unsqueeze(0), "loss_entail",
                                  f"Trainer.train_one_epoch() batch {batch_idx}",
                                  batch_idx=batch_idx, epoch=self.current_epoch,
                                  debug=debug, config=self.config)

            if self.config.get('debug', False):
                print(f"  Losses:")
                print(f"    Total: {loss.item():.4f}")
                print(f"    Contrastive: {loss_contrastive.item():.4f}")
                print(f"    Entail: {loss_entail.item():.4f}")
                print(f"    Scale: {_scale.item():.4f}")
                print(f"    Curv: {_curv.item():.4f}")

            self.optimizer.zero_grad()
            loss.backward()

            # Check 3: Gradients after backward pass
            if nan_check:
                grad_check_failed = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"\n🚨 NaN/Inf DETECTED IN GRADIENT: {name}")
                            print(f"  Batch: {batch_idx}, Epoch: {self.current_epoch}")
                            print(f"  Gradient shape: {param.grad.shape}")
                            print(f"  Has NaN: {torch.isnan(param.grad).any().item()}")
                            print(f"  Has Inf: {torch.isinf(param.grad).any().item()}")
                            if not torch.isnan(param.grad).all():
                                print(f"  Grad stats (excluding NaN): min={param.grad[~torch.isnan(param.grad)].min().item():.6f}, "
                                      f"max={param.grad[~torch.isnan(param.grad)].max().item():.6f}")
                            grad_check_failed = True
                            break  # Stop at first NaN gradient

                if grad_check_failed:
                    print(f"\n❌ Training terminated due to NaN/Inf in gradients.")
                    # Save gradient diagnostic
                    diag_file = os.path.join(self.output_dir,
                        f"gradient_nan_epoch{self.current_epoch}_batch{batch_idx}.txt")
                    with open(diag_file, 'w') as f:
                        f.write(f"Gradient NaN Diagnostic\n")
                        f.write(f"{'='*70}\n\n")
                        f.write(f"Epoch: {self.current_epoch}, Batch: {batch_idx}\n\n")
                        for n, p in self.model.named_parameters():
                            if p.grad is not None:
                                has_nan = torch.isnan(p.grad).any().item()
                                has_inf = torch.isinf(p.grad).any().item()
                                f.write(f"{n}: has_nan={has_nan}, has_inf={has_inf}\n")
                    print(f"📝 Gradient diagnostic saved to: {diag_file}")
                    return float('nan')

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.model.parameters() if p.grad is not None
            ) ** 0.5
            self.grad_norm.update(total_norm)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.epoch_loss.update(loss.item())
            self.epoch_contrastive_loss.update(loss_contrastive.item())
            self.epoch_entail_loss.update(loss_entail.item())

            with torch.no_grad():
                v2a_logits = outputs['logging']['video_2_action_logits']
                v2adv_logits = outputs['logging']['video_2_adverb_logits']
                in_batch_labels = torch.arange(v2a_logits.size(0), device=v2a_logits.device)
                action_batch_acc = (v2a_logits.argmax(dim=1) == in_batch_labels).float().mean().item()
                adverb_batch_acc = (v2adv_logits.argmax(dim=1) == in_batch_labels).float().mean().item()
                self.epoch_action_acc.update(action_batch_acc)
                self.epoch_adverb_acc.update(adverb_batch_acc)

                train_action_norms.append(embedding_norm(outputs['action_embeds_hyp']).cpu())
                train_adverb_norms.append(embedding_norm(outputs['adverb_embeds_hyp']).cpu())
                train_video_norms.append(embedding_norm(outputs['video_embeds_hyp']).cpu())

            self.batch_time.update(time.time() - start - self.data_time.val)

            pbar.set_postfix({
                'loss': loss.item(),
                'l_c': loss_contrastive.item(),
                'l_e': loss_entail.item(),
                'act_acc': action_batch_acc,
                'adv_acc': adverb_batch_acc,
                'grad': total_norm,
            })

            if self.wandb_enabled and batch_idx % 10 == 0:
                wandb.log({
                    'step': global_step + batch_idx,
                    'step/loss_total': loss.item(),
                    'step/loss_contrastive': loss_contrastive.item(),
                    'step/loss_entail': loss_entail.item(),
                    'step/grad_norm': total_norm,
                    'step/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step/batch_time': self.batch_time.val,
                    'step/data_time': self.data_time.val,
                    'step/action_top1_inbatch': action_batch_acc,
                    'step/adverb_top1_inbatch': adverb_batch_acc,
                })

            start = time.time()

        avg_loss = self.epoch_loss.avg
        avg_contrastive_loss = self.epoch_contrastive_loss.avg
        avg_entail_loss = self.epoch_entail_loss.avg
        avg_action_acc = self.epoch_action_acc.avg
        avg_adverb_acc = self.epoch_adverb_acc.avg

        self.train_losses.append(avg_loss)

        if self.wandb_enabled:
            all_action_norms = torch.cat(train_action_norms).numpy()
            all_adverb_norms = torch.cat(train_adverb_norms).numpy()
            all_video_norms = torch.cat(train_video_norms).numpy()
            wandb.log({
                'epoch': self.current_epoch,
                'train/loss_total': avg_loss,
                'train/loss_contrastive': avg_contrastive_loss,
                'train/loss_entail': avg_entail_loss,
                'train/action_top1_inbatch': avg_action_acc,
                'train/adverb_top1_inbatch': avg_adverb_acc,
                'train/batch_time': self.batch_time.avg,
                'train/data_time': self.data_time.avg,
                'train/grad_norm': self.grad_norm.avg,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/norm_action': wandb.Histogram(all_action_norms),
                'train/norm_adverb': wandb.Histogram(all_adverb_norms),
                'train/norm_video': wandb.Histogram(all_video_norms),
            })

        print('E: %d | L: %.2E | L_c: %.2E | L_e: %.2E | Act@1: %.3f | Adv@1: %.3f | Grad: %.2E | LR: %.2E' %
              (self.current_epoch, avg_loss, avg_contrastive_loss, avg_entail_loss,
               avg_action_acc, avg_adverb_acc, self.grad_norm.avg, self.optimizer.param_groups[0]['lr']))

        return avg_loss

    def validate(self):
        self.model.eval()

        val_loss_meter = AverageMeter()
        val_contrastive_loss_meter = AverageMeter()
        val_entail_loss_meter = AverageMeter()

        dist_action_top1 = MulticlassAccuracy(num_classes=self.num_actions, top_k=1).to(self.device)
        dist_action_top5 = MulticlassAccuracy(num_classes=self.num_actions, top_k=5).to(self.device)
        dist_adverb_top1 = MulticlassAccuracy(num_classes=self.num_adverbs, top_k=1).to(self.device)
        dist_adverb_top5 = MulticlassAccuracy(num_classes=self.num_adverbs, top_k=5).to(self.device)

        inner_action_top1 = MulticlassAccuracy(num_classes=self.num_actions, top_k=1).to(self.device)
        inner_action_top5 = MulticlassAccuracy(num_classes=self.num_actions, top_k=5).to(self.device)
        inner_adverb_top1 = MulticlassAccuracy(num_classes=self.num_adverbs, top_k=1).to(self.device)
        inner_adverb_top5 = MulticlassAccuracy(num_classes=self.num_adverbs, top_k=5).to(self.device)

        # Compositional accuracy tracker
        compositional_correct = 0
        total_samples = 0

        val_video_norms = []

        with torch.no_grad():
            all_action_indices = torch.arange(self.num_actions, device=self.device)
            all_adverb_indices = torch.arange(self.num_adverbs, device=self.device)
            action_raw = self.model.aa_model.get_action_embeddings(all_action_indices)
            adverb_raw = self.model.aa_model.get_adverb_embeddings(all_adverb_indices)
            action_gallery = self.model.projection_module.project(action_raw)
            adverb_gallery = self.model.projection_module.project(adverb_raw)
            _curv = self.model.projection_module.curv.exp()

            for batch_dict in tqdm(self.val_loader, desc='Validation'):
                outputs = self.model(batch_dict)
                val_loss_meter.update(outputs['loss'].item())
                val_contrastive_loss_meter.update(outputs['logging']['contrastive_loss'].item())
                val_entail_loss_meter.update(outputs['logging']['entailment_loss'].item())

                video_embeds = outputs['video_embeds_hyp']
                val_video_norms.append(embedding_norm(video_embeds).cpu())

                action_scores_dist = -L.pairwise_dist(video_embeds, action_gallery, _curv)
                adverb_scores_dist = -L.pairwise_dist(video_embeds, adverb_gallery, _curv)
                action_scores_inner = L.pairwise_inner(video_embeds, action_gallery, _curv)
                adverb_scores_inner = L.pairwise_inner(video_embeds, adverb_gallery, _curv)

                action_labels = torch.LongTensor(
                    [self.model.aa_model.action_2_idx[a.lower()] for a in batch_dict['action']]
                ).to(self.device)
                adverb_labels = torch.LongTensor(
                    [self.model.aa_model.adverb_2_idx[a.lower()] for a in batch_dict['adverb']]
                ).to(self.device)

                dist_action_top1.update(action_scores_dist, action_labels)
                dist_action_top5.update(action_scores_dist, action_labels)
                dist_adverb_top1.update(adverb_scores_dist, adverb_labels)
                dist_adverb_top5.update(adverb_scores_dist, adverb_labels)

                inner_action_top1.update(action_scores_inner, action_labels)
                inner_action_top5.update(action_scores_inner, action_labels)
                inner_adverb_top1.update(adverb_scores_inner, adverb_labels)
                inner_adverb_top5.update(adverb_scores_inner, adverb_labels)

                # Compositional accuracy (both action AND adverb correct)
                action_correct = (action_scores_dist.argmax(dim=1) == action_labels)
                adverb_correct = (adverb_scores_dist.argmax(dim=1) == adverb_labels)
                compositional_correct += (action_correct & adverb_correct).sum().item()
                total_samples += video_embeds.size(0)

        avg_val_loss = val_loss_meter.avg

        d_act1 = dist_action_top1.compute().item()
        d_act5 = dist_action_top5.compute().item()
        d_adv1 = dist_adverb_top1.compute().item()
        d_adv5 = dist_adverb_top5.compute().item()

        i_act1 = inner_action_top1.compute().item()
        i_act5 = inner_action_top5.compute().item()
        i_adv1 = inner_adverb_top1.compute().item()
        i_adv5 = inner_adverb_top5.compute().item()

        # Compositional accuracy
        compositional_acc = compositional_correct / total_samples if total_samples > 0 else 0.0

        self.val_losses.append(avg_val_loss)

        print('E %d | Val Loss: %.4f' % (self.current_epoch, avg_val_loss))
        print('  [dist]  Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (d_act1, d_act5, d_adv1, d_adv5))
        print('  [inner] Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (i_act1, i_act5, i_adv1, i_adv5))
        print('  [compositional] Both Action+Adverb Correct: %.4f' % compositional_acc)

        if self.wandb_enabled:
            wandb.log({
                'epoch': self.current_epoch,
                'val/loss_total': avg_val_loss,
                'val/loss_contrastive': val_contrastive_loss_meter.avg,
                'val/loss_entail': val_entail_loss_meter.avg,
                'val/dist_action_top1': d_act1,
                'val/dist_action_top5': d_act5,
                'val/dist_adverb_top1': d_adv1,
                'val/dist_adverb_top5': d_adv5,
                'val/dist_combined_top1': (d_act1 + d_adv1) / 2,
                'val/inner_action_top1': i_act1,
                'val/inner_action_top5': i_act5,
                'val/inner_adverb_top1': i_adv1,
                'val/inner_adverb_top5': i_adv5,
                'val/inner_combined_top1': (i_act1 + i_adv1) / 2,
                'val/compositional_accuracy': compositional_acc,
                'val/norm_action_vocab': wandb.Histogram(embedding_norm(action_gallery).cpu().numpy()),
                'val/norm_adverb_vocab': wandb.Histogram(embedding_norm(adverb_gallery).cpu().numpy()),
                'val/norm_video': wandb.Histogram(torch.cat(val_video_norms).numpy()),
            })

        if d_act1 > self.best_action_acc:
            self.best_action_acc = d_act1
        if d_adv1 > self.best_adverb_acc:
            self.best_adverb_acc = d_adv1
        if compositional_acc > self.best_compositional_acc:
            self.best_compositional_acc = compositional_acc

        return avg_val_loss

    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            epoch_time = time.time() - epoch_start_time

            print(f'\n{"="*80}')
            print(f'Epoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  Best Val Loss: {self.best_val_loss:.4f}')
            print(f'  Best Action Acc: {self.best_action_acc:.4f} | Best Adverb Acc: {self.best_adverb_acc:.4f}')
            print(f'  Best Compositional Acc: {self.best_compositional_acc:.4f}')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            print(f'{"="*80}\n')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.save_checkpoints:
                    self.save_checkpoint('best_model.pth')

                if self.wandb_enabled:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_action_acc'] = self.best_action_acc
                    wandb.run.summary['best_adverb_acc'] = self.best_adverb_acc
                    wandb.run.summary['best_compositional_acc'] = self.best_compositional_acc

            if self.save_checkpoints and (epoch + 1) % self.config.get('save_freq', 100) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

            if self.wandb_enabled:
                wandb.log({
                    'epoch': epoch,
                    'timing/epoch_time': epoch_time,
                    'timing/samples_per_second': len(self.train_loader.dataset) / epoch_time,
                })

        print(f'\n{"="*80}')
        print(f'Training Complete!')
        print(f'  Best Val Loss: {self.best_val_loss:.4f}')
        print(f'  Best Action Acc: {self.best_action_acc:.4f}')
        print(f'  Best Adverb Acc: {self.best_adverb_acc:.4f}')
        print(f'  Best Compositional Acc: {self.best_compositional_acc:.4f}')
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
            'best_compositional_acc': self.best_compositional_acc,
            'config': self.config
        }
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint: {filename}')

        if self.wandb_enabled and filename == 'best_model.pth':
            artifact = wandb.Artifact(
                name=f'model-{wandb.run.id}',
                type='model',
                description=f'Best model at epoch {self.current_epoch} with val_loss={self.best_val_loss:.4f}, '
                            f'action_acc={self.best_action_acc:.4f}, adverb_acc={self.best_adverb_acc:.4f}, '
                            f'compositional_acc={self.best_compositional_acc:.4f}',
                metadata={
                    'epoch': self.current_epoch,
                    'val_loss': self.best_val_loss,
                    'action_acc': self.best_action_acc,
                    'adverb_acc': self.best_adverb_acc,
                    'compositional_acc': self.best_compositional_acc,
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
        self.best_compositional_acc = checkpoint.get('best_compositional_acc', 0.0)
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
        print(f'  Best Val Loss: {self.best_val_loss:.4f}')
        print(f'  Best Action Acc: {self.best_action_acc:.4f}')
        print(f'  Best Adverb Acc: {self.best_adverb_acc:.4f}')
        print(f'  Best Compositional Acc: {self.best_compositional_acc:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Hyperbolic Action-Adverb Training')

    # Data arguments
    parser.add_argument('--hierarchy-path', type=str, default='datasets/VATEX_Adverbs/action_adverb_hierarchy.json')
    parser.add_argument('--data-dir', type=str, default='splits/without_unlabelled/seen_compositions/vatex_adverbs')
    parser.add_argument('--features-dir', type=str, default='datasets/VATEX_Adverbs/VATEX_Adverbs_features/')
    parser.add_argument('--glove-path', type=str, default='datasets/glove.6B.300d.txt')

    # Model architecture arguments
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--output-dim', type=int, default=300)
    parser.add_argument('--glove-dim', type=int, default=300)

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    # Checkpointing
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                        help='Enable checkpoint saving (disabled by default)')
    parser.add_argument('--save-freq', type=int, default=100)

    # System arguments
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--debug', action='store_true')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--resume-from', type=str, default=None)

    # Wandb arguments
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.add_argument('--wandb-project', type=str, default='hyperbolic-action-adverb')
    parser.add_argument('--wandb-name', type=str, default=None)

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    hierarchy = Hierarchy(args.hierarchy_path)

    use_gpu = args.gpu and torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    config = {
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'glove_dim': args.glove_dim,
        'glove_path': args.glove_path,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'save_checkpoints': args.save_checkpoints,
        'save_freq': args.save_freq,
        'gpu': args.gpu,
        'device': device,
        'debug': args.debug,
        'hierarchy': hierarchy,
        'output_dir': args.output_dir,
        'resume_from': args.resume_from,
        'wandb': args.wandb,
        'seed': args.seed,
    }

    train_dataset = ActionAdverbDataset(args.data_dir, args.features_dir, hierarchy, split='train')
    val_dataset = ActionAdverbDataset(args.data_dir, args.features_dir, hierarchy, split='test')

    # Auto-detect input dimensions from first sample
    print("Detecting feature dimensions from data...")
    sample = train_dataset[0]
    sample_flow = sample['flow_features']
    sample_rgb = sample['rgb_features']
    if sample_flow.ndim == 2:
        sample_flow = np.mean(sample_flow, axis=0)
    if sample_rgb.ndim == 2:
        sample_rgb = np.mean(sample_rgb, axis=0)
    detected_input_dim = sample_flow.shape[0] + sample_rgb.shape[0]
    print(f"Detected dimensions: flow={sample_flow.shape[0]}, rgb={sample_rgb.shape[0]}, total={detected_input_dim}")
    config['input_dim'] = detected_input_dim

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)
        wandb.config.update({
            'num_actions': len(hierarchy.get_all_actions()),
            'num_adverbs': len(hierarchy.get_all_adverbs()),
        })

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    print("Initializing model...")
    model = CombinedModel(config)

    print("Creating trainer...")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)

    if args.wandb:
        wandb.log({
            'dataset/train_size': len(train_dataset),
            'dataset/test_size': len(val_dataset),
            'dataset/train_batches': len(train_loader),
            'dataset/test_batches': len(val_loader),
        })

    trainer.train(args.epochs)

    if args.wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary['model/total_params'] = total_params
        wandb.run.summary['model/trainable_params'] = trainable_params
        wandb.run.summary['model/non_trainable_params'] = total_params - trainable_params
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        wandb.finish()


if __name__ == '__main__':
    main()
