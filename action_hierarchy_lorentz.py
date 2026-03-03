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


def embedding_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm of embedding vectors. Shape: (B, D) -> (B,)"""
    return torch.norm(x, dim=-1)


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


def get_all_pairs(action_vocab, adverb_vocab):
    """Generate all possible (action, adverb) pairs (cartesian product)."""
    all_pairs = [(action, adverb)
                 for action in sorted(action_vocab)
                 for adverb in sorted(adverb_vocab)]
    return all_pairs


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


class AAModelWithPairs(nn.Module):
    def __init__(self, config: dict, init_glove=True):
        super(AAModelWithPairs, self).__init__()
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

        # NEW: Pair composition layer
        # concat(action_emb, adverb_emb) [2*D] → projection [D]
        self.pair_composition = nn.Sequential(
            nn.Linear(2 * config['output_dim'], config['output_dim']),
            nn.ReLU(),
            nn.Linear(config['output_dim'], config['output_dim'])
        )

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

    def get_pair_embeddings(self, action_embeds, adverb_embeds):
        """Combine action + adverb into pair embedding."""
        concatenated = torch.cat([action_embeds, adverb_embeds], dim=-1)  # (B, 2D)
        return self.pair_composition(concatenated)  # (B, D)

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
        video_features = self.get_video_features(flow, rgb)
        action_embeds, adverb_embeds = self.get_text_features(batch['action'], batch['adverb'])

        # NEW: Compute pair embeddings
        pair_embeds = self.get_pair_embeddings(action_embeds, adverb_embeds)

        return {
            'video_features': video_features,
            'action_embeds': action_embeds,
            'adverb_embeds': adverb_embeds,
            'pair_embeds': pair_embeds,
        }


class ActionHierarchyProjection(nn.Module):
    """
    3-level hierarchy: Action ⊃ Pair ⊃ Video

    Hierarchy structure:
    - Level 1 (top): Action
    - Level 2 (middle): Action-Adverb Pair
    - Level 3 (bottom): Video
    """
    def __init__(self, curv_init: float = 1.0, learn_curv: bool = True, entail_weight: float = 0.2, config: dict = None):
        super(ActionHierarchyProjection, self).__init__()
        self.curv = nn.Parameter(torch.Tensor([curv_init]), requires_grad=learn_curv)
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight
        self.alpha = nn.Parameter(torch.tensor(config.get('output_dim', 512) ** -0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def entailment_loss(self, parent_embeds, child_embeds):
        """Compute entailment loss (parent ⊃ child)."""
        _angle = L.oxy_angle(parent_embeds, child_embeds, self.curv.exp())
        _aperture = L.half_aperture(parent_embeds, self.curv.exp())
        return self.entail_weight * torch.clamp(_angle - _aperture, min=0).mean()

    def project(self, features):
        """Project to hyperbolic space."""
        features = features * self.alpha.exp()
        return L.exp_map0(features, self.curv.exp())

    def forward(self, batch):
        # Clamp curvature
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Project all embeddings to hyperbolic space
        action_hyp = self.project(batch['action_embeds'])
        adverb_hyp = self.project(batch['adverb_embeds'])  # Not in hierarchy but kept
        pair_hyp = self.project(batch['pair_embeds'])
        video_hyp = self.project(batch['video_features'])

        batch_size = video_hyp.size(0)
        labels = torch.arange(batch_size).to(video_hyp.device)

        # Clamp logit scale
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=math.log(100))
        _scale = self.logit_scale.exp()

        # === CONTRASTIVE LOSSES ===
        # Level 1-2: Action ↔ Pair
        action2pair = -L.pairwise_dist(action_hyp, pair_hyp, _curv)
        pair2action = -L.pairwise_dist(pair_hyp, action_hyp, _curv)

        # Level 2-3: Pair ↔ Video
        pair2video = -L.pairwise_dist(pair_hyp, video_hyp, _curv)
        video2pair = -L.pairwise_dist(video_hyp, pair_hyp, _curv)

        # Level 1-3: Action ↔ Video (auxiliary)
        action2video = -L.pairwise_dist(action_hyp, video_hyp, _curv)
        video2action = -L.pairwise_dist(video_hyp, action_hyp, _curv)

        contrastive_loss = (
            nn.functional.cross_entropy(action2pair * _scale, labels) +
            nn.functional.cross_entropy(pair2action * _scale, labels) +
            nn.functional.cross_entropy(pair2video * _scale, labels) +
            nn.functional.cross_entropy(video2pair * _scale, labels) +
            nn.functional.cross_entropy(action2video * _scale, labels) +
            nn.functional.cross_entropy(video2action * _scale, labels)
        ) / 6

        # === 3-LEVEL ENTAILMENT LOSSES ===
        # Level 1 ⊃ Level 2: Action ⊃ Pair
        entail_action_pair = self.entailment_loss(action_hyp, pair_hyp)

        # Level 2 ⊃ Level 3: Pair ⊃ Video
        entail_pair_video = self.entailment_loss(pair_hyp, video_hyp)

        # Level 1 ⊃ Level 3: Action ⊃ Video (transitive, weighted 0.5)
        entail_action_video = 0.5 * self.entailment_loss(action_hyp, video_hyp)

        entailment_loss = (
            entail_action_pair +
            entail_pair_video +
            entail_action_video
        )

        total_loss = contrastive_loss + entailment_loss

        return {
            "loss": total_loss,
            'action_embeds_hyp': action_hyp,
            'adverb_embeds_hyp': adverb_hyp,
            'pair_embeds_hyp': pair_hyp,
            'video_embeds_hyp': video_hyp,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "entail_action_pair": entail_action_pair,
                "entail_pair_video": entail_pair_video,
                "entail_action_video": entail_action_video,
                "logit_scale": _scale,
                "curv": _curv,
                "action2pair_logits": action2pair,
                "pair2video_logits": pair2video,
                "action2video_logits": action2video,
            },
        }


class CombinedModel(nn.Module):
    def __init__(self, config):
        super(CombinedModel, self).__init__()
        self.aa_model = AAModelWithPairs(config)
        self.projection_module = ActionHierarchyProjection(config=config)
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
        self.epoch_pair_acc = AverageMeter()
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
        self.best_pair_acc = 0.0
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
        self.epoch_pair_acc.reset()
        self.batch_time.reset()
        self.data_time.reset()
        self.grad_norm.reset()

        start = time.time()
        global_step = self.current_epoch * len(self.train_loader)

        train_action_norms = []
        train_pair_norms = []
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

            if self.config.get('debug', False):
                print(f"  Losses:")
                print(f"    Total: {loss.item():.4f}")
                print(f"    Contrastive: {loss_contrastive.item():.4f}")
                print(f"    Entail: {loss_entail.item():.4f}")
                print(f"    Scale: {_scale.item():.4f}")
                print(f"    Curv: {_curv.item():.4f}")

            self.optimizer.zero_grad()
            loss.backward()

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
                a2p_logits = outputs['logging']['action2pair_logits']
                p2v_logits = outputs['logging']['pair2video_logits']
                in_batch_labels = torch.arange(a2p_logits.size(0), device=a2p_logits.device)
                action_batch_acc = (a2p_logits.argmax(dim=1) == in_batch_labels).float().mean().item()
                pair_batch_acc = (p2v_logits.argmax(dim=1) == in_batch_labels).float().mean().item()
                self.epoch_action_acc.update(action_batch_acc)
                self.epoch_pair_acc.update(pair_batch_acc)

                train_action_norms.append(embedding_norm(outputs['action_embeds_hyp']).cpu())
                train_pair_norms.append(embedding_norm(outputs['pair_embeds_hyp']).cpu())
                train_video_norms.append(embedding_norm(outputs['video_embeds_hyp']).cpu())

            self.batch_time.update(time.time() - start - self.data_time.val)

            pbar.set_postfix({
                'loss': loss.item(),
                'l_c': loss_contrastive.item(),
                'l_e': loss_entail.item(),
                'act_acc': action_batch_acc,
                'pair_acc': pair_batch_acc,
                'grad': total_norm,
            })

            if self.wandb_enabled and batch_idx % 10 == 0:
                wandb.log({
                    'step': global_step + batch_idx,
                    'step/loss_total': loss.item(),
                    'step/loss_contrastive': loss_contrastive.item(),
                    'step/loss_entail': loss_entail.item(),
                    'step/entail_action_pair': outputs['logging']['entail_action_pair'].item(),
                    'step/entail_pair_video': outputs['logging']['entail_pair_video'].item(),
                    'step/entail_action_video': outputs['logging']['entail_action_video'].item(),
                    'step/grad_norm': total_norm,
                    'step/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step/batch_time': self.batch_time.val,
                    'step/data_time': self.data_time.val,
                    'step/action2pair_top1_inbatch': action_batch_acc,
                    'step/pair2video_top1_inbatch': pair_batch_acc,
                    'step/logit_scale': _scale.item(),
                    'step/curvature': _curv.item(),
                })

            start = time.time()

        avg_loss = self.epoch_loss.avg
        avg_contrastive_loss = self.epoch_contrastive_loss.avg
        avg_entail_loss = self.epoch_entail_loss.avg
        avg_action_acc = self.epoch_action_acc.avg
        avg_pair_acc = self.epoch_pair_acc.avg

        self.train_losses.append(avg_loss)

        if self.wandb_enabled:
            all_action_norms = torch.cat(train_action_norms).numpy()
            all_pair_norms = torch.cat(train_pair_norms).numpy()
            all_video_norms = torch.cat(train_video_norms).numpy()
            wandb.log({
                'epoch': self.current_epoch,
                'train/loss_total': avg_loss,
                'train/loss_contrastive': avg_contrastive_loss,
                'train/loss_entail': avg_entail_loss,
                'train/action2pair_top1_inbatch': avg_action_acc,
                'train/pair2video_top1_inbatch': avg_pair_acc,
                'train/batch_time': self.batch_time.avg,
                'train/data_time': self.data_time.avg,
                'train/grad_norm': self.grad_norm.avg,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/norm_action': wandb.Histogram(all_action_norms),
                'train/norm_pair': wandb.Histogram(all_pair_norms),
                'train/norm_video': wandb.Histogram(all_video_norms),
            })

        print('E: %d | L: %.2E | L_c: %.2E | L_e: %.2E | Act@1: %.3f | Pair@1: %.3f | Grad: %.2E | LR: %.2E' %
              (self.current_epoch, avg_loss, avg_contrastive_loss, avg_entail_loss,
               avg_action_acc, avg_pair_acc, self.grad_norm.avg, self.optimizer.param_groups[0]['lr']))

        return avg_loss

    def validate(self):
        self.model.eval()

        val_loss_meter = AverageMeter()
        val_contrastive_loss_meter = AverageMeter()
        val_entail_loss_meter = AverageMeter()

        # Build galleries once
        all_actions = sorted(self.config['hierarchy'].get_all_actions())
        all_adverbs = sorted(self.config['hierarchy'].get_all_adverbs())
        all_pairs = get_all_pairs(all_actions, all_adverbs)
        pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}
        num_pairs = len(all_pairs)

        dist_action_top1 = MulticlassAccuracy(num_classes=self.num_actions, top_k=1).to(self.device)
        dist_action_top5 = MulticlassAccuracy(num_classes=self.num_actions, top_k=5).to(self.device)
        dist_pair_top1 = MulticlassAccuracy(num_classes=num_pairs, top_k=1).to(self.device)
        dist_pair_top5 = MulticlassAccuracy(num_classes=num_pairs, top_k=5).to(self.device)

        val_video_norms = []

        with torch.no_grad():
            # Build action gallery
            action_indices = torch.arange(self.num_actions, device=self.device)
            action_raw = self.model.aa_model.get_action_embeddings(action_indices)
            action_gallery = self.model.projection_module.project(action_raw)

            # Build pair gallery (all possible pairs)
            print(f"Building pair gallery with {num_pairs} pairs...")
            pair_gallery_list = []
            for action_str, adverb_str in tqdm(all_pairs, desc='Building pair gallery'):
                action_idx = self.model.aa_model.action_2_idx[action_str.lower()]
                adverb_idx = self.model.aa_model.adverb_2_idx[adverb_str.lower()]
                action_emb = self.model.aa_model.get_action_embeddings(torch.tensor([action_idx], device=self.device))
                adverb_emb = self.model.aa_model.get_adverb_embeddings(torch.tensor([adverb_idx], device=self.device))
                pair_emb = self.model.aa_model.get_pair_embeddings(action_emb, adverb_emb)
                pair_gallery_list.append(pair_emb)
            pair_gallery = self.model.projection_module.project(torch.cat(pair_gallery_list, dim=0))

            _curv = self.model.projection_module.curv.exp()

            for batch_dict in tqdm(self.val_loader, desc='Validation'):
                outputs = self.model(batch_dict)
                val_loss_meter.update(outputs['loss'].item())
                val_contrastive_loss_meter.update(outputs['logging']['contrastive_loss'].item())
                val_entail_loss_meter.update(outputs['logging']['entailment_loss'].item())

                video_embeds = outputs['video_embeds_hyp']
                val_video_norms.append(embedding_norm(video_embeds).cpu())

                # Retrieve against galleries
                action_scores_dist = -L.pairwise_dist(video_embeds, action_gallery, _curv)
                pair_scores_dist = -L.pairwise_dist(video_embeds, pair_gallery, _curv)

                # Get labels
                action_labels = torch.LongTensor(
                    [self.model.aa_model.action_2_idx[a.lower()] for a in batch_dict['action']]
                ).to(self.device)
                pair_labels = torch.LongTensor(
                    [pair_to_idx[(a, adv)] for a, adv in zip(batch_dict['action'], batch_dict['adverb'])]
                ).to(self.device)

                # Update metrics
                dist_action_top1.update(action_scores_dist, action_labels)
                dist_action_top5.update(action_scores_dist, action_labels)
                dist_pair_top1.update(pair_scores_dist, pair_labels)
                dist_pair_top5.update(pair_scores_dist, pair_labels)

        avg_val_loss = val_loss_meter.avg

        d_act1 = dist_action_top1.compute().item()
        d_act5 = dist_action_top5.compute().item()
        d_pair1 = dist_pair_top1.compute().item()
        d_pair5 = dist_pair_top5.compute().item()

        # Compositional accuracy = pair_top1
        compositional_acc = d_pair1

        self.val_losses.append(avg_val_loss)

        print('E %d | Val Loss: %.4f' % (self.current_epoch, avg_val_loss))
        print('  [dist]  Action Top-1: %.4f Top-5: %.4f | Pair Top-1: %.4f Top-5: %.4f' % (d_act1, d_act5, d_pair1, d_pair5))
        print('  [compositional] Pair Correct: %.4f' % compositional_acc)

        if self.wandb_enabled:
            wandb.log({
                'epoch': self.current_epoch,
                'val/loss_total': avg_val_loss,
                'val/loss_contrastive': val_contrastive_loss_meter.avg,
                'val/loss_entail': val_entail_loss_meter.avg,
                'val/action_top1': d_act1,
                'val/action_top5': d_act5,
                'val/pair_top1': d_pair1,
                'val/pair_top5': d_pair5,
                'val/compositional_accuracy': compositional_acc,
                'val/norm_action_gallery': wandb.Histogram(embedding_norm(action_gallery).cpu().numpy()),
                'val/norm_pair_gallery': wandb.Histogram(embedding_norm(pair_gallery).cpu().numpy()),
                'val/norm_video': wandb.Histogram(torch.cat(val_video_norms).numpy()),
            })

        if d_act1 > self.best_action_acc:
            self.best_action_acc = d_act1
        if d_pair1 > self.best_pair_acc:
            self.best_pair_acc = d_pair1
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
            print(f'  Best Action Acc: {self.best_action_acc:.4f} | Best Pair Acc: {self.best_pair_acc:.4f}')
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
                    wandb.run.summary['best_pair_acc'] = self.best_pair_acc
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
        print(f'  Best Pair Acc: {self.best_pair_acc:.4f}')
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
            'best_pair_acc': self.best_pair_acc,
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
                            f'action_acc={self.best_action_acc:.4f}, pair_acc={self.best_pair_acc:.4f}, '
                            f'compositional_acc={self.best_compositional_acc:.4f}',
                metadata={
                    'epoch': self.current_epoch,
                    'val_loss': self.best_val_loss,
                    'action_acc': self.best_action_acc,
                    'pair_acc': self.best_pair_acc,
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
        self.best_pair_acc = checkpoint.get('best_pair_acc', 0.0)
        self.best_compositional_acc = checkpoint.get('best_compositional_acc', 0.0)
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
        print(f'  Best Val Loss: {self.best_val_loss:.4f}')
        print(f'  Best Action Acc: {self.best_action_acc:.4f}')
        print(f'  Best Pair Acc: {self.best_pair_acc:.4f}')
        print(f'  Best Compositional Acc: {self.best_compositional_acc:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Hyperbolic Action Hierarchy Training (Action → Pair → Video)')

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
    parser.add_argument('--wandb-project', type=str, default='hyperbolic-action-hierarchy')
    parser.add_argument('--wandb-name', type=str, default=None)

    args = parser.parse_args()

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
            'hierarchy_type': 'action',
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
