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
        self.action_parent_vocab = hierarchy.get_all_parent_actions()
        self.adverb_parent_vocab = hierarchy.get_all_parent_adverbs()

        self.video_encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )

        self.action_embeddings = nn.Embedding(len(self.action_vocab), config['glove_dim'])
        self.adverb_embeddings = nn.Embedding(len(self.adverb_vocab), config['glove_dim'])
        self.action_parent_embeddings = nn.Embedding(len(self.action_parent_vocab), config['glove_dim'])
        self.adverb_parent_embeddings = nn.Embedding(len(self.adverb_parent_vocab), config['glove_dim'])

        self.action_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.adverb_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.action_parent_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])
        self.adverb_parent_text_proj = nn.Linear(config['glove_dim'], config['output_dim'])

        if init_glove:
            self.init_glove_embeddings()

    def init_glove_embeddings(self, freeze=False):
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

    def get_action_parent_embeddings(self, indices):
        return self.action_parent_text_proj(self.action_parent_embeddings(indices))

    def get_adverb_parent_embeddings(self, indices):
        return self.adverb_parent_text_proj(self.adverb_parent_embeddings(indices))

    def get_video_features(self, flow, rgb):
        video_input = torch.cat((flow, rgb), dim=-1)
        return self.video_encoder(video_input)

    def get_text_features(self, actions, adverbs):
        action_indices = torch.LongTensor([self.action_2_idx[a.lower()] for a in actions]).to(self.config['device'])
        adverb_indices = torch.LongTensor([self.adverb_2_idx[a.lower()] for a in adverbs]).to(self.config['device'])
        return self.get_action_embeddings(action_indices), self.get_adverb_embeddings(adverb_indices)

    def get_parent_features(self, actions, adverbs):
        """Lookup parent embeddings for child actions/adverbs in batch."""
        hierarchy = self.config['hierarchy']

        action_parents = [hierarchy.get_action_parent(a) for a in actions]
        adverb_parents = [hierarchy.get_adverb_parent(a) for a in adverbs]

        action_parent_indices = torch.LongTensor([
            self.action_parent_2_idx[p.lower()] for p in action_parents
        ]).to(self.config['device'])

        adverb_parent_indices = torch.LongTensor([
            self.adverb_parent_2_idx[p.lower()] for p in adverb_parents
        ]).to(self.config['device'])

        return (
            self.get_action_parent_embeddings(action_parent_indices),
            self.get_adverb_parent_embeddings(adverb_parent_indices)
        )

    def forward(self, batch):
        flow = batch['flow_features'].to(self.config['device'])
        rgb = batch['rgb_features'].to(self.config['device'])
        video_features = self.get_video_features(flow, rgb)
        action_embeds, adverb_embeds = self.get_text_features(batch['action'], batch['adverb'])
        action_parent_embeds, adverb_parent_embeds = self.get_parent_features(batch['action'], batch['adverb'])
        return {
            'video_features': video_features,
            'action_embeds': action_embeds,
            'adverb_embeds': adverb_embeds,
            'action_parent_embeds': action_parent_embeds,
            'adverb_parent_embeds': adverb_parent_embeds,
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
        self.parent_contrastive_weight = config.get('parent_contrastive_weight', 0.5)
        self.parent_entailment_weight = config.get('parent_entailment_weight', 0.3)
        self.alpha = nn.Parameter(torch.tensor(config.get('output_dim', 512) ** -0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def entailment_loss(self, parent_embeds, child_embeds):
        _angle = L.oxy_angle(parent_embeds, child_embeds, self.curv.exp())
        _aperture = L.half_aperture(parent_embeds, self.curv.exp())
        return self.entail_weight * torch.clamp(_angle - _aperture, min=0).mean()

    def project(self, features):
        features = features * self.alpha.exp()
        return L.exp_map0(features, self.curv.exp())

    def forward(self, batch):
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Project all embeddings to hyperbolic space
        action_embeds_hyp = self.project(batch['action_embeds'])
        adverb_embeds_hyp = self.project(batch['adverb_embeds'])
        action_parent_embeds_hyp = self.project(batch['action_parent_embeds'])
        adverb_parent_embeds_hyp = self.project(batch['adverb_parent_embeds'])
        video_embeds_hyp = self.project(batch['video_features'])

        batch_size = video_embeds_hyp.size(0)
        labels = torch.arange(batch_size).to(video_embeds_hyp.device)

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=math.log(100))
        _scale = self.logit_scale.exp()

        # === VIDEO-LEVEL CONTRASTIVE LOSSES (video with child concepts) ===
        v2a = -L.pairwise_dist(video_embeds_hyp, action_embeds_hyp, _curv)
        a2v = -L.pairwise_dist(action_embeds_hyp, video_embeds_hyp, _curv)
        v2adv = -L.pairwise_dist(video_embeds_hyp, adverb_embeds_hyp, _curv)
        adv2v = -L.pairwise_dist(adverb_embeds_hyp, video_embeds_hyp, _curv)

        video_contrastive_loss = (
            nn.functional.cross_entropy(v2a * _scale, labels) +
            nn.functional.cross_entropy(a2v * _scale, labels) +
            nn.functional.cross_entropy(v2adv * _scale, labels) +
            nn.functional.cross_entropy(adv2v * _scale, labels)
        ) / 4

        # === HIERARCHY-LEVEL CONTRASTIVE LOSSES (parent with child concepts) ===
        ap2a = -L.pairwise_dist(action_parent_embeds_hyp, action_embeds_hyp, _curv)
        a2ap = -L.pairwise_dist(action_embeds_hyp, action_parent_embeds_hyp, _curv)
        advp2adv = -L.pairwise_dist(adverb_parent_embeds_hyp, adverb_embeds_hyp, _curv)
        adv2advp = -L.pairwise_dist(adverb_embeds_hyp, adverb_parent_embeds_hyp, _curv)

        hierarchy_contrastive_loss = self.parent_contrastive_weight * (
            nn.functional.cross_entropy(ap2a * _scale, labels) +
            nn.functional.cross_entropy(a2ap * _scale, labels) +
            nn.functional.cross_entropy(advp2adv * _scale, labels) +
            nn.functional.cross_entropy(adv2advp * _scale, labels)
        ) / 4

        contrastive_loss = video_contrastive_loss + hierarchy_contrastive_loss

        # === 2-LEVEL ENTAILMENT LOSSES ===
        # Parent → Child
        parent_to_child_entailment = (
            self.entailment_loss(action_parent_embeds_hyp, action_embeds_hyp) +
            self.entailment_loss(adverb_parent_embeds_hyp, adverb_embeds_hyp)
        ) * self.parent_entailment_weight

        # Child → Video
        child_to_video_entailment = (
            self.entailment_loss(action_embeds_hyp, video_embeds_hyp) +
            self.entailment_loss(adverb_embeds_hyp, video_embeds_hyp)
        )

        # Parent → Video (transitive)
        parent_to_video_entailment = (
            self.entailment_loss(action_parent_embeds_hyp, video_embeds_hyp) +
            self.entailment_loss(adverb_parent_embeds_hyp, video_embeds_hyp)
        ) * self.parent_entailment_weight

        entailment_loss = parent_to_child_entailment + child_to_video_entailment + parent_to_video_entailment

        loss = contrastive_loss + entailment_loss

        return {
            "loss": loss,
            'action_embeds_hyp': action_embeds_hyp,
            'adverb_embeds_hyp': adverb_embeds_hyp,
            'action_parent_embeds_hyp': action_parent_embeds_hyp,
            'adverb_parent_embeds_hyp': adverb_parent_embeds_hyp,
            'video_embeds_hyp': video_embeds_hyp,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "video_contrastive_loss": video_contrastive_loss,
                "hierarchy_contrastive_loss": hierarchy_contrastive_loss,
                "entailment_loss": entailment_loss,
                "parent_to_child_entailment": parent_to_child_entailment,
                "child_to_video_entailment": child_to_video_entailment,
                "parent_to_video_entailment": parent_to_video_entailment,
                "logit_scale": _scale,
                "curv": _curv,
                "video_2_action_logits": v2a,
                "video_2_adverb_logits": v2adv,
                "action_parent_2_action_logits": ap2a,
                "adverb_parent_2_adverb_logits": advp2adv,
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
        self.num_action_parents = len(hierarchy.get_all_parent_actions())
        self.num_adverb_parents = len(hierarchy.get_all_parent_adverbs())

        self.epoch_loss = AverageMeter()
        self.epoch_entail_loss = AverageMeter()
        self.epoch_contrastive_loss = AverageMeter()
        self.epoch_video_contrastive_loss = AverageMeter()
        self.epoch_hierarchy_contrastive_loss = AverageMeter()
        self.epoch_parent_to_child_entailment = AverageMeter()
        self.epoch_child_to_video_entailment = AverageMeter()
        self.epoch_parent_to_video_entailment = AverageMeter()
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

        if self.wandb_enabled:
            wandb.watch(model, log='all', log_freq=100, log_graph=True)

        if resume_from:
            self.load_checkpoint(resume_from)

    def train_one_epoch(self):
        self.model.train()
        self.epoch_loss.reset()
        self.epoch_contrastive_loss.reset()
        self.epoch_video_contrastive_loss.reset()
        self.epoch_hierarchy_contrastive_loss.reset()
        self.epoch_entail_loss.reset()
        self.epoch_parent_to_child_entailment.reset()
        self.epoch_child_to_video_entailment.reset()
        self.epoch_parent_to_video_entailment.reset()
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
            loss_video_contrastive = outputs['logging']['video_contrastive_loss']
            loss_hierarchy_contrastive = outputs['logging']['hierarchy_contrastive_loss']
            loss_entail = outputs['logging']['entailment_loss']
            loss_parent_to_child = outputs['logging']['parent_to_child_entailment']
            loss_child_to_video = outputs['logging']['child_to_video_entailment']
            loss_parent_to_video = outputs['logging']['parent_to_video_entailment']
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
            self.epoch_video_contrastive_loss.update(loss_video_contrastive.item())
            self.epoch_hierarchy_contrastive_loss.update(loss_hierarchy_contrastive.item())
            self.epoch_entail_loss.update(loss_entail.item())
            self.epoch_parent_to_child_entailment.update(loss_parent_to_child.item())
            self.epoch_child_to_video_entailment.update(loss_child_to_video.item())
            self.epoch_parent_to_video_entailment.update(loss_parent_to_video.item())

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
                    'step/loss_video_contrastive': loss_video_contrastive.item(),
                    'step/loss_hierarchy_contrastive': loss_hierarchy_contrastive.item(),
                    'step/loss_entail': loss_entail.item(),
                    'step/loss_parent_to_child': loss_parent_to_child.item(),
                    'step/loss_child_to_video': loss_child_to_video.item(),
                    'step/loss_parent_to_video': loss_parent_to_video.item(),
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
        avg_video_contrastive_loss = self.epoch_video_contrastive_loss.avg
        avg_hierarchy_contrastive_loss = self.epoch_hierarchy_contrastive_loss.avg
        avg_entail_loss = self.epoch_entail_loss.avg
        avg_parent_to_child = self.epoch_parent_to_child_entailment.avg
        avg_child_to_video = self.epoch_child_to_video_entailment.avg
        avg_parent_to_video = self.epoch_parent_to_video_entailment.avg
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
                'train/loss_video_contrastive': avg_video_contrastive_loss,
                'train/loss_hierarchy_contrastive': avg_hierarchy_contrastive_loss,
                'train/loss_entail': avg_entail_loss,
                'train/loss_parent_to_child': avg_parent_to_child,
                'train/loss_child_to_video': avg_child_to_video,
                'train/loss_parent_to_video': avg_parent_to_video,
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

        dist_action_parent_top1 = MulticlassAccuracy(num_classes=self.num_action_parents, top_k=1).to(self.device)
        dist_action_parent_top5 = MulticlassAccuracy(num_classes=self.num_action_parents, top_k=5).to(self.device)
        dist_adverb_parent_top1 = MulticlassAccuracy(num_classes=self.num_adverb_parents, top_k=1).to(self.device)
        dist_adverb_parent_top5 = MulticlassAccuracy(num_classes=self.num_adverb_parents, top_k=5).to(self.device)

        inner_action_parent_top1 = MulticlassAccuracy(num_classes=self.num_action_parents, top_k=1).to(self.device)
        inner_action_parent_top5 = MulticlassAccuracy(num_classes=self.num_action_parents, top_k=5).to(self.device)
        inner_adverb_parent_top1 = MulticlassAccuracy(num_classes=self.num_adverb_parents, top_k=1).to(self.device)
        inner_adverb_parent_top5 = MulticlassAccuracy(num_classes=self.num_adverb_parents, top_k=5).to(self.device)

        # Compositional accuracy tracker
        compositional_correct = 0
        total_samples = 0

        # Hyperbolic geometry metrics
        entailment_violations_parent_child = []  # Parent → Child violations
        entailment_violations_child_video = []   # Child → Video violations
        entailment_violations_parent_video = []  # Parent → Video violations
        transitivity_violations = []  # Transitivity violations (parent→child & child→video but not parent→video)
        cone_apertures_action_parent = []
        cone_apertures_adverb_parent = []
        depth_action = []
        depth_adverb = []
        depth_action_parent = []
        depth_adverb_parent = []
        depth_video = []

        val_video_norms = []

        with torch.no_grad():
            all_action_indices = torch.arange(self.num_actions, device=self.device)
            all_adverb_indices = torch.arange(self.num_adverbs, device=self.device)
            action_raw = self.model.aa_model.get_action_embeddings(all_action_indices)
            adverb_raw = self.model.aa_model.get_adverb_embeddings(all_adverb_indices)
            action_gallery = self.model.projection_module.project(action_raw)
            adverb_gallery = self.model.projection_module.project(adverb_raw)

            action_parent_indices = torch.arange(self.num_action_parents, device=self.device)
            adverb_parent_indices = torch.arange(self.num_adverb_parents, device=self.device)
            action_parent_raw = self.model.aa_model.get_action_parent_embeddings(action_parent_indices)
            adverb_parent_raw = self.model.aa_model.get_adverb_parent_embeddings(adverb_parent_indices)
            action_parent_gallery = self.model.projection_module.project(action_parent_raw)
            adverb_parent_gallery = self.model.projection_module.project(adverb_parent_raw)

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

                action_parent_scores_dist = -L.pairwise_dist(video_embeds, action_parent_gallery, _curv)
                adverb_parent_scores_dist = -L.pairwise_dist(video_embeds, adverb_parent_gallery, _curv)
                action_parent_scores_inner = L.pairwise_inner(video_embeds, action_parent_gallery, _curv)
                adverb_parent_scores_inner = L.pairwise_inner(video_embeds, adverb_parent_gallery, _curv)

                action_labels = torch.LongTensor(
                    [self.model.aa_model.action_2_idx[a.lower()] for a in batch_dict['action']]
                ).to(self.device)
                adverb_labels = torch.LongTensor(
                    [self.model.aa_model.adverb_2_idx[a.lower()] for a in batch_dict['adverb']]
                ).to(self.device)

                hierarchy = self.config['hierarchy']
                action_parent_labels = torch.LongTensor([
                    self.model.aa_model.action_parent_2_idx[hierarchy.get_action_parent(a).lower()]
                    for a in batch_dict['action']
                ]).to(self.device)
                adverb_parent_labels = torch.LongTensor([
                    self.model.aa_model.adverb_parent_2_idx[hierarchy.get_adverb_parent(a).lower()]
                    for a in batch_dict['adverb']
                ]).to(self.device)

                dist_action_top1.update(action_scores_dist, action_labels)
                dist_action_top5.update(action_scores_dist, action_labels)
                dist_adverb_top1.update(adverb_scores_dist, adverb_labels)
                dist_adverb_top5.update(adverb_scores_dist, adverb_labels)

                inner_action_top1.update(action_scores_inner, action_labels)
                inner_action_top5.update(action_scores_inner, action_labels)
                inner_adverb_top1.update(adverb_scores_inner, adverb_labels)
                inner_adverb_top5.update(adverb_scores_inner, adverb_labels)

                # Update parent metrics
                dist_action_parent_top1.update(action_parent_scores_dist, action_parent_labels)
                dist_action_parent_top5.update(action_parent_scores_dist, action_parent_labels)
                dist_adverb_parent_top1.update(adverb_parent_scores_dist, adverb_parent_labels)
                dist_adverb_parent_top5.update(adverb_parent_scores_dist, adverb_parent_labels)

                inner_action_parent_top1.update(action_parent_scores_inner, action_parent_labels)
                inner_action_parent_top5.update(action_parent_scores_inner, action_parent_labels)
                inner_adverb_parent_top1.update(adverb_parent_scores_inner, adverb_parent_labels)
                inner_adverb_parent_top5.update(adverb_parent_scores_inner, adverb_parent_labels)

                # Compositional accuracy (both action AND adverb correct)
                action_correct = (action_scores_dist.argmax(dim=1) == action_labels)
                adverb_correct = (adverb_scores_dist.argmax(dim=1) == adverb_labels)
                compositional_correct += (action_correct & adverb_correct).sum().item()
                total_samples += video_embeds.size(0)

                # Collect depth distributions (distance from origin)
                depth_video.append(embedding_norm(video_embeds).cpu())
                depth_action.append(embedding_norm(outputs['action_embeds_hyp']).cpu())
                depth_adverb.append(embedding_norm(outputs['adverb_embeds_hyp']).cpu())
                depth_action_parent.append(embedding_norm(outputs['action_parent_embeds_hyp']).cpu())
                depth_adverb_parent.append(embedding_norm(outputs['adverb_parent_embeds_hyp']).cpu())

                # Compute cone apertures for parents
                action_parent_apertures = L.half_aperture(outputs['action_parent_embeds_hyp'], _curv)
                adverb_parent_apertures = L.half_aperture(outputs['adverb_parent_embeds_hyp'], _curv)
                cone_apertures_action_parent.append(action_parent_apertures.cpu())
                cone_apertures_adverb_parent.append(adverb_parent_apertures.cpu())

                # Entailment violations: Check if angle > aperture (cone violation)
                # Parent → Child
                action_parent_2_child_angle = L.oxy_angle(
                    outputs['action_parent_embeds_hyp'],
                    outputs['action_embeds_hyp'],
                    _curv
                )
                adverb_parent_2_child_angle = L.oxy_angle(
                    outputs['adverb_parent_embeds_hyp'],
                    outputs['adverb_embeds_hyp'],
                    _curv
                )
                entailment_violations_parent_child.append(
                    (action_parent_2_child_angle > action_parent_apertures).float().cpu()
                )
                entailment_violations_parent_child.append(
                    (adverb_parent_2_child_angle > adverb_parent_apertures).float().cpu()
                )

                # Child → Video
                action_2_video_angle = L.oxy_angle(outputs['action_embeds_hyp'], video_embeds, _curv)
                adverb_2_video_angle = L.oxy_angle(outputs['adverb_embeds_hyp'], video_embeds, _curv)
                action_aperture = L.half_aperture(outputs['action_embeds_hyp'], _curv)
                adverb_aperture = L.half_aperture(outputs['adverb_embeds_hyp'], _curv)
                entailment_violations_child_video.append(
                    (action_2_video_angle > action_aperture).float().cpu()
                )
                entailment_violations_child_video.append(
                    (adverb_2_video_angle > adverb_aperture).float().cpu()
                )

                # Parent → Video (transitive)
                action_parent_2_video_angle = L.oxy_angle(
                    outputs['action_parent_embeds_hyp'],
                    video_embeds,
                    _curv
                )
                adverb_parent_2_video_angle = L.oxy_angle(
                    outputs['adverb_parent_embeds_hyp'],
                    video_embeds,
                    _curv
                )
                entailment_violations_parent_video.append(
                    (action_parent_2_video_angle > action_parent_apertures).float().cpu()
                )
                entailment_violations_parent_video.append(
                    (adverb_parent_2_video_angle > adverb_parent_apertures).float().cpu()
                )

                # Transitivity check: if parent→child and child→video, then parent→video should hold
                # Violation: (parent entails child) AND (child entails video) AND NOT (parent entails video)
                action_parent_entails_child = (action_parent_2_child_angle <= action_parent_apertures)
                action_child_entails_video = (action_2_video_angle <= action_aperture)
                action_parent_entails_video = (action_parent_2_video_angle <= action_parent_apertures)
                action_transitivity_violations = (action_parent_entails_child & action_child_entails_video & ~action_parent_entails_video).float()

                adverb_parent_entails_child = (adverb_parent_2_child_angle <= adverb_parent_apertures)
                adverb_child_entails_video = (adverb_2_video_angle <= adverb_aperture)
                adverb_parent_entails_video = (adverb_parent_2_video_angle <= adverb_parent_apertures)
                adverb_transitivity_violations = (adverb_parent_entails_child & adverb_child_entails_video & ~adverb_parent_entails_video).float()

                transitivity_violations.append(action_transitivity_violations.cpu())
                transitivity_violations.append(adverb_transitivity_violations.cpu())

        avg_val_loss = val_loss_meter.avg

        # Child accuracies
        d_act1 = dist_action_top1.compute().item()
        d_act5 = dist_action_top5.compute().item()
        d_adv1 = dist_adverb_top1.compute().item()
        d_adv5 = dist_adverb_top5.compute().item()

        i_act1 = inner_action_top1.compute().item()
        i_act5 = inner_action_top5.compute().item()
        i_adv1 = inner_adverb_top1.compute().item()
        i_adv5 = inner_adverb_top5.compute().item()

        # Parent accuracies
        d_act_parent1 = dist_action_parent_top1.compute().item()
        d_act_parent5 = dist_action_parent_top5.compute().item()
        d_adv_parent1 = dist_adverb_parent_top1.compute().item()
        d_adv_parent5 = dist_adverb_parent_top5.compute().item()

        i_act_parent1 = inner_action_parent_top1.compute().item()
        i_act_parent5 = inner_action_parent_top5.compute().item()
        i_adv_parent1 = inner_adverb_parent_top1.compute().item()
        i_adv_parent5 = inner_adverb_parent_top5.compute().item()

        # Compositional accuracy
        compositional_acc = compositional_correct / total_samples if total_samples > 0 else 0.0

        # Compute hyperbolic geometry metrics
        all_entailment_violations_pc = torch.cat(entailment_violations_parent_child).numpy()
        all_entailment_violations_cv = torch.cat(entailment_violations_child_video).numpy()
        all_entailment_violations_pv = torch.cat(entailment_violations_parent_video).numpy()
        all_transitivity_violations = torch.cat(transitivity_violations).numpy()

        violation_rate_parent_child = all_entailment_violations_pc.mean()
        violation_rate_child_video = all_entailment_violations_cv.mean()
        violation_rate_parent_video = all_entailment_violations_pv.mean()
        transitivity_violation_rate = all_transitivity_violations.mean()

        all_cone_apertures_action = torch.cat(cone_apertures_action_parent).numpy()
        all_cone_apertures_adverb = torch.cat(cone_apertures_adverb_parent).numpy()
        avg_cone_aperture_action = all_cone_apertures_action.mean()
        avg_cone_aperture_adverb = all_cone_apertures_adverb.mean()

        all_depth_video = torch.cat(depth_video).numpy()
        all_depth_action = torch.cat(depth_action).numpy()
        all_depth_adverb = torch.cat(depth_adverb).numpy()
        all_depth_action_parent = torch.cat(depth_action_parent).numpy()
        all_depth_adverb_parent = torch.cat(depth_adverb_parent).numpy()

        self.val_losses.append(avg_val_loss)

        print('E %d | Val Loss: %.4f' % (self.current_epoch, avg_val_loss))
        print('  [dist]  Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (d_act1, d_act5, d_adv1, d_adv5))
        print('  [inner] Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (i_act1, i_act5, i_adv1, i_adv5))
        print('  [parent-dist] Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (d_act_parent1, d_act_parent5, d_adv_parent1, d_adv_parent5))
        print('  [parent-inner] Action Top-1: %.4f Top-5: %.4f | Adverb Top-1: %.4f Top-5: %.4f' % (i_act_parent1, i_act_parent5, i_adv_parent1, i_adv_parent5))
        print('  [compositional] Both Action+Adverb Correct: %.4f' % compositional_acc)
        print('  [entailment violations] Parent→Child: %.4f | Child→Video: %.4f | Parent→Video: %.4f | Transitivity: %.4f' %
              (violation_rate_parent_child, violation_rate_child_video, violation_rate_parent_video, transitivity_violation_rate))
        print('  [cone apertures] Action Parent: %.4f | Adverb Parent: %.4f' % (avg_cone_aperture_action, avg_cone_aperture_adverb))
        print('  [depth] Video: %.4f±%.4f | Action: %.4f±%.4f | Adverb: %.4f±%.4f' %
              (all_depth_video.mean(), all_depth_video.std(), all_depth_action.mean(), all_depth_action.std(),
               all_depth_adverb.mean(), all_depth_adverb.std()))
        print('  [depth] Action Parent: %.4f±%.4f | Adverb Parent: %.4f±%.4f' %
              (all_depth_action_parent.mean(), all_depth_action_parent.std(),
               all_depth_adverb_parent.mean(), all_depth_adverb_parent.std()))

        if self.wandb_enabled:
            wandb.log({
                'epoch': self.current_epoch,
                'val/loss_total': avg_val_loss,
                'val/loss_contrastive': val_contrastive_loss_meter.avg,
                'val/loss_entail': val_entail_loss_meter.avg,

                # Child metrics (dist)
                'val/dist_action_top1': d_act1,
                'val/dist_action_top5': d_act5,
                'val/dist_adverb_top1': d_adv1,
                'val/dist_adverb_top5': d_adv5,
                'val/dist_combined_top1': (d_act1 + d_adv1) / 2,

                # Child metrics (inner)
                'val/inner_action_top1': i_act1,
                'val/inner_action_top5': i_act5,
                'val/inner_adverb_top1': i_adv1,
                'val/inner_adverb_top5': i_adv5,
                'val/inner_combined_top1': (i_act1 + i_adv1) / 2,

                # Parent metrics (dist)
                'val/dist_action_parent_top1': d_act_parent1,
                'val/dist_action_parent_top5': d_act_parent5,
                'val/dist_adverb_parent_top1': d_adv_parent1,
                'val/dist_adverb_parent_top5': d_adv_parent5,
                'val/dist_parent_combined_top1': (d_act_parent1 + d_adv_parent1) / 2,
                'val/dist_parent_combined_top5': (d_act_parent5 + d_adv_parent5) / 2,

                # Parent metrics (inner)
                'val/inner_action_parent_top1': i_act_parent1,
                'val/inner_action_parent_top5': i_act_parent5,
                'val/inner_adverb_parent_top1': i_adv_parent1,
                'val/inner_adverb_parent_top5': i_adv_parent5,
                'val/inner_parent_combined_top1': (i_act_parent1 + i_adv_parent1) / 2,
                'val/inner_parent_combined_top5': (i_act_parent5 + i_adv_parent5) / 2,

                # Compositional accuracy
                'val/compositional_accuracy': compositional_acc,

                # Entailment violations
                'val/entailment_violation_parent_to_child': violation_rate_parent_child,
                'val/entailment_violation_child_to_video': violation_rate_child_video,
                'val/entailment_violation_parent_to_video': violation_rate_parent_video,
                'val/transitivity_violation_rate': transitivity_violation_rate,

                # Cone apertures
                'val/cone_aperture_action_parent_mean': avg_cone_aperture_action,
                'val/cone_aperture_adverb_parent_mean': avg_cone_aperture_adverb,
                'val/cone_aperture_action_parent': wandb.Histogram(all_cone_apertures_action),
                'val/cone_aperture_adverb_parent': wandb.Histogram(all_cone_apertures_adverb),

                # Depth distributions
                'val/depth_video_mean': all_depth_video.mean(),
                'val/depth_video_std': all_depth_video.std(),
                'val/depth_action_mean': all_depth_action.mean(),
                'val/depth_action_std': all_depth_action.std(),
                'val/depth_adverb_mean': all_depth_adverb.mean(),
                'val/depth_adverb_std': all_depth_adverb.std(),
                'val/depth_action_parent_mean': all_depth_action_parent.mean(),
                'val/depth_action_parent_std': all_depth_action_parent.std(),
                'val/depth_adverb_parent_mean': all_depth_adverb_parent.mean(),
                'val/depth_adverb_parent_std': all_depth_adverb_parent.std(),

                # Embedding norm histograms
                'val/norm_action_vocab': wandb.Histogram(embedding_norm(action_gallery).cpu().numpy()),
                'val/norm_adverb_vocab': wandb.Histogram(embedding_norm(adverb_gallery).cpu().numpy()),
                'val/norm_action_parent_vocab': wandb.Histogram(embedding_norm(action_parent_gallery).cpu().numpy()),
                'val/norm_adverb_parent_vocab': wandb.Histogram(embedding_norm(adverb_parent_gallery).cpu().numpy()),
                'val/norm_video': wandb.Histogram(torch.cat(val_video_norms).numpy()),
                'val/norm_depth_video': wandb.Histogram(all_depth_video),
                'val/norm_depth_action': wandb.Histogram(all_depth_action),
                'val/norm_depth_adverb': wandb.Histogram(all_depth_adverb),
                'val/norm_depth_action_parent': wandb.Histogram(all_depth_action_parent),
                'val/norm_depth_adverb_parent': wandb.Histogram(all_depth_adverb_parent),
            })

        if d_act1 > self.best_action_acc:
            self.best_action_acc = d_act1
        if d_adv1 > self.best_adverb_acc:
            self.best_adverb_acc = d_adv1

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

    # Hierarchical loss weights
    parser.add_argument('--parent-contrastive-weight', type=float, default=0.5,
                        help='Weight for parent-level contrastive loss')
    parser.add_argument('--parent-entailment-weight', type=float, default=0.3,
                        help='Weight for parent-level entailment losses')

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
        'parent_contrastive_weight': args.parent_contrastive_weight,
        'parent_entailment_weight': args.parent_entailment_weight,
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
