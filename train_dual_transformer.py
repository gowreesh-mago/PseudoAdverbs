"""
Dual Transformer Action-Adverb Classifier Training Script

This script implements two separate transformer encoders for action and adverb classification.
Uses CLS tokens for classification and tracks top-1 and top-5 accuracies for both tasks.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import tqdm
import wandb
import numpy as np

from dataset import AdverbDataset
from opts import parser
from wandb_config import init_wandb_with_config


# =============================================================================
# Helper Functions
# =============================================================================

def extract_action_adverb_labels(labels, dset):
    """
    Extract action and adverb indices from composite class labels.

    Args:
        labels: (B,) tensor of class labels
        dset: Dataset object with idx2class, action2idx, adverb2idx mappings

    Returns:
        gt_actions: (B,) tensor of action indices
        gt_adverbs: (B,) tensor of adverb indices
    """
    gt_actions = []
    gt_adverbs = []

    for label_idx in labels.cpu().numpy():
        action, adverb = dset.idx2class[label_idx]
        gt_actions.append(dset.action2idx[action])
        gt_adverbs.append(dset.adverb2idx[adverb])

    gt_actions = torch.tensor(gt_actions, device=labels.device)
    gt_adverbs = torch.tensor(gt_adverbs, device=labels.device)

    return gt_actions, gt_adverbs


def calculate_joint_accuracy(action_logits, adverb_logits, labels, dset):
    """
    Calculate top-1 accuracies for action, adverb, and joint predictions.

    Args:
        action_logits: (B, num_actions) tensor
        adverb_logits: (B, num_adverbs) tensor
        labels: (B,) tensor of class labels
        dset: Dataset object

    Returns:
        joint_accuracy: float
        action_accuracy: float
        adverb_accuracy: float
    """
    # Get predictions
    pred_actions = torch.argmax(action_logits, dim=1)
    pred_adverbs = torch.argmax(adverb_logits, dim=1)

    # Extract ground truth
    gt_actions, gt_adverbs = extract_action_adverb_labels(labels, dset)

    # Calculate accuracies
    action_correct = (pred_actions == gt_actions).float()
    adverb_correct = (pred_adverbs == gt_adverbs).float()
    joint_correct = (action_correct * adverb_correct)  # Both must be correct

    action_accuracy = action_correct.mean().item()
    adverb_accuracy = adverb_correct.mean().item()
    joint_accuracy = joint_correct.mean().item()

    return joint_accuracy, action_accuracy, adverb_accuracy


def calculate_top_k_accuracy(action_logits, adverb_logits, labels, k, dset):
    """
    Calculate top-k accuracies for action, adverb, and joint predictions.

    Args:
        action_logits: (B, num_actions) tensor
        adverb_logits: (B, num_adverbs) tensor
        labels: (B,) tensor of class labels
        k: int, top-k value
        dset: Dataset object

    Returns:
        top_k_joint_acc: float
        top_k_action_acc: float
        top_k_adverb_acc: float
    """
    batch_size = labels.size(0)

    # Get top-k predictions
    _, top_k_actions = torch.topk(action_logits, k, dim=1)  # (B, k)
    _, top_k_adverbs = torch.topk(adverb_logits, k, dim=1)  # (B, k)

    # Extract ground truth
    gt_actions, gt_adverbs = extract_action_adverb_labels(labels, dset)

    # Check if ground truth is in top-k
    gt_actions_expanded = gt_actions.unsqueeze(1).expand(-1, k)  # (B, k)
    gt_adverbs_expanded = gt_adverbs.unsqueeze(1).expand(-1, k)  # (B, k)

    action_in_topk = (top_k_actions == gt_actions_expanded).any(dim=1).float()  # (B,)
    adverb_in_topk = (top_k_adverbs == gt_adverbs_expanded).any(dim=1).float()  # (B,)
    joint_in_topk = (action_in_topk * adverb_in_topk)  # Both must be in top-k

    top_k_action_acc = action_in_topk.mean().item()
    top_k_adverb_acc = adverb_in_topk.mean().item()
    top_k_joint_acc = joint_in_topk.mean().item()

    return top_k_joint_acc, top_k_action_acc, top_k_adverb_acc


def create_padding_mask(pad, seq_len):
    """
    Create padding mask for transformer.

    Args:
        pad: (B,) tensor of padding lengths
        seq_len: Total sequence length (T+1 for CLS)

    Returns:
        mask: (B, seq_len) where True = valid token, False = padding
    """
    batch_size = pad.size(0)
    mask = torch.arange(seq_len, device=pad.device).expand(batch_size, seq_len)
    # CLS token (position 0) is always valid
    # Positions 1 to (seq_len - pad) are valid
    mask = mask < (seq_len - pad.unsqueeze(1))
    return mask


# =============================================================================
# Model Classes
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DualTransformerClassifier(nn.Module):
    """
    Dual transformer model with separate encoders for action and adverb classification.
    Uses learnable CLS tokens for classification.
    """
    def __init__(self, dset, args):
        super(DualTransformerClassifier, self).__init__()

        self.dset = dset
        self.num_actions = len(dset.action2idx)
        self.num_adverbs = len(dset.adverb2idx)
        self.d_model = args.d_model
        self.shared_input_projection = args.shared_input_projection
        self.shared_pos_encoding = args.shared_pos_encoding

        # CLS tokens (learnable)
        self.action_cls_token = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.adverb_cls_token = nn.Parameter(torch.randn(1, 1, args.d_model))

        # Input projections
        if self.shared_input_projection:
            self.input_projection = nn.Linear(2048, args.d_model)
        else:
            self.action_input_projection = nn.Linear(2048, args.d_model)
            self.adverb_input_projection = nn.Linear(2048, args.d_model)

        # Positional encodings
        if self.shared_pos_encoding:
            self.pos_encoder = PositionalEncoding(args.d_model, dropout=args.transformer_dropout)
        else:
            self.action_pos_encoder = PositionalEncoding(args.d_model, dropout=args.transformer_dropout)
            self.adverb_pos_encoder = PositionalEncoding(args.d_model, dropout=args.transformer_dropout)

        # Transformer encoders
        action_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.transformer_nhead,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.transformer_dropout,
            batch_first=False
        )
        self.action_transformer = nn.TransformerEncoder(
            action_encoder_layer,
            num_layers=args.num_transformer_layers
        )

        adverb_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.transformer_nhead,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.transformer_dropout,
            batch_first=False
        )
        self.adverb_transformer = nn.TransformerEncoder(
            adverb_encoder_layer,
            num_layers=args.num_transformer_layers
        )

        # Classification heads
        self.action_classifier = nn.Linear(args.d_model, self.num_actions)
        self.adverb_classifier = nn.Linear(args.d_model, self.num_adverbs)

    def forward(self, features, pad):
        """
        Forward pass through dual transformer.

        Args:
            features: (B, T, 2048) - Concatenated RGB+Flow features
            pad: (B,) - Padding lengths for each sequence

        Returns:
            action_logits: (B, num_actions)
            adverb_logits: (B, num_adverbs)
        """
        batch_size, seq_len, feat_dim = features.shape

        # Project to d_model
        if self.shared_input_projection:
            features_proj = self.input_projection(features)
            features_proj_action = features_proj
            features_proj_adverb = features_proj
        else:
            features_proj_action = self.action_input_projection(features)
            features_proj_adverb = self.adverb_input_projection(features)

        # Prepare CLS tokens
        action_cls = self.action_cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        adverb_cls = self.adverb_cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)

        # Concatenate CLS with features
        action_input = torch.cat([action_cls, features_proj_action], dim=1)  # (B, T+1, d_model)
        adverb_input = torch.cat([adverb_cls, features_proj_adverb], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        if self.shared_pos_encoding:
            action_input = self.pos_encoder(action_input)
            adverb_input = self.pos_encoder(adverb_input)
        else:
            action_input = self.action_pos_encoder(action_input)
            adverb_input = self.adverb_pos_encoder(adverb_input)

        # Create padding masks (include CLS token)
        src_key_padding_mask = create_padding_mask(pad, seq_len + 1)
        src_key_padding_mask = ~src_key_padding_mask  # Invert for transformer convention

        # Transpose to (T+1, B, d_model) for transformer
        action_input = action_input.transpose(0, 1)  # (T+1, B, d_model)
        adverb_input = adverb_input.transpose(0, 1)  # (T+1, B, d_model)

        # Pass through transformers
        action_encoded = self.action_transformer(
            action_input,
            src_key_padding_mask=src_key_padding_mask
        )
        adverb_encoded = self.adverb_transformer(
            adverb_input,
            src_key_padding_mask=src_key_padding_mask
        )

        # Back to (B, T+1, d_model) and extract CLS
        action_encoded = action_encoded.transpose(0, 1)  # (B, T+1, d_model)
        adverb_encoded = adverb_encoded.transpose(0, 1)  # (B, T+1, d_model)

        action_cls_output = action_encoded[:, 0, :]  # (B, d_model)
        adverb_cls_output = adverb_encoded[:, 0, :]  # (B, d_model)

        # Classification
        action_logits = self.action_classifier(action_cls_output)  # (B, num_actions)
        adverb_logits = self.adverb_classifier(adverb_cls_output)  # (B, num_adverbs)

        return action_logits, adverb_logits


# =============================================================================
# Training and Testing Functions
# =============================================================================

def train_dual_transformer(model, train_loader, optimizer, criterion, writer, epoch, args):
    """Training function for dual transformer."""
    model.train()

    # Tracking
    total_action_loss = 0.0
    total_adverb_loss = 0.0
    total_combined_loss = 0.0
    total_action_acc = 0.0
    total_adverb_acc = 0.0
    total_joint_acc = 0.0

    all_action_logits = []
    all_adverb_logits = []
    all_labels = []

    for idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch}"):
        features = data[0].cuda()
        labels = data[1].cuda()
        pad = data[2].cuda()

        # Forward
        action_logits, adverb_logits = model(features, pad)

        # Extract ground truth
        gt_actions, gt_adverbs = extract_action_adverb_labels(labels, model.dset)

        # Losses
        action_loss = criterion(action_logits, gt_actions)
        adverb_loss = criterion(adverb_logits, gt_adverbs)
        combined_loss = args.action_weight * action_loss + args.adverb_weight * adverb_loss

        # Top-1 Accuracies
        joint_acc, action_acc, adverb_acc = calculate_joint_accuracy(
            action_logits, adverb_logits, labels, model.dset
        )

        # Top-5 Accuracies (batch-level)
        top5_joint_acc_batch, top5_action_acc_batch, top5_adverb_acc_batch = calculate_top_k_accuracy(
            action_logits, adverb_logits, labels, k=5, dset=model.dset
        )

        # Backward
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        # Update metrics
        total_action_loss += action_loss.item()
        total_adverb_loss += adverb_loss.item()
        total_combined_loss += combined_loss.item()
        total_action_acc += action_acc
        total_adverb_acc += adverb_acc
        total_joint_acc += joint_acc

        # Store for epoch metrics
        all_action_logits.append(action_logits.detach())
        all_adverb_logits.append(adverb_logits.detach())
        all_labels.append(labels)

        # Step-level wandb logging
        if not args.no_wandb:
            wandb.log({
                'step/action_loss': action_loss.item(),
                'step/adverb_loss': adverb_loss.item(),
                'step/combined_loss': combined_loss.item(),
                'step/top1_action_accuracy': action_acc,
                'step/top1_adverb_accuracy': adverb_acc,
                'step/top1_joint_accuracy': joint_acc,
                'step/top5_action_accuracy': top5_action_acc_batch,
                'step/top5_adverb_accuracy': top5_adverb_acc_batch,
                'step/top5_joint_accuracy': top5_joint_acc_batch,
            })

    # Epoch-level metrics
    num_batches = len(train_loader)
    avg_action_loss = total_action_loss / num_batches
    avg_adverb_loss = total_adverb_loss / num_batches
    avg_combined_loss = total_combined_loss / num_batches
    avg_action_acc = total_action_acc / num_batches
    avg_adverb_acc = total_adverb_acc / num_batches
    avg_joint_acc = total_joint_acc / num_batches

    # Top-5 accuracies
    all_action_logits = torch.cat(all_action_logits)
    all_adverb_logits = torch.cat(all_adverb_logits)
    all_labels = torch.cat(all_labels)
    top5_joint_acc, top5_action_acc, top5_adverb_acc = calculate_top_k_accuracy(
        all_action_logits, all_adverb_logits, all_labels, k=5, dset=model.dset
    )

    # WandB epoch logging
    if not args.no_wandb:
        wandb.log({
            'epoch': epoch,
            'train/action_loss': avg_action_loss,
            'train/adverb_loss': avg_adverb_loss,
            'train/combined_loss': avg_combined_loss,
            'train/top1_action_accuracy': avg_action_acc,
            'train/top1_adverb_accuracy': avg_adverb_acc,
            'train/top1_joint_accuracy': avg_joint_acc,
            'train/top5_action_accuracy': top5_action_acc,
            'train/top5_adverb_accuracy': top5_adverb_acc,
            'train/top5_joint_accuracy': top5_joint_acc,
        })

    # TensorBoard logging
    writer.add_scalar('Loss/Train/Action', avg_action_loss, epoch)
    writer.add_scalar('Loss/Train/Adverb', avg_adverb_loss, epoch)
    writer.add_scalar('Loss/Train/Combined', avg_combined_loss, epoch)
    writer.add_scalar('Acc/Train/Top1_Action', avg_action_acc, epoch)
    writer.add_scalar('Acc/Train/Top1_Adverb', avg_adverb_acc, epoch)
    writer.add_scalar('Acc/Train/Top1_Joint', avg_joint_acc, epoch)
    writer.add_scalar('Acc/Train/Top5_Action', top5_action_acc, epoch)
    writer.add_scalar('Acc/Train/Top5_Adverb', top5_adverb_acc, epoch)
    writer.add_scalar('Acc/Train/Top5_Joint', top5_joint_acc, epoch)

    print(f'E: {epoch} | Train')
    print(f'  Loss - Action: {avg_action_loss:.4f} | Adverb: {avg_adverb_loss:.4f} | Combined: {avg_combined_loss:.4f}')
    print(f'  Top-1 - Joint: {avg_joint_acc:.4f} | Action: {avg_action_acc:.4f} | Adverb: {avg_adverb_acc:.4f}')
    print(f'  Top-5 - Joint: {top5_joint_acc:.4f} | Action: {top5_action_acc:.4f} | Adverb: {top5_adverb_acc:.4f}')


def test_dual_transformer(model, test_loader, criterion, writer, epoch, args):
    """Testing function for dual transformer."""
    model.eval()

    total_action_loss = 0.0
    total_adverb_loss = 0.0
    total_combined_loss = 0.0
    total_action_acc = 0.0
    total_adverb_acc = 0.0
    total_joint_acc = 0.0

    all_action_logits = []
    all_adverb_logits = []
    all_labels = []

    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test Epoch {epoch}"):
            features = data[0].cuda()
            labels = data[1].cuda()
            pad = data[2].cuda()

            # Forward
            action_logits, adverb_logits = model(features, pad)

            # Extract ground truth
            gt_actions, gt_adverbs = extract_action_adverb_labels(labels, model.dset)

            # Losses
            action_loss = criterion(action_logits, gt_actions)
            adverb_loss = criterion(adverb_logits, gt_adverbs)
            combined_loss = args.action_weight * action_loss + args.adverb_weight * adverb_loss

            # Accuracies
            joint_acc, action_acc, adverb_acc = calculate_joint_accuracy(
                action_logits, adverb_logits, labels, model.dset
            )

            # Update metrics
            total_action_loss += action_loss.item()
            total_adverb_loss += adverb_loss.item()
            total_combined_loss += combined_loss.item()
            total_action_acc += action_acc
            total_adverb_acc += adverb_acc
            total_joint_acc += joint_acc

            # Store for epoch metrics
            all_action_logits.append(action_logits)
            all_adverb_logits.append(adverb_logits)
            all_labels.append(labels)

    # Epoch-level metrics
    num_batches = len(test_loader)
    avg_action_loss = total_action_loss / num_batches
    avg_adverb_loss = total_adverb_loss / num_batches
    avg_combined_loss = total_combined_loss / num_batches
    avg_action_acc = total_action_acc / num_batches
    avg_adverb_acc = total_adverb_acc / num_batches
    avg_joint_acc = total_joint_acc / num_batches

    # Top-5 accuracies
    all_action_logits = torch.cat(all_action_logits)
    all_adverb_logits = torch.cat(all_adverb_logits)
    all_labels = torch.cat(all_labels)
    top5_joint_acc, top5_action_acc, top5_adverb_acc = calculate_top_k_accuracy(
        all_action_logits, all_adverb_logits, all_labels, k=5, dset=model.dset
    )

    # WandB logging
    if not args.no_wandb:
        wandb.log({
            'epoch': epoch,
            'test/action_loss': avg_action_loss,
            'test/adverb_loss': avg_adverb_loss,
            'test/combined_loss': avg_combined_loss,
            'test/top1_action_accuracy': avg_action_acc,
            'test/top1_adverb_accuracy': avg_adverb_acc,
            'test/top1_joint_accuracy': avg_joint_acc,
            'test/top5_action_accuracy': top5_action_acc,
            'test/top5_adverb_accuracy': top5_adverb_acc,
            'test/top5_joint_accuracy': top5_joint_acc,
        })

    # TensorBoard logging
    writer.add_scalar('Loss/Test/Action', avg_action_loss, epoch)
    writer.add_scalar('Loss/Test/Adverb', avg_adverb_loss, epoch)
    writer.add_scalar('Loss/Test/Combined', avg_combined_loss, epoch)
    writer.add_scalar('Acc/Test/Top1_Action', avg_action_acc, epoch)
    writer.add_scalar('Acc/Test/Top1_Adverb', avg_adverb_acc, epoch)
    writer.add_scalar('Acc/Test/Top1_Joint', avg_joint_acc, epoch)
    writer.add_scalar('Acc/Test/Top5_Action', top5_action_acc, epoch)
    writer.add_scalar('Acc/Test/Top5_Adverb', top5_adverb_acc, epoch)
    writer.add_scalar('Acc/Test/Top5_Joint', top5_joint_acc, epoch)

    print(f'E: {epoch} | Test')
    print(f'  Loss - Action: {avg_action_loss:.4f} | Adverb: {avg_adverb_loss:.4f} | Combined: {avg_combined_loss:.4f}')
    print(f'  Top-1 - Joint: {avg_joint_acc:.4f} | Action: {avg_action_acc:.4f} | Adverb: {avg_adverb_acc:.4f}')
    print(f'  Top-5 - Joint: {top5_joint_acc:.4f} | Action: {top5_action_acc:.4f} | Adverb: {top5_adverb_acc:.4f}')

    return avg_joint_acc  # For scheduler


# =============================================================================
# Main Function
# =============================================================================

def main(args):
    """Main training function."""
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize wandb
    init_wandb_with_config(
        project_name="pseudo-adverbs-dual-transformer",
        run_name=args.wandb_run_name,
        model_config={**vars(args), 'model_type': 'dual_transformer'},
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.wandb_config,
        disable_wandb=args.no_wandb
    )

    # Create datasets
    print("Loading training dataset...")
    train_set = AdverbDataset(
        args.data_dir,
        args.train_feature_dir,
        agg='sdp',  # MUST be 'sdp' for transformer
        modality=args.modality,
        window_size=args.t_train,
        phase='train',
        load_in_memory=args.load_in_memory,
        unlabelled_ratio=0,  # No unlabelled data
        classification_mode=True,
        class_mode=args.class_mode
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    print("Loading test dataset...")
    test_set = AdverbDataset(
        args.data_dir,
        args.test_feature_dir,
        agg='sdp',
        modality=args.modality,
        window_size=args.t_test,
        phase='test',
        load_in_memory=args.load_in_memory,
        unlabelled_ratio=0,
        classification_mode=True,
        class_mode=args.class_mode
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    print(f"Train set: {len(train_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    print(f"Number of actions: {len(train_set.action2idx)}")
    print(f"Number of adverbs: {len(train_set.adverb2idx)}")
    print(f"Number of classes: {train_set.num_classes}")

    # Create model
    print("Creating dual transformer model...")
    model = DualTransformerClassifier(train_set, args).cuda()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Load checkpoint if specified
    start_epoch = 0
    if args.load is not None:
        print(f"Loading checkpoint from {args.load}")
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'log'))

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    test_dual_transformer(model, test_loader, criterion, writer, start_epoch, args)

    for epoch in range(start_epoch, start_epoch + args.max_epochs + 1):
        train_dual_transformer(model, train_loader, optimizer, criterion, writer, epoch, args)

        if epoch % args.eval_interval == 0:
            joint_acc = test_dual_transformer(model, test_loader, criterion, writer, epoch, args)
            scheduler.step(joint_acc)  # Update LR based on joint accuracy

        if epoch % args.save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{epoch}.pth')
            torch.save({
                'net': model.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    if not args.no_wandb:
        wandb.finish()

    print("\nTraining completed!")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
