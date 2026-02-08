import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_word_embeddings(emb_file, vocab):
    vocab = [word.lower() for word in vocab]

    embeddings = {}
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word_vec = torch.FloatTensor(list(map(float, line[1:])))
            embeddings[line[0]] = word_vec
    embeddings = [embeddings[word] for word in vocab]
    embeddings = torch.stack(embeddings)
    print('loaded word embeddings')
    return embeddings

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        network = []
        for i in range(num_layers-1):
            network.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            network.append(nn.ReLU(True))
        network.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            network.append(nn.ReLU(True))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        output = self.network(x)
        return output

class SDPAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, emb_dim, heads=1, dropout=0.1):
        super(SDPAttention, self).__init__()
        self.d_k = int(d_k/heads)
        self.d_v = int(d_v/heads)
        d_model = int(d_model/heads)
        self.q_linear = nn.Linear(int(emb_dim/heads), self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.v_linear = nn.Linear(d_model, self.d_v)
        self.h = heads

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(heads*self.d_v, emb_dim)

    def attention(self, q, k, v, dropout=None, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores.masked_fill_(~mask.unsqueeze(1).unsqueeze(1).repeat(1, self.h, 1, 1), float("-inf"))
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        output = output.reshape(output.shape[0], output.shape[1], output.shape[-1])
        return output, scores

    def forward(self, features, queries, mask=None):
        bs = queries.shape[0]
        q = self.q_linear(queries.view(bs, -1, self.h, int(queries.shape[-1]/self.h)))
        k = self.k_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        v = self.v_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        output, scores = self.attention(q, k, v, self.dropout, mask=mask)
        concat = output.transpose(1,2).contiguous().view(bs, self.d_v*self.h)
        output = self.out(concat)
        return output, scores

    def run_attention(self, features, attention):
        bs = features.shape[0]
        v = self.v_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        v = v.transpose(1,2)

        output = torch.matmul(attention, v)
        output = output.reshape(output.shape[0], output.shape[1], output.shape[-1])
        concat = output.transpose(1,2).contiguous().view(bs, self.d_v*self.h)
        output = self.out(concat)
        return output, attention

class ActionModifiers(nn.Module):
    def __init__(self, dset, args):
        super(ActionModifiers, self).__init__()
        self.num_heads = 4
        if args.temporal_agg == 'sdp':
            self.video_embedder = SDPAttention(dset.feature_dim, args.emb_dim, args.emb_dim, 
                                               args.emb_dim, heads=self.num_heads)
        else:
            self.video_embedder = MLP(dset.feature_dim, args.emb_dim)

        self.action_modifiers = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim))
                                            for _ in range(len(dset.adverbs))])
        self.action_embedder = nn.Embedding(len(dset.actions), args.emb_dim)

        if args.glove_init:
            pretrained_weight = load_word_embeddings(args.glove_path, dset.actions)
            self.action_embedder.weight.data.copy_(pretrained_weight)

        for param in self.action_embedder.parameters():
            param.requires_grad = False

        self.margin = 0.5
        self.transformer = False
        if args.temporal_agg == 'sdp':
            self.transformer = True

        self.compare_metric = lambda vid_feats, act_adv_embed: -F.pairwise_distance(vid_feats, act_adv_embed)
        self.dset = dset

        ## precompute validation pairs
        adverbs, actions = zip(*self.dset.pairs)
        self.val_adverbs = torch.LongTensor([dset.adverb2idx[adv.strip()] for adv in adverbs]).cuda()
        self.adverbs = torch.LongTensor([dset.adverb2idx[adv.strip()] for adv in self.dset.adverbs]).cuda()
        self.val_actions = torch.LongTensor([dset.action2idx[act.strip()] for act in actions]).cuda()

    def apply_modifiers(self, modifiers, embedding):
        output = torch.bmm(modifiers, embedding.unsqueeze(2)).squeeze(2)
        output = F.relu(output)
        return output

    def train_forward(self, x, threshold_adverbs=None, attention=None):
        features, adverbs, actions = x[0], x[1], x[2]
        neg_adverbs, neg_actions = x[4], x[5]
        batch_size = features.shape[0]
        temporal_dim = features.shape[1]
        pad=x[3]
        mask = torch.arange(temporal_dim).expand(len(pad), temporal_dim).cuda() < temporal_dim - pad.unsqueeze(1)
        action_embedding = self.action_embedder(actions)
        neg_action_embedding = self.action_embedder(neg_actions)
        if self.transformer:
            if attention is not None:
                video_embedding, attention_weights = self.video_embedder.run_attention(features, attention)
            else:
                video_embedding, attention_weights = self.video_embedder(features, action_embedding, mask=mask)
        else:
            video_embedding = self.video_embedder(features)
            attention_weights = None

        pos_modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in adverbs])
        positive = self.apply_modifiers(pos_modifiers, action_embedding)
        negative_act = self.apply_modifiers(pos_modifiers, neg_action_embedding)

        neg_modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in neg_adverbs])
        negative_adv = self.apply_modifiers(neg_modifiers, action_embedding)

        loss_triplet_act = F.triplet_margin_loss(video_embedding, positive, negative_act, margin=self.margin)
        if threshold_adverbs is not None:
            if threshold_adverbs.sum() == 0:
                loss_triplet_adv = torch.tensor(0)
                if threshold_adverbs.is_cuda:
                    loss_triplet_adv = loss_triplet_adv.cuda()
            else:
                loss_triplet_adv_all = F.triplet_margin_loss(video_embedding, positive, negative_adv, margin=self.margin, reduce=False)
                loss_triplet_adv_all[~threshold_adverbs] = 0
                loss_triplet_adv = loss_triplet_adv_all.sum()/threshold_adverbs.sum()
        else:
            loss_triplet_adv = F.triplet_margin_loss(video_embedding, positive, negative_adv, margin=self.margin)
        loss = [loss_triplet_act, loss_triplet_adv]
        return loss, None, attention_weights, video_embedding

    def val_forward(self, x, attention=None):
        features = x[0]
        actions = x[2]
        batch_size = features.shape[0]

        if self.transformer:
            pad=x[3]
            temporal_dim = features.shape[1]
            mask = torch.arange(temporal_dim).expand(len(pad), temporal_dim).cuda() < temporal_dim - pad.unsqueeze(1)
            action_gt_embedding = self.action_embedder(actions)
            if attention is not None:
                self.video_embedder.run_attention(features, attention)
            else:
                video_embedding, attention_weights = self.video_embedder(features, action_gt_embedding, mask=mask)
        else:
            video_embedding = self.video_embedder(features)
            attention_weights = None
        action_embedding = self.action_embedder(self.val_actions)
        modifiers = torch.stack([self.action_modifiers[adv.item()] for adv in self.val_adverbs])
        action_adverb_embeddings = self.apply_modifiers(modifiers, action_embedding)

        scores = {}
        for i, (adverb, action) in enumerate(self.dset.pairs):
            pair_embedding = action_adverb_embeddings[i, None].expand(batch_size,
                                                                      action_adverb_embeddings.size(1))
            score = self.compare_metric(video_embedding, pair_embedding)
            scores[(adverb, action)] = score
        return None, scores, attention_weights, video_embedding

    def forward(self, x, threshold_adverbs=None, attention=None):
        if self.training:
            loss, pred, att, vid_feats = self.train_forward(x, threshold_adverbs=threshold_adverbs, attention=None)
        else:
            with torch.no_grad():
                loss, pred, att, vid_feats = self.val_forward(x)
        return loss, pred, att, vid_feats

class StackedAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, emb_dim, heads=1, num_layers=3, dropout=0.0):
        super(StackedAttention, self).__init__()
        
        self.layers = nn.ModuleList([
            SDPAttention(d_model, d_k, d_v, emb_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, queries, mask=None):
        output = queries
        attention_scores = []
        
        for attn_layer, norm in zip(self.layers, self.layer_norms):
            # Apply attention
            attn_out, scores = attn_layer(features, output, mask)
            attention_scores.append(scores)
            
            # Residual connection + dropout + layer norm
            output = norm(output + self.dropout(attn_out))
        
        return output, attention_scores


class ActionAdverbClassifier(nn.Module):
    def __init__(self, dset, args):
        super(ActionAdverbClassifier, self).__init__()
        self.num_heads = 4
        self.dset = dset
        self.num_layers = args.num_layers
        # Feature encoder
        if args.temporal_agg == 'sdp':
            if self.num_layers > 1:
                self.feature_encoder = StackedAttention(dset.feature_dim, args.emb_dim, args.emb_dim,
                                              args.emb_dim, heads=self.num_heads, num_layers=self.num_layers)
            else:
                self.feature_encoder = SDPAttention(dset.feature_dim, args.emb_dim, args.emb_dim,
                                              args.emb_dim, heads=self.num_heads)
            self.transformer = True
        else:
            self.feature_encoder = MLP(dset.feature_dim, args.emb_dim, num_layers=args.num_layers)
            self.transformer = False

        # Classification head
        self.classifier = nn.Linear(args.emb_dim, dset.num_classes)

        # For attention in SDP mode
        if hasattr(dset, 'action_embedder'):
            self.action_embedder = dset.action_embedder
        else:
            # Dummy action embedder for attention mechanism
            self.action_embedder = nn.Embedding(len(dset.actions), args.emb_dim)
            if args.glove_init:
                pretrained_weight = load_word_embeddings(args.glove_path, dset.actions)
                self.action_embedder.weight.data.copy_(pretrained_weight)

    def forward(self, x):
        features = x[0]
        batch_size = features.shape[0]

        if self.transformer:
            pad = x[2]
            temporal_dim = features.shape[1]
            mask = torch.arange(temporal_dim).expand(len(pad), temporal_dim).cuda() < temporal_dim - pad.unsqueeze(1)

            # Use a dummy action embedding for attention (all zeros)
            dummy_action = torch.zeros(batch_size, dtype=torch.long).cuda()
            action_embedding = self.action_embedder(dummy_action)

            video_embedding, attention_weights = self.feature_encoder(features, action_embedding, mask=mask)
        else:
            video_embedding = self.feature_encoder(features)
            attention_weights = None

        # Classification
        logits = self.classifier(video_embedding)

        return logits, attention_weights

class ClassificationEvaluator:
    def __init__(self, dset):
        self.dset = dset
        self.num_classes = dset.num_classes

        # Create reverse mappings for actions and adverbs
        self.action2idx = dset.action2idx
        self.adverb2idx = dset.adverb2idx
        self.idx2action = dset.idx2action
        self.idx2adverb = dset.idx2adverb

    def _get_action_adverb_from_class(self, class_idx):
        """Extract action and adverb from class index."""
        pair = self.dset.idx2class[class_idx]
        action, adverb = pair
        return self.action2idx[action], self.adverb2idx[adverb]

    def calculate_accuracy(self, logits, labels):
        """Calculate top-1 accuracy."""
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float()
        accuracy = correct.mean()
        return accuracy

    def calculate_top_k_accuracy(self, logits, labels, k=5):
        """Calculate top-k accuracy for joint prediction."""
        _, top_k_pred = torch.topk(logits, k, dim=1)
        labels_expanded = labels.unsqueeze(1).expand_as(top_k_pred)
        correct = (top_k_pred == labels_expanded).any(dim=1).float()
        accuracy = correct.mean()
        return accuracy

    def calculate_top_k_action_accuracy(self, logits, labels, k=5):
        """
        Calculate top-k accuracy for actions only.
        A prediction is correct if the true action appears in the top-k predictions,
        regardless of the adverb.
        """
        # Get top-k predictions
        _, top_k_pred_indices = torch.topk(logits, k, dim=1)

        # Extract ground truth action
        gt_actions = []
        for label_idx in labels.cpu().numpy():
            gt_action_idx, _ = self._get_action_adverb_from_class(label_idx)
            gt_actions.append(gt_action_idx)
        gt_actions = torch.tensor(gt_actions)

        # Extract predicted actions from top-k
        correct = torch.zeros(labels.size(0), dtype=torch.bool)
        for i in range(labels.size(0)):
            predicted_actions = []
            for pred_idx in top_k_pred_indices[i].cpu().numpy():
                pred_action_idx, _ = self._get_action_adverb_from_class(pred_idx)
                predicted_actions.append(pred_action_idx)

            # Check if ground truth action is in top-k predicted actions
            if gt_actions[i].item() in predicted_actions:
                correct[i] = True

        accuracy = correct.float().mean().item()
        return accuracy

    def calculate_top_k_adverb_accuracy(self, logits, labels, k=5):
        """
        Calculate top-k accuracy for adverbs only.
        A prediction is correct if the true adverb appears in the top-k predictions,
        regardless of the action.
        """
        # Get top-k predictions
        _, top_k_pred_indices = torch.topk(logits, k, dim=1)

        # Extract ground truth adverb
        gt_adverbs = []
        for label_idx in labels.cpu().numpy():
            _, gt_adverb_idx = self._get_action_adverb_from_class(label_idx)
            gt_adverbs.append(gt_adverb_idx)
        gt_adverbs = torch.tensor(gt_adverbs)

        # Extract predicted adverbs from top-k
        correct = torch.zeros(labels.size(0), dtype=torch.bool)
        for i in range(labels.size(0)):
            predicted_adverbs = []
            for pred_idx in top_k_pred_indices[i].cpu().numpy():
                _, pred_adverb_idx = self._get_action_adverb_from_class(pred_idx)
                predicted_adverbs.append(pred_adverb_idx)

            # Check if ground truth adverb is in top-k predicted adverbs
            if gt_adverbs[i].item() in predicted_adverbs:
                correct[i] = True

        accuracy = correct.float().mean().item()
        return accuracy

    def calculate_top_k_joint_accuracy(self, logits, labels, k=5):
        """
        Calculate top-k accuracy for joint action-adverb prediction.
        A prediction is correct if BOTH the action AND adverb from the same
        prediction appear in the top-k, considering them as pairs.
        """
        # Get top-k predictions
        _, top_k_pred_indices = torch.topk(logits, k, dim=1)

        # Extract ground truth action and adverb
        gt_pairs = []
        for label_idx in labels.cpu().numpy():
            gt_action_idx, gt_adverb_idx = self._get_action_adverb_from_class(label_idx)
            gt_pairs.append((gt_action_idx, gt_adverb_idx))

        # Check if ground truth pair is in top-k
        correct = torch.zeros(labels.size(0), dtype=torch.bool)
        for i in range(labels.size(0)):
            predicted_pairs = []
            for pred_idx in top_k_pred_indices[i].cpu().numpy():
                pred_action_idx, pred_adverb_idx = self._get_action_adverb_from_class(pred_idx)
                predicted_pairs.append((pred_action_idx, pred_adverb_idx))

            # Check if ground truth pair is in top-k predicted pairs
            if gt_pairs[i] in predicted_pairs:
                correct[i] = True

        accuracy = correct.float().mean().item()
        return accuracy

    def get_per_class_accuracy(self, logits, labels):
        """Calculate per-class accuracy."""
        predictions = torch.argmax(logits, dim=1)
        per_class_acc = {}

        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                class_correct = (predictions[class_mask] == class_idx).float().mean()
                pair = self.dset.idx2class[class_idx]
                per_class_acc[pair] = class_correct.item()

        return per_class_acc

    def calculate_joint_metrics(self, logits, labels):
        """
        Calculate comprehensive joint action-adverb metrics.

        Returns a dictionary with:
        - joint_accuracy: Both action and adverb correct
        - action_accuracy: Action correct (regardless of adverb)
        - adverb_accuracy: Adverb correct (regardless of action)
        - action_precision: Precision for action predictions
        - action_recall: Recall for action predictions
        - action_f1: Macro F1 for actions
        - adverb_precision: Precision for adverb predictions
        - adverb_recall: Recall for adverb predictions
        - adverb_f1: Macro F1 for adverbs
        - per_action_accuracy: Dict of per-action joint accuracy
        - per_adverb_accuracy: Dict of per-adverb joint accuracy
        """
        predictions = torch.argmax(logits, dim=1)

        # Extract ground truth actions and adverbs
        gt_actions = []
        gt_adverbs = []
        pred_actions = []
        pred_adverbs = []

        for label_idx, pred_idx in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            gt_action_idx, gt_adverb_idx = self._get_action_adverb_from_class(label_idx)
            pred_action_idx, pred_adverb_idx = self._get_action_adverb_from_class(pred_idx)

            gt_actions.append(gt_action_idx)
            gt_adverbs.append(gt_adverb_idx)
            pred_actions.append(pred_action_idx)
            pred_adverbs.append(pred_adverb_idx)

        gt_actions = torch.tensor(gt_actions)
        gt_adverbs = torch.tensor(gt_adverbs)
        pred_actions = torch.tensor(pred_actions)
        pred_adverbs = torch.tensor(pred_adverbs)

        # Joint accuracy: both action AND adverb correct
        joint_correct = ((pred_actions == gt_actions) & (pred_adverbs == gt_adverbs)).float()
        joint_accuracy = joint_correct.mean().item()

        # Action-only accuracy
        action_correct = (pred_actions == gt_actions).float()
        action_accuracy = action_correct.mean().item()

        # Adverb-only accuracy
        adverb_correct = (pred_adverbs == gt_adverbs).float()
        adverb_accuracy = adverb_correct.mean().item()

        # Per-action joint accuracy
        per_action_acc = {}
        for action_name, action_idx in self.action2idx.items():
            action_mask = (gt_actions == action_idx)
            if action_mask.sum() > 0:
                acc = joint_correct[action_mask].mean().item()
                per_action_acc[action_name] = acc

        # Per-adverb joint accuracy
        per_adverb_acc = {}
        for adverb_name, adverb_idx in self.adverb2idx.items():
            adverb_mask = (gt_adverbs == adverb_idx)
            if adverb_mask.sum() > 0:
                acc = joint_correct[adverb_mask].mean().item()
                per_adverb_acc[adverb_name] = acc

        # Calculate precision, recall, F1 for actions
        action_metrics = self._calculate_precision_recall_f1(
            gt_actions.numpy(),
            pred_actions.numpy(),
            len(self.action2idx)
        )

        # Calculate precision, recall, F1 for adverbs
        adverb_metrics = self._calculate_precision_recall_f1(
            gt_adverbs.numpy(),
            pred_adverbs.numpy(),
            len(self.adverb2idx)
        )

        metrics = {
            'joint_accuracy': joint_accuracy,
            'action_accuracy': action_accuracy,
            'adverb_accuracy': adverb_accuracy,
            'action_precision': action_metrics['precision'],
            'action_recall': action_metrics['recall'],
            'action_f1': action_metrics['f1'],
            'adverb_precision': adverb_metrics['precision'],
            'adverb_recall': adverb_metrics['recall'],
            'adverb_f1': adverb_metrics['f1'],
            'per_action_accuracy': per_action_acc,
            'per_adverb_accuracy': per_adverb_acc,
        }

        return metrics

    def _calculate_precision_recall_f1(self, y_true, y_pred, num_classes):
        """Calculate macro precision, recall, and F1 score."""
        import numpy as np

        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for class_idx in range(num_classes):
            # True positives
            tp = ((y_pred == class_idx) & (y_true == class_idx)).sum()
            # False positives
            fp = ((y_pred == class_idx) & (y_true != class_idx)).sum()
            # False negatives
            fn = ((y_pred != class_idx) & (y_true == class_idx)).sum()

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_per_class.append(precision)

            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class.append(recall)

            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_per_class.append(f1)

        return {
            'precision': np.mean(precision_per_class),
            'recall': np.mean(recall_per_class),
            'f1': np.mean(f1_per_class),
        }

    def get_confusion_matrices(self, logits, labels):
        """
        Generate confusion matrices for actions and adverbs separately.

        Returns:
        - action_confusion: Confusion matrix for actions
        - adverb_confusion: Confusion matrix for adverbs
        """
        import numpy as np

        predictions = torch.argmax(logits, dim=1)

        # Extract ground truth and predicted actions/adverbs
        gt_actions = []
        gt_adverbs = []
        pred_actions = []
        pred_adverbs = []

        for label_idx, pred_idx in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            gt_action_idx, gt_adverb_idx = self._get_action_adverb_from_class(label_idx)
            pred_action_idx, pred_adverb_idx = self._get_action_adverb_from_class(pred_idx)

            gt_actions.append(gt_action_idx)
            gt_adverbs.append(gt_adverb_idx)
            pred_actions.append(pred_action_idx)
            pred_adverbs.append(pred_adverb_idx)

        # Create confusion matrices
        num_actions = len(self.action2idx)
        num_adverbs = len(self.adverb2idx)

        action_confusion = np.zeros((num_actions, num_actions), dtype=np.int32)
        adverb_confusion = np.zeros((num_adverbs, num_adverbs), dtype=np.int32)

        for gt_act, pred_act in zip(gt_actions, pred_actions):
            action_confusion[gt_act, pred_act] += 1

        for gt_adv, pred_adv in zip(gt_adverbs, pred_adverbs):
            adverb_confusion[gt_adv, pred_adv] += 1

        return {
            'action_confusion': action_confusion,
            'adverb_confusion': adverb_confusion,
        }

class Evaluator:
    def __init__(self, dset, model):
        self.dset = dset
        pairs = [(dset.adverb2idx[adv.strip()], dset.action2idx[act]) for adv, act in dset.pairs]
        self.pairs = torch.LongTensor(pairs).cuda()

        ## mask over pairs for ground-truth action given in testing
        action_gt_mask = []
        for _act in dset.actions:
            mask = [1 if _act==act else 0 for adv, act in dset.pairs]
            action_gt_mask.append(torch.BoolTensor(mask))
        self.action_gt_mask = torch.stack(action_gt_mask, 0).cuda()

        antonym_mask = []
        for _adv in dset.adverbs:
            mask = [1 if (_adv==adv or _adv==dset.antonyms[adv]) else 0 for adv, act in dset.pairs]
            antonym_mask.append(torch.BoolTensor(mask))
        self.antonym_mask = torch.stack(antonym_mask, 0).cuda()

    def get_gt_action_scores(self, scores, action_gt):
        mask = self.action_gt_mask[action_gt]
        action_gt_scores = scores.clone()
        action_gt_scores[~mask] = -1e10
        return action_gt_scores

    def get_antonym_scores(self, scores, adverb_gt):
        mask = self.antonym_mask[adverb_gt]
        antonym_scores = scores.clone()
        antonym_scores[~mask] = -1e10
        return antonym_scores

    def get_gt_action_antonym_scores(self, scores, action_gt, adverb_gt):
        mask = self.antonym_mask[adverb_gt] & self.action_gt_mask[action_gt]
        action_gt_antonym_scores = scores.clone()
        action_gt_antonym_scores[~mask] = -1e10
        return action_gt_antonym_scores

    def get_scores(self, scores, action_gt, adverb_gt):
        scores = torch.stack([scores[(adv, act)] for adv, act in self.dset.pairs], 1)
        action_gt_scores = self.get_gt_action_scores(scores, action_gt)
        antonym_action_gt_scores = self.get_gt_action_antonym_scores(scores, action_gt, adverb_gt)
        return scores, action_gt_scores, antonym_action_gt_scores

    
