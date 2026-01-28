import os
import shutil
import torch
import numpy as np

def save_args(args):
    shutil.copy('train.py', args.checkpoint_dir)
    #shutil.copy('models.py', args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

def introduce_adverbs(optimizer, lr):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'action_modifiers':
            param_group['lr'] = lr * 0.1 
        else:
            param_group['lr'] = lr * 0.1

def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(checkpoint_dir, 'ckpt_E_%d'%(epoch)))

def calculate_p1(dset, scores, adverb_gt):
    """Top-1 adverb accuracy (extracts adverb from best scoring pair)."""
    pair_pred = np.argmax(scores.numpy(), axis=1)
    adverb_pred = [dset.adverb2idx[dset.pairs[pred][0]] for pred in pair_pred]
    acc = (adverb_pred == adverb_gt.cpu().numpy()).mean()
    return acc

def calculate_p5(dset, scores, adverb_gt):
    """Top-5 adverb accuracy (checks if correct adverb is in top 5 pairs)."""
    top5_pred = np.argsort(scores.numpy(), axis=1)[:, -5:]  # Top 5 pair indices
    adverb_gt_np = adverb_gt.cpu().numpy()
    correct = 0
    for i, top5_pairs in enumerate(top5_pred):
        top5_adverbs = [dset.adverb2idx[dset.pairs[pred][0]] for pred in top5_pairs]
        if adverb_gt_np[i] in top5_adverbs:
            correct += 1
    return correct / len(adverb_gt_np)

def calculate_mean_p1(dset, scores, adverb_gt):
    """Balanced (macro-averaged) adverb accuracy across classes."""
    pair_pred = np.argmax(scores.numpy(), axis=1)
    adverb_pred = [dset.adverb2idx[dset.pairs[pred][0]] for pred in pair_pred]
    accs = (adverb_pred == adverb_gt.cpu().numpy())
    adverb_gt_cpu = adverb_gt.cpu().numpy()
    per_class = [[accs[i] for i in range(scores.shape[0]) if adverb_gt_cpu[i] == adv] for adv in dset.adverb2idx.values()]
    per_class_accs = [sum(l)/float(len(l)) for l in per_class if len(l) > 0]
    acc = sum(per_class_accs)/len(per_class_accs)
    return acc

def calculate_p1_action(dset, scores, action_gt):
    """Top-1 action accuracy (extracts action from best scoring pair)."""
    pair_pred = np.argmax(scores.numpy(), axis=1)
    action_pred = [dset.action2idx[dset.pairs[pred][1]] for pred in pair_pred]
    acc = (action_pred == action_gt.cpu().numpy()).mean()
    return acc

def calculate_p5_action(dset, scores, action_gt):
    """Top-5 action accuracy (checks if correct action is in top 5 pairs)."""
    top5_pred = np.argsort(scores.numpy(), axis=1)[:, -5:]
    action_gt_np = action_gt.cpu().numpy()
    correct = 0
    for i, top5_pairs in enumerate(top5_pred):
        top5_actions = [dset.action2idx[dset.pairs[pred][1]] for pred in top5_pairs]
        if action_gt_np[i] in top5_actions:
            correct += 1
    return correct / len(action_gt_np)

def calculate_p1_pair(dset, scores, adverb_gt, action_gt):
    """Top-1 pair accuracy (both action AND adverb must be correct)."""
    pair_pred = np.argmax(scores.numpy(), axis=1)
    adverb_pred = np.array([dset.adverb2idx[dset.pairs[pred][0]] for pred in pair_pred])
    action_pred = np.array([dset.action2idx[dset.pairs[pred][1]] for pred in pair_pred])

    # Both adverb and action must be correct
    adverb_correct = (adverb_pred == adverb_gt.cpu().numpy())
    action_correct = (action_pred == action_gt.cpu().numpy())
    acc = (adverb_correct & action_correct).mean()
    return acc

def calculate_p5_pair(dset, scores, adverb_gt, action_gt):
    """Top-5 pair accuracy (correct pair must be in top 5)."""
    top5_pred = np.argsort(scores.numpy(), axis=1)[:, -5:]
    adverb_gt_np = adverb_gt.cpu().numpy()
    action_gt_np = action_gt.cpu().numpy()
    correct = 0
    for i, top5_pairs in enumerate(top5_pred):
        for pred_idx in top5_pairs:
            pred_adverb = dset.adverb2idx[dset.pairs[pred_idx][0]]
            pred_action = dset.action2idx[dset.pairs[pred_idx][1]]
            if pred_adverb == adverb_gt_np[i] and pred_action == action_gt_np[i]:
                correct += 1
                break
    return correct / len(adverb_gt_np)


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
