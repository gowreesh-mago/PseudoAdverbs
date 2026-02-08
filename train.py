import os
import shutil
import tqdm
import numpy as np
import torch
import torch.optim as optim
import time
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_args, introduce_adverbs, save_checkpoint, calculate_p1, calculate_mean_p1, AverageMeter

from opts import parser
from dataset import AdverbDataset
from model import ActionModifiers, Evaluator, ActionAdverbClassifier, ClassificationEvaluator
from wandb_config import init_wandb_with_config
import wandb

from torch.utils.tensorboard import SummaryWriter


def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_args(args)
    
    # Initialize wandb with config file support
    init_wandb_with_config(
        project_name="pseudo-adverbs",
        run_name=args.wandb_run_name,
        model_config=vars(args),
        checkpoint_dir=args.checkpoint_dir,
        config_path=getattr(args, 'wandb_config', None),
        disable_wandb=args.no_wandb
    )
    


    train_set = AdverbDataset(args.data_dir, args.train_feature_dir, agg=args.temporal_agg,
                              modality=args.modality, window_size=args.t_train,
                              adverb_filter=args.adverb_filter, phase='train',
                              load_in_memory=args.load_in_memory,
                              unlabelled_ratio=args.unlabelled_ratio,
                              unlabelled_feature_dir=args.unlabelled_feature_dir,
                              classification_mode=args.classification_mode,
                              class_mode=args.class_mode)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers)
    test_set = AdverbDataset(args.data_dir, args.test_feature_dir, agg=args.temporal_agg,
                             modality=args.modality, window_size=args.t_test,
                             adverb_filter=args.adverb_filter, phase='test',
                             load_in_memory=args.load_in_memory,
                             unlabelled_ratio=args.unlabelled_ratio,
                             unlabelled_feature_dir=args.unlabelled_feature_dir,
                             classification_mode=args.classification_mode,
                             class_mode=args.class_mode)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    if args.classification_mode:
        model = ActionAdverbClassifier(train_set, args).cuda()
        evaluator = ClassificationEvaluator(train_set)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        model = ActionModifiers(train_set, args).cuda()
        evaluator = Evaluator(train_set, model)
        adverb_thresholds = None
        if args.adaptive_threshold:
            adverb_thresholds = torch.Tensor([args.pseudo_label_threshold]*len(model.dset.adverb2idx.keys())).cuda()

    if args.classification_mode:
        # Simple optimizer for classification mode
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        # Original optimizer setup for metric learning mode
        modifier_params = [param for name, param in model.named_parameters()
                           if ('action_modifiers' in name) and param.requires_grad]
        other_params = [param for name, param in model.named_parameters()
                        if ('action_modifiers' not in name) and param.requires_grad]
        if not args.pretrain_action:
            optim_params = [{'name': 'action_modifiers', 'params': modifier_params},
                            {'name': 'embedding', 'params': other_params}]
        else:
            optim_params = [{'name': 'action_modifiers', 'params': modifier_params, 'lr':0},
                            {'name': 'embedding', 'params': other_params}]
        optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        pretrained_state_dict = checkpoint['net']
        model_state_dict = model.state_dict()
        pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        start_epoch = checkpoint['epoch']

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'log'))

    pseudo_start_epoch = args.pseudo_start_epoch
    pseudo_weight = args.pseudo_weight
    pseudo_always_action = args.pseudo_action_pretraining

    if args.classification_mode:
        # Classification training loop
        test_classification(model, test_loader, evaluator, criterion, writer, start_epoch, args)
        for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
            train_classification(model, train_loader, optimizer, criterion, writer, epoch, args)
            if epoch % args.eval_interval == 0:
                test_classification(model, test_loader, evaluator, criterion, writer, epoch, args)
            if epoch % args.save_interval == 0 and epoch > 0:
                save_checkpoint(model, epoch, args.checkpoint_dir)
        
    else:
        # Original metric learning training loop
        if args.pretrain_action:
            pseudo_weight = 0.0
        else:
            pseudo_weight = 0.0
        test(model, test_loader, evaluator, writer, start_epoch, args)
        for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
            if args.pretrain_action and epoch == args.adverb_start:
                introduce_adverbs(optimizer, args.lr)
            adverb_thresholds = train(model, train_loader, optimizer, writer, epoch, args.unlabelled_ratio, pseudo_weight, pseudo_always_action, args.num_pseudo_labelled, args.pseudo_selection, adverb_thresholds, args)
            if epoch % args.eval_interval == 0:
                test(model, test_loader, evaluator, writer, epoch, args)
            if epoch % args.save_interval == 0 and epoch > 0:
                save_checkpoint(model, epoch, args.checkpoint_dir)
            if epoch >= pseudo_start_epoch:
                pseudo_weight = args.pseudo_weight
    writer.close()
    if not args.no_wandb:
        wandb.finish()

def pseudo_label_adverbs(model, features, actions, pad, num, method, threshold=0, adverb_thresholds=None):
    threshold_mask = torch.ones((num, actions.shape[0]), dtype=torch.bool)
    dummy_adverbs = torch.zeros(actions.shape).cuda()
    model.eval()
    data = [features, dummy_adverbs, actions, pad]
    predictions, attention = model(data)[1:3]
    combined_labels = np.array(list(predictions.keys()))
    predictions_tensor = torch.stack(list(predictions.values()))
    action_gt_mask = []
    actions_np = actions.cpu().numpy()
    for _adv, _act in model.dset.pairs:
        mask = model.dset.action2idx[_act]==actions_np
        action_gt_mask.append(torch.BoolTensor(mask))
    action_gt_mask = torch.stack(action_gt_mask, 0)
    predictions_tensor[~action_gt_mask] = -1e10 #Not masking max diff at start as gives 0
    if args.unseen_mask:
        unseen_mask = torch.zeros(predictions_tensor.shape[0], dtype=torch.bool)
        for i, (_adv, _act) in enumerate(model.dset.pairs):
            if (_act, _adv) in model.dset.unlabelled_pairs:
                unseen_mask[i] = True
        predictions_tensor[~unseen_mask, :] = -1e10
    reshaped_pred = predictions_tensor.reshape((int(len(model.dset.adverbs)/2), 2, len(model.dset.actions), -1))
    ant_pred = reshaped_pred[:, torch.LongTensor([1,0])]
    if method == 'closest':
        vals, predictions_ind = torch.topk(predictions_tensor, num, dim=0)
    elif method == 'diff':
        vals, predictions_ind = torch.topk((reshaped_pred-ant_pred).reshape(predictions_tensor.shape), num, dim=0)

    highest_pred = combined_labels[predictions_ind.cpu().numpy()]
    highest_pred = highest_pred[:,:,0]
    pseudo_adverbs = torch.tensor(np.vectorize(model.dset.adverb2idx.get)(highest_pred))

    conf = torch.zeros(2, 1, highest_pred.shape[-1])
    if threshold > 0:
        adv_vals = predictions_tensor.gather(0, predictions_ind)
        ant_vals = ant_pred.reshape(predictions_tensor.shape).gather(0,predictions_ind)
        scores = torch.stack([adv_vals, ant_vals])
        if args.conf_type == 'softmax':
            conf = torch.nn.functional.softmax(scores, 0)
        elif args.conf_type == 'margin':
            conf = scores - scores[torch.LongTensor([1,0])]
        if args.adaptive_threshold:
            adaptive_thres = adverb_thresholds[pseudo_adverbs]
            threshold_mask = conf[0] > adaptive_thres
        else:
            threshold_mask = conf[0] > threshold

    ants = np.vectorize(model.dset.antonyms.get)(highest_pred)
    unlabelled_neg_adverbs = torch.tensor(np.vectorize(model.dset.adverb2idx.get)(ants))
    model.train()
    return pseudo_adverbs, unlabelled_neg_adverbs, threshold_mask, attention, conf[0]


def train(model, train_loader, optimizer, writer, epoch, unlabelled_ratio, pseudo_loss_factor, pseudo_always_action, num_pseudo, pseudo_selection, adverb_thresholds, args):
    model.train()
    train_loss = 0.0
    act_loss = 0.0
    adv_loss = 0.0
    pseudo_train_loss = 0.0
    pseudo_act_loss = 0.0
    pseudo_adv_loss = 0.0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()
    all_pseudo_labels = torch.Tensor().long()
    unmasked_pseudo_labels = torch.Tensor().long()
    all_train_labels = torch.Tensor()
    all_actions = torch.Tensor().long().cuda()
    conf_sums = torch.Tensor([0]*len(model.dset.adverb2idx.keys())).cuda()
    for idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.unlabelled_ratio > 0:
            unlabelled_features = data[-4].cuda()
            unlabelled_actions = data[-3].cuda()
            unlabelled_pad = data[-2].cuda()
            unlabelled_neg_actions = data[-1].cuda()
            unlabelled_features = unlabelled_features.reshape(-1, *unlabelled_features.shape[2:])
            unlabelled_actions = unlabelled_actions.reshape(-1)
            unlabelled_neg_actions = unlabelled_neg_actions.reshape(-1)
            unlabelled_pad = unlabelled_pad.reshape(-1)
            data = [d.cuda() for d in data[:-4]]
        else:
            data = [d.cuda() for d in data]
        data_time.update(time.time() - start)
        all_loss = model(data)[0]
        loss = sum(all_loss)
        pseudo_loss = 0
        iter_pseudo_act_loss = 0
        if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action):
            pseudo_labelled_adverbs, unlabelled_neg_adverbs, threshold_mask, attention, adverb_scores = pseudo_label_adverbs(model, unlabelled_features, unlabelled_actions, unlabelled_pad, num_pseudo, pseudo_selection, args.pseudo_label_threshold, adverb_thresholds=adverb_thresholds)
            if args.adaptive_threshold:
                for j in range(adverb_scores.shape[1]):
                    conf_sums[pseudo_labelled_adverbs[:,j]] += adverb_scores[:,j]
            unmasked_pseudo_labels = torch.cat([unmasked_pseudo_labels, pseudo_labelled_adverbs[threshold_mask].reshape(-1)])
            all_pseudo_labels = torch.cat([all_pseudo_labels, pseudo_labelled_adverbs.reshape(-1)])
            all_actions = torch.cat([all_actions, unlabelled_actions.repeat_interleave(num_pseudo)])
            for i in range(0, pseudo_labelled_adverbs.shape[0]):
                unlabelled_data = [unlabelled_features, pseudo_labelled_adverbs[i].cuda(), unlabelled_actions, unlabelled_pad, unlabelled_neg_adverbs[i].cuda(), unlabelled_neg_actions]
                if args.pseudo_label_threshold > 0:
                    all_pseudo_loss = model(unlabelled_data, threshold_adverbs=threshold_mask[i])[0]
                else:
                    all_pseudo_loss = model(unlabelled_data)[0]
                pseudo_loss += sum(all_pseudo_loss)
                iter_pseudo_act_loss += all_pseudo_loss[0].item()
                pseudo_adv_loss += all_pseudo_loss[1].item()
            pseudo_train_loss += pseudo_loss.item()
            pseudo_act_loss += iter_pseudo_act_loss
            if pseudo_loss_factor > 0:
                total_loss = loss + (pseudo_loss/pseudo_labelled_adverbs.shape[0]) * pseudo_loss_factor
            else:
                total_loss = loss + (iter_pseudo_act_loss/pseudo_labelled_adverbs.shape[0])
        else:
            total_loss = loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item()
        act_loss += all_loss[0].item()
        adv_loss += all_loss[1].item() ##should be introduced after action mods have started training

        # Log step-level metrics to wandb
        if not args.no_wandb:
            step_wandb_log = {
                'step/loss_total': loss.item(),
                'step/loss_action': all_loss[0].item(),
                'step/loss_adverb': all_loss[1].item(),
            }
            if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action) and pseudo_loss > 0:
                step_wandb_log['step/loss_pseudo'] = pseudo_loss.item() / pseudo_labelled_adverbs.shape[0]
            wandb.log(step_wandb_log)

        batch_time.update(time.time() - start - data_time.val)
        start = time.time()
        if epoch == 0:
            all_train_labels = torch.cat([all_train_labels, data[1].cpu()])

    ## Updated adverb thresholds
    if args.adaptive_threshold and unlabelled_ratio > 0 and pseudo_loss_factor > 0:
        class_counts = torch.bincount(all_pseudo_labels)
        av_count = class_counts.sum()/class_counts.shape[0]
        adverb_thresholds = ((conf_sums/av_count)**args.smoothing)*args.pseudo_label_threshold

    train_loss /= len(train_loader)
    act_loss /= len(train_loader)
    adv_loss /= len(train_loader)
    if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action):
        pseudo_train_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        pseudo_act_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        pseudo_adv_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        if unmasked_pseudo_labels.shape[0] > 0:
            writer.add_histogram('PseudoLabelDist', unmasked_pseudo_labels, epoch)
        writer.add_scalar('PseudoLabel/AboveThreshold', threshold_mask.sum(), epoch)
        if args.adaptive_threshold:
            for i in range(adverb_thresholds.shape[0]):
                writer.add_scalar('PseduoLabelThresholds/' + str(i), adverb_thresholds[i], epoch)
    writer.add_scalar('Loss/Train/Total', train_loss, epoch)
    writer.add_scalar('Loss/Train/Action', act_loss, epoch)
    writer.add_scalar('Loss/Train/Adverb', adv_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoTotal', pseudo_train_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoAction', pseudo_act_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoAdverb', pseudo_adv_loss, epoch)
    
    # Log to wandb
    if not args.no_wandb:
        wandb_log = {
            'epoch': epoch,
            'train/loss_total': train_loss,
            'train/loss_action': act_loss,
            'train/loss_adverb': adv_loss,
            'train/loss_pseudo_total': pseudo_train_loss,
            'train/loss_pseudo_action': pseudo_act_loss,
            'train/loss_pseudo_adverb': pseudo_adv_loss,
            'train/batch_time': batch_time.avg,
            'train/data_time': data_time.avg,
            'manifold_type': args.manifold
        }
        if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action):
            wandb_log['train/pseudo_above_threshold'] = threshold_mask.sum().item()
        wandb.log(wandb_log)

    if epoch == 0:
        writer.add_histogram('TrainLabelDist', all_train_labels, epoch)
    print('E: %d | L: %.2E | L_act: %.2E | L_adv: %.2E | Batch Time: %.2E | Data Time: %.2E'%(epoch, train_loss, act_loss, adv_loss, batch_time.avg, data_time.avg))
    gc.collect()
    return adverb_thresholds

def test(model, test_loader, evaluator, writer, epoch, args):
    model.eval()
    accuracies = []
    all_antonym_action_gt_scores = torch.Tensor().cuda()
    all_adverb_gt = torch.Tensor().cuda()
    for idx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        data = [d.cuda() for d in data]
        predictions = model(data)[1]
        adverb_gt, action_gt = data[1], data[2]
        scores, action_gt_scores, antonym_action_gt_scores = evaluator.get_scores(predictions, action_gt, adverb_gt)
        all_antonym_action_gt_scores = torch.cat([all_antonym_action_gt_scores, antonym_action_gt_scores])
        all_adverb_gt = torch.cat([all_adverb_gt, adverb_gt])
        acc = calculate_p1(model.dset, antonym_action_gt_scores.cpu(), adverb_gt.cpu())
        print('E %d | Video-to-Adverb Antonym P@1: %.3f'%(epoch, acc))
        accuracies.append(acc)
    acc_mean = calculate_mean_p1(model.dset, all_antonym_action_gt_scores.cpu(), all_adverb_gt.cpu())
    writer.add_scalar('Acc/Test/Video-to-Adverb Antonym', sum(accuracies)/len(accuracies), epoch)
    writer.add_scalar('Acc/Test/Video-to-Adverb Antonym Mean', acc_mean, epoch)
    
    # Log test metrics to wandb
    if not args.no_wandb:
        wandb.log({
            'test/video_to_adverb_antonym_acc': sum(accuracies)/len(accuracies),
            'test/video_to_adverb_antonym_mean': acc_mean
        })

def train_classification(model, train_loader, optimizer, criterion, writer, epoch, args):
    """Training function for classification mode."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    all_logits = []
    all_labels = []

    for idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        features = data[0].cuda()
        labels = data[1].cuda()

        if len(data) > 2:  # SDP mode with padding
            pad = data[2].cuda()
            model_input = [features, labels, pad]
        else:
            model_input = [features, labels]

        logits, _ = model(model_input)
        loss = criterion(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        total_correct = (predictions == labels).sum().item()
        total_samples = labels.size(0)
        accuracy = (predictions == labels).float().mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy.item()
        num_batches += 1

        # Store for epoch-level metrics
        all_logits.append(logits.detach())
        all_labels.append(labels)

        # Log step-level metrics to wandb
        if not args.no_wandb:
            step_wandb_log = {
                'step/classification_loss': loss.item(),
                'step/classification_accuracy': accuracy.item(),
            }
            wandb.log(step_wandb_log)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0

    # Calculate joint metrics for the entire epoch
    if len(all_logits) > 0:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Get evaluator from model's dataset
        from model import ClassificationEvaluator
        evaluator = ClassificationEvaluator(model.dset)
        joint_metrics = evaluator.calculate_joint_metrics(all_logits, all_labels)

        # Calculate top-5 metrics
        top5_action_acc = evaluator.calculate_top_k_action_accuracy(all_logits, all_labels, k=5)
        top5_adverb_acc = evaluator.calculate_top_k_adverb_accuracy(all_logits, all_labels, k=5)
        top5_joint_acc = evaluator.calculate_top_k_joint_accuracy(all_logits, all_labels, k=5)

        # Log to TensorBoard
        writer.add_scalar('Loss/Train/Classification', avg_loss, epoch)
        writer.add_scalar('Acc/Train/Classification', avg_acc, epoch)
        writer.add_scalar('Acc/Train/Joint_Accuracy', joint_metrics['joint_accuracy'], epoch)
        writer.add_scalar('Acc/Train/Action_Only_Accuracy', joint_metrics['action_accuracy'], epoch)
        writer.add_scalar('Acc/Train/Adverb_Only_Accuracy', joint_metrics['adverb_accuracy'], epoch)
        writer.add_scalar('Acc/Train/Top5_Action_Accuracy', top5_action_acc, epoch)
        writer.add_scalar('Acc/Train/Top5_Adverb_Accuracy', top5_adverb_acc, epoch)
        writer.add_scalar('Acc/Train/Top5_Joint_Accuracy', top5_joint_acc, epoch)
        writer.add_scalar('Metrics/Train/Action_F1', joint_metrics['action_f1'], epoch)
        writer.add_scalar('Metrics/Train/Adverb_F1', joint_metrics['adverb_f1'], epoch)

        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/classification_loss': avg_loss,
                'train/classification_accuracy': avg_acc,
                'train/joint_accuracy': joint_metrics['joint_accuracy'],
                'train/action_accuracy': joint_metrics['action_accuracy'],
                'train/adverb_accuracy': joint_metrics['adverb_accuracy'],
                'train/top5_action_accuracy': top5_action_acc,
                'train/top5_adverb_accuracy': top5_adverb_acc,
                'train/top5_joint_accuracy': top5_joint_acc,
                'train/action_f1': joint_metrics['action_f1'],
                'train/adverb_f1': joint_metrics['adverb_f1'],
            })

        print(f'E: {epoch} | Classification Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}')
        print(f'  Joint Acc: {joint_metrics["joint_accuracy"]:.4f} | Action Acc: {joint_metrics["action_accuracy"]:.4f} | Adverb Acc: {joint_metrics["adverb_accuracy"]:.4f}')
        print(f'  Top-5 Joint: {top5_joint_acc:.4f} | Top-5 Action: {top5_action_acc:.4f} | Top-5 Adverb: {top5_adverb_acc:.4f}')
    else:
        writer.add_scalar('Loss/Train/Classification', avg_loss, epoch)
        writer.add_scalar('Acc/Train/Classification', avg_acc, epoch)

        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/classification_loss': avg_loss,
                'train/classification_accuracy': avg_acc,
            })

        print(f'E: {epoch} | Classification Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}')

def test_classification(model, test_loader, evaluator, criterion, writer, epoch, args):
    """Testing function for classification mode."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            features = data[0].cuda()
            labels = data[1].cuda()


            if len(data) > 2:  # SDP mode with padding
                pad = data[2].cuda()
                model_input = [features, labels, pad]
            else:
                model_input = [features, labels]

            logits, _ = model(model_input)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(labels)

    if len(all_logits) == 0:
        print("No valid test samples found!")
        return

    # Concatenate all results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate basic metrics
    avg_loss = total_loss / len(test_loader)
    top1_acc = evaluator.calculate_accuracy(all_logits, all_labels)
    top5_acc = evaluator.calculate_top_k_accuracy(all_logits, all_labels, k=5)

    # Calculate top-5 metrics for action, adverb, and joint
    top5_action_acc = evaluator.calculate_top_k_action_accuracy(all_logits, all_labels, k=5)
    top5_adverb_acc = evaluator.calculate_top_k_adverb_accuracy(all_logits, all_labels, k=5)
    top5_joint_acc = evaluator.calculate_top_k_joint_accuracy(all_logits, all_labels, k=5)

    # Calculate joint action-adverb metrics
    joint_metrics = evaluator.calculate_joint_metrics(all_logits, all_labels)

    # Calculate confusion matrices
    confusion_matrices = evaluator.get_confusion_matrices(all_logits, all_labels)

    # Log to TensorBoard
    writer.add_scalar('Loss/Test/Classification', avg_loss, epoch)
    writer.add_scalar('Acc/Test/Classification_Top1', top1_acc, epoch)
    writer.add_scalar('Acc/Test/Classification_Top5', top5_acc, epoch)

    # Log top-5 metrics to TensorBoard
    writer.add_scalar('Acc/Test/Top5_Action_Accuracy', top5_action_acc, epoch)
    writer.add_scalar('Acc/Test/Top5_Adverb_Accuracy', top5_adverb_acc, epoch)
    writer.add_scalar('Acc/Test/Top5_Joint_Accuracy', top5_joint_acc, epoch)

    # Log joint metrics to TensorBoard
    writer.add_scalar('Acc/Test/Joint_Accuracy', joint_metrics['joint_accuracy'], epoch)
    writer.add_scalar('Acc/Test/Action_Only_Accuracy', joint_metrics['action_accuracy'], epoch)
    writer.add_scalar('Acc/Test/Adverb_Only_Accuracy', joint_metrics['adverb_accuracy'], epoch)
    writer.add_scalar('Metrics/Test/Action_Precision', joint_metrics['action_precision'], epoch)
    writer.add_scalar('Metrics/Test/Action_Recall', joint_metrics['action_recall'], epoch)
    writer.add_scalar('Metrics/Test/Action_F1', joint_metrics['action_f1'], epoch)
    writer.add_scalar('Metrics/Test/Adverb_Precision', joint_metrics['adverb_precision'], epoch)
    writer.add_scalar('Metrics/Test/Adverb_Recall', joint_metrics['adverb_recall'], epoch)
    writer.add_scalar('Metrics/Test/Adverb_F1', joint_metrics['adverb_f1'], epoch)

    # Log per-action accuracies
    for action_name, acc in joint_metrics['per_action_accuracy'].items():
        writer.add_scalar(f'Acc/Test/PerAction/{action_name}', acc, epoch)

    # Log per-adverb accuracies
    for adverb_name, acc in joint_metrics['per_adverb_accuracy'].items():
        writer.add_scalar(f'Acc/Test/PerAdverb/{adverb_name}', acc, epoch)

    # Log to wandb
    if not args.no_wandb:
        wandb_log = {
            'test/classification_loss': avg_loss,
            'test/classification_top1_accuracy': top1_acc,
            'test/classification_top5_accuracy': top5_acc,
            'test/top5_action_accuracy': top5_action_acc,
            'test/top5_adverb_accuracy': top5_adverb_acc,
            'test/top5_joint_accuracy': top5_joint_acc,
            'test/joint_accuracy': joint_metrics['joint_accuracy'],
            'test/action_accuracy': joint_metrics['action_accuracy'],
            'test/adverb_accuracy': joint_metrics['adverb_accuracy'],
            'test/action_precision': joint_metrics['action_precision'],
            'test/action_recall': joint_metrics['action_recall'],
            'test/action_f1': joint_metrics['action_f1'],
            'test/adverb_precision': joint_metrics['adverb_precision'],
            'test/adverb_recall': joint_metrics['adverb_recall'],
            'test/adverb_f1': joint_metrics['adverb_f1'],
        }

        # Add per-action accuracies to wandb
        for action_name, acc in joint_metrics['per_action_accuracy'].items():
            wandb_log[f'test/per_action_acc/{action_name}'] = acc

        # Add per-adverb accuracies to wandb
        for adverb_name, acc in joint_metrics['per_adverb_accuracy'].items():
            wandb_log[f'test/per_adverb_acc/{adverb_name}'] = acc

        # Log confusion matrices as images to wandb
        # Action confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(confusion_matrices['action_confusion'],
                    annot=False, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=list(evaluator.action2idx.keys()),
                    yticklabels=list(evaluator.action2idx.keys()))
        ax.set_xlabel('Predicted Action')
        ax.set_ylabel('True Action')
        ax.set_title(f'Action Confusion Matrix (Epoch {epoch})')
        plt.tight_layout()
        wandb_log['test/action_confusion_matrix'] = wandb.Image(fig)
        plt.close(fig)

        # Adverb confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(confusion_matrices['adverb_confusion'],
                    annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=list(evaluator.adverb2idx.keys()),
                    yticklabels=list(evaluator.adverb2idx.keys()))
        ax.set_xlabel('Predicted Adverb')
        ax.set_ylabel('True Adverb')
        ax.set_title(f'Adverb Confusion Matrix (Epoch {epoch})')
        plt.tight_layout()
        wandb_log['test/adverb_confusion_matrix'] = wandb.Image(fig)
        plt.close(fig)

        wandb.log(wandb_log)

    # Print summary
    print(f'E: {epoch} | Test Loss: {avg_loss:.4f} | Top-1 Acc: {top1_acc:.4f} | Top-5 Acc: {top5_acc:.4f}')
    print(f'  Joint Acc: {joint_metrics["joint_accuracy"]:.4f} | Action Acc: {joint_metrics["action_accuracy"]:.4f} | Adverb Acc: {joint_metrics["adverb_accuracy"]:.4f}')
    print(f'  Top-5 Joint: {top5_joint_acc:.4f} | Top-5 Action: {top5_action_acc:.4f} | Top-5 Adverb: {top5_adverb_acc:.4f}')
    print(f'  Action F1: {joint_metrics["action_f1"]:.4f} | Adverb F1: {joint_metrics["adverb_f1"]:.4f}')

def calculate_p1_action(dset, scores, action_gt):
    pair_pred = np.argmax(scores.numpy(), axis=1)
    action_pred = [dset.action2idx[dset.pairs[pred][1]] for pred in pair_pred]
    acc = (action_pred == action_gt.cpu().numpy()).mean()
    return acc

if __name__ == '__main__':
    args = parser.parse_args()
    if args.modality == 'both':
        args.modality = ['rgb', 'flow']
    else:
        args.modality = [args.modality]
    main(args)
