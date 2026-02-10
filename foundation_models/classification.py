import logging
import pandas as pd
import argparse
import numpy as np
import av
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import os
import json


def per_class_acc(results, label_list, k):
    """Return ({label: top-k accuracy}, {label: count}) for each class in results."""
    from collections import defaultdict
    hits = defaultdict(int)
    counts = defaultdict(int)
    for r in results:
        label = label_list[r['gt']]
        counts[label] += 1
        if r['gt'] in r['ranked'][:k]:
            hits[label] += 1
    acc = {label: hits[label] / counts[label] for label in counts}
    return acc, dict(counts)


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def load_video(video_path, clip_len=8, frame_sample_rate=1):
    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream found in {video_path}")
    indices = sample_frame_indices(
        clip_len=clip_len,
        frame_sample_rate=frame_sample_rate,
        seg_len=container.streams.video[0].frames
    )
    video = read_video_pyav(container, indices)
    pil_frames = [Image.fromarray(frame) for frame in video]
    return pil_frames


def top_k_acc(results, k):
    return sum(r['gt'] in r['ranked'][:k] for r in results) / len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for classification')
    ACTION_COLUMN = 'clustered_action'
    ADVERB_COLUMN = 'clustered_adverb'
    parser.add_argument('--csv_path', type=str, default='/home/gmago/AA/PseudoAdverbs/datasets/VATEX_Adverbs/annotations.csv', help='Path to the CSV file containing the annotations')
    parser.add_argument('--model_name', type=str, default='microsoft/xclip-base-patch32', help='Name of the pre-trained model to use')
    parser.add_argument('--video_dir', type=str, default='/ivi/zfs/s0/original_homes/gmago/AA/VaTeX/videos/vatex-dataset', help='Directory containing the video files')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--sample_method', type=str, default='random', choices=['random', 'sequential'], help='Method to sample videos from the dataset')
    parser.add_argument('--action_caption_template', type=str, default='Action being performed in the video: {action}', help='Template for action captions')
    parser.add_argument('--adverb_caption_template', type=str, default='Most relevant adverb for the action being performed in the video: {adverb}', help='Template for adverb captions')
    parser.add_argument('--hierarchy_json_path', type=str, default='datasets/VATEX_Adverbs/action_adverb_hierarchy.json')
    parser.add_argument('--clip_len', type=int, default=8, help='Number of frames to sample per video')
    parser.add_argument('--output_path', type=str, default='classification_results.json', help='Path to save the output metrics JSON')
    parser.add_argument('--debug', action='store_true', help='Enable per-sample top-5 debug logs')
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('classification_eval.log'),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset from {args.csv_path}")
    dataset_csv = pd.read_csv(args.csv_path)

    # Sample rows
    if args.max_samples is not None and len(dataset_csv) > args.max_samples:
        if args.sample_method == 'random':
            dataset_csv = dataset_csv.sample(n=min(args.max_samples, len(dataset_csv)), random_state=42)
        else:
            dataset_csv = dataset_csv.head(args.max_samples)
        dataset_csv = dataset_csv.reset_index(drop=True)

    all_actions = dataset_csv[ACTION_COLUMN].unique().tolist()
    all_adverbs = dataset_csv[ADVERB_COLUMN].unique().tolist()

    logger.info(f"Unique actions: {len(all_actions)}, unique adverbs: {len(all_adverbs)}")

    action_to_idx = {a: i for i, a in enumerate(all_actions)}
    adverb_to_idx = {a: i for i, a in enumerate(all_adverbs)}

    candidate_action_captions = [args.action_caption_template.format(action=action) for action in all_actions]
    candidate_adverb_captions = [args.adverb_caption_template.format(adverb=adverb) for adverb in all_adverbs]

    print(f"Candidate action captions: {candidate_action_captions}")
    print(f"\nCandidate adverb captions: {candidate_adverb_captions}")

    logger.info(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")

    results_action = []
    results_adverb = []
    total = len(dataset_csv)
    skipped = 0

    for i, (_, row) in enumerate(dataset_csv.iterrows()):
        video_path = os.path.join(args.video_dir, f"{row['clip_id']}.mp4")

        if not os.path.exists(video_path):
            logger.warning(f"[{i+1}/{total}] Skipped (not found): {video_path}")
            skipped += 1
            continue

        logger.info(f"[{i+1}/{total}] Processing {video_path}")

        try:
            pil_frames = load_video(video_path, clip_len=args.clip_len)
        except (ValueError, Exception) as e:
            logger.warning(f"[{i+1}/{total}] Skipped (load error): {video_path} — {e}")
            skipped += 1
            continue

        gt_action_idx = action_to_idx[row[ACTION_COLUMN]]
        gt_adverb_idx = adverb_to_idx[row[ADVERB_COLUMN]]

        # Action pass
        text_inputs = processor.tokenizer(candidate_action_captions, return_tensors="pt", padding=True).to(device)
        video_inputs = processor.video_processor(pil_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**{**text_inputs, **video_inputs})
        action_logits = outputs.logits_per_video[0]
        ranked_action = action_logits.argsort(descending=True).cpu().tolist()
        results_action.append({'gt': gt_action_idx, 'ranked': ranked_action})

        # Adverb pass (reuse video_inputs)
        text_inputs = processor.tokenizer(candidate_adverb_captions, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**{**text_inputs, **video_inputs})
        adverb_logits = outputs.logits_per_video[0]
        ranked_adverb = adverb_logits.argsort(descending=True).cpu().tolist()
        results_adverb.append({'gt': gt_adverb_idx, 'ranked': ranked_adverb})

        # Debug: top-5 predictions with scores
        action_probs = action_logits.softmax(dim=0).cpu()
        adverb_probs = adverb_logits.softmax(dim=0).cpu()
        top5_actions = [(all_actions[idx], action_probs[idx].item()) for idx in ranked_action[:5]]
        top5_adverbs = [(all_adverbs[idx], adverb_probs[idx].item()) for idx in ranked_adverb[:5]]
        gt_action_rank = ranked_action.index(gt_action_idx) + 1
        gt_adverb_rank = ranked_adverb.index(gt_adverb_idx) + 1
        logger.debug(
            f"  ACTION gt={row[ACTION_COLUMN]!r} (rank {gt_action_rank}/{len(all_actions)}) "
            f"top5={top5_actions}"
        )
        logger.debug(
            f"  ADVERB gt={row[ADVERB_COLUMN]!r} (rank {gt_adverb_rank}/{len(all_adverbs)}) "
            f"top5={top5_adverbs}"
        )

    logger.info(f"Processed: {len(results_action)}/{total}  Skipped (missing): {skipped}")

    if not results_action:
        logger.error("No samples were processed. Check video paths.")
    else:
        action_top1 = top_k_acc(results_action, 1)
        action_top5 = top_k_acc(results_action, 5)
        adverb_top1 = top_k_acc(results_adverb, 1)
        adverb_top5 = top_k_acc(results_adverb, 5)
        joint_top1 = sum(
            ra['gt'] == ra['ranked'][0] and rb['gt'] == rb['ranked'][0]
            for ra, rb in zip(results_action, results_adverb)
        ) / len(results_action)
        joint_top5 = sum(
            ra['gt'] in ra['ranked'][:5] and rb['gt'] in rb['ranked'][:5]
            for ra, rb in zip(results_action, results_adverb)
        ) / len(results_action)

        # Per-class breakdowns
        per_action_top1, action_counts = per_class_acc(results_action, all_actions, 1)
        per_action_top5, _             = per_class_acc(results_action, all_actions, 5)
        per_adverb_top1, adverb_counts = per_class_acc(results_adverb, all_adverbs, 1)
        per_adverb_top5, _             = per_class_acc(results_adverb, all_adverbs, 5)

        logger.info(f"Action  Top-1: {action_top1:.4f}  Top-5: {action_top5:.4f}")
        logger.info(f"Adverb  Top-1: {adverb_top1:.4f}  Top-5: {adverb_top5:.4f}")
        logger.info(f"Joint   Top-1: {joint_top1:.4f}  Top-5: {joint_top5:.4f}")

        logger.info("--- Per-action (top1 / top5 / n_samples) ---")
        for label in sorted(per_action_top1):
            n = action_counts[label]
            logger.info(f"  {label}: top1={per_action_top1[label]:.4f}  top5={per_action_top5[label]:.4f}  n={n}")

        logger.info("--- Per-adverb (top1 / top5 / n_samples) ---")
        for label in sorted(per_adverb_top1):
            n = adverb_counts[label]
            logger.info(f"  {label}: top1={per_adverb_top1[label]:.4f}  top5={per_adverb_top5[label]:.4f}  n={n}")

        metrics = {
            'model': args.model_name,
            'num_samples': len(results_action),
            'skipped': skipped,
            'action_top1': action_top1,
            'action_top5': action_top5,
            'adverb_top1': adverb_top1,
            'adverb_top5': adverb_top5,
            'joint_top1': joint_top1,
            'joint_top5': joint_top5,
            'per_action': {
                label: {'top1': per_action_top1[label], 'top5': per_action_top5[label], 'n': action_counts[label]}
                for label in per_action_top1
            },
            'per_adverb': {
                label: {'top1': per_adverb_top1[label], 'top5': per_adverb_top5[label], 'n': adverb_counts[label]}
                for label in per_adverb_top1
            },
        }
        with open(args.output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {args.output_path}")
