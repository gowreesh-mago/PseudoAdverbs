import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import logging
import argparse
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='OpenGVLab/InternVideo2_Stage2_6B_224p')
    parser.add_argument('--video_dir', type=str, default='datasets/VATEX_Adverbs/videos')
    parser.add_argument('--annotations', type=str, default='datasets/VATEX_Adverbs/annotations.csv')
    parser.add_argument('--adverbs_file', type=str, default='datasets/VATEX_Adverbs/adverbs.csv')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_adverb_data(adverbs_file):
    df = pd.read_csv(adverbs_file)
    antonym_map = dict(zip(df['adverb'], df['antonym']))
    all_adverbs = list(set(df['adverb'].tolist() + df['antonym'].tolist()))
    return antonym_map, all_adverbs

def replace_adverb_in_caption(caption, adverb, replacement):
    pattern = r'\b' + re.escape(adverb) + r'\b'
    return re.sub(pattern, replacement, caption, flags=re.IGNORECASE)

def generate_negative_captions(caption, original_adverb, all_adverbs):
    negatives = []
    negative_adverbs = []
    for adv in all_adverbs:
        if adv != original_adverb:
            neg_caption = replace_adverb_in_caption(caption, original_adverb, adv)
            if neg_caption != caption:
                negatives.append(neg_caption)
                negative_adverbs.append(adv)
    return negatives, negative_adverbs

def load_video(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    frames = frames / 255.0
    return frames

def compute_similarity(video_embeds, text_embeds):
    video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return video_embeds @ text_embeds.T

def compute_recall(similarity_matrix, k_values=[1, 5, 10]):
    n = similarity_matrix.shape[0]
    gt = torch.arange(n, device=similarity_matrix.device)
    recalls = {}
    for k in k_values:
        if k > similarity_matrix.shape[1]:
            recalls[f'R@{k}'] = float('nan')
            continue
        topk = similarity_matrix.topk(k, dim=1).indices
        correct = (topk == gt.unsqueeze(1)).any(dim=1).float().sum()
        recalls[f'R@{k}'] = (correct / n * 100).item()
    return recalls

def compute_adverb_negative_recall(video_embed, original_embed, negative_embeds, k_values=[1, 3, 5]):
    video_embed = video_embed / video_embed.norm(dim=-1, keepdim=True)
    original_embed = original_embed / original_embed.norm(dim=-1, keepdim=True)
    negative_embeds = negative_embeds / negative_embeds.norm(dim=-1, keepdim=True)

    all_embeds = torch.cat([original_embed.unsqueeze(0), negative_embeds], dim=0)
    scores = (video_embed @ all_embeds.T).squeeze(0)

    rankings = scores.argsort(descending=True)
    gt_rank = (rankings == 0).nonzero(as_tuple=True)[0].item() + 1

    recalls = {}
    for k in k_values:
        if k >= len(rankings):
            recalls[f'R@{k}'] = float(gt_rank == 1) * 100
        else:
            recalls[f'R@{k}'] = float(gt_rank <= k) * 100

    return recalls, gt_rank, scores

def log_retrieval_examples(similarity_matrix, texts, direction='v2t', num_examples=5):
    logger.info(f"\n{'='*60}")
    logger.info(f"{'VIDEO-TO-TEXT' if direction == 'v2t' else 'TEXT-TO-VIDEO'} RETRIEVAL EXAMPLES")
    logger.info(f"{'='*60}")

    n = min(num_examples, similarity_matrix.shape[0])
    indices = torch.randperm(similarity_matrix.shape[0])[:n]

    for idx in indices:
        idx = idx.item()
        if direction == 'v2t':
            rankings = similarity_matrix[idx].argsort(descending=True)
            gt_rank = (rankings == idx).nonzero(as_tuple=True)[0].item() + 1
            top5 = rankings[:5].tolist()
            logger.info(f"\nQuery Video {idx}:")
            logger.info(f"  Ground Truth Text: \"{texts[idx]}\"")
            logger.info(f"  GT Rank: {gt_rank}")
            logger.info(f"  Top-5 Retrieved:")
            for i, t_idx in enumerate(top5):
                marker = " <-- GT" if t_idx == idx else ""
                logger.info(f"    {i+1}. \"{texts[t_idx]}\"{marker}")
        else:
            rankings = similarity_matrix[:, idx].argsort(descending=True)
            gt_rank = (rankings == idx).nonzero(as_tuple=True)[0].item() + 1
            top5 = rankings[:5].tolist()
            logger.info(f"\nQuery Text {idx}: \"{texts[idx]}\"")
            logger.info(f"  GT Rank: {gt_rank}")
            logger.info(f"  Top-5 Retrieved Videos: {top5}")
            if idx in top5:
                logger.info(f"    (Correct video {idx} at position {top5.index(idx)+1})")

def log_adverb_negative_examples(examples, num_examples=5):
    logger.info(f"\n{'='*60}")
    logger.info("ADVERB NEGATIVE RETRIEVAL EXAMPLES")
    logger.info(f"{'='*60}")

    n = min(num_examples, len(examples))
    indices = np.random.permutation(len(examples))[:n]

    for idx in indices:
        ex = examples[idx]
        logger.info(f"\nVideo {ex['video_idx']} (Adverb: {ex['adverb']}):")
        logger.info(f"  Original Caption: \"{ex['original_text']}\"")
        logger.info(f"  Num Negatives: {ex['num_negatives']}")
        logger.info(f"  GT Rank: {ex['gt_rank']} / {ex['num_negatives'] + 1}")

        sorted_indices = ex['scores'].argsort(descending=True).tolist()
        logger.info(f"  Top-5 Rankings:")
        for i, rank_idx in enumerate(sorted_indices[:5]):
            score = ex['scores'][rank_idx].item()
            if rank_idx == 0:
                logger.info(f"    {i+1}. [ORIGINAL] \"{ex['original_text']}\" (score={score:.4f})")
            else:
                neg_idx = rank_idx - 1
                if neg_idx < len(ex['negative_texts']):
                    logger.info(f"    {i+1}. [NEG:{ex['negative_adverbs'][neg_idx]}] \"{ex['negative_texts'][neg_idx]}\" (score={score:.4f})")

def compute_per_adverb_metrics(examples):
    adverb_results = {}
    for ex in examples:
        adv = ex['adverb']
        if adv not in adverb_results:
            adverb_results[adv] = {'correct': 0, 'total': 0, 'ranks': [], 'num_candidates': []}
        adverb_results[adv]['total'] += 1
        adverb_results[adv]['ranks'].append(ex['gt_rank'])
        adverb_results[adv]['num_candidates'].append(ex['num_negatives'] + 1)
        if ex['gt_rank'] == 1:
            adverb_results[adv]['correct'] += 1

    per_adverb = {}
    for adv, data in adverb_results.items():
        per_adverb[adv] = {
            'R@1': data['correct'] / data['total'] * 100,
            'count': data['total'],
            'mean_rank': np.mean(data['ranks']),
            'mean_candidates': np.mean(data['num_candidates'])
        }
    return per_adverb

def main():
    args = parse_args()

    logger.info(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    logger.info(f"Loading adverb data from {args.adverbs_file}")
    antonym_map, all_adverbs = load_adverb_data(args.adverbs_file)
    logger.info(f"Loaded {len(antonym_map)} antonym pairs, {len(all_adverbs)} unique adverbs")

    logger.info(f"Loading annotations from {args.annotations}")
    df = pd.read_csv(args.annotations)

    if args.max_samples:
        df = df.head(args.max_samples)

    logger.info(f"Total samples: {len(df)}")

    video_embeddings = []
    text_embeddings = []
    valid_texts = []
    valid_indices = []
    adverb_negative_examples = []

    logger.info("Computing embeddings...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(args.video_dir, f"{row['clip_id']}.mp4")

        if not os.path.exists(video_path):
            continue

        adverb = row.get('adverb', row.get('clustered_adverb', None))

        try:
            video = load_video(video_path, args.num_frames)
            video = video.unsqueeze(0).to(args.device, dtype=torch.float16)

            text = row['caption'] if 'caption' in row else f"{row['action']} {row['adverb']}"

            text_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text_inputs = {k: v.to(args.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                video_embed = model.encode_video(video)
                text_embed = model.encode_text(text_inputs)

            video_embeddings.append(video_embed.cpu())
            text_embeddings.append(text_embed.cpu())
            valid_texts.append(text)
            valid_indices.append(idx)

            if adverb is not None and adverb in all_adverbs:
                negative_texts, negative_adverbs = generate_negative_captions(text, adverb, all_adverbs)

                if len(negative_texts) > 0:
                    neg_inputs = tokenizer(negative_texts, return_tensors='pt', padding=True, truncation=True)
                    neg_inputs = {k: v.to(args.device) for k, v in neg_inputs.items()}

                    with torch.no_grad():
                        neg_embeds = model.encode_text(neg_inputs)

                    recalls, gt_rank, scores = compute_adverb_negative_recall(
                        video_embed.cpu(), text_embed.cpu(), neg_embeds.cpu()
                    )

                    adverb_negative_examples.append({
                        'video_idx': len(valid_indices) - 1,
                        'adverb': adverb,
                        'original_text': text,
                        'negative_texts': negative_texts,
                        'negative_adverbs': negative_adverbs,
                        'num_negatives': len(negative_texts),
                        'gt_rank': gt_rank,
                        'recalls': recalls,
                        'scores': scores
                    })

        except Exception as e:
            logger.warning(f"Failed to process {video_path}: {e}")
            continue

    if len(video_embeddings) == 0:
        logger.error("No valid samples found!")
        return

    video_embeddings = torch.cat(video_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    logger.info(f"\nProcessed {len(valid_indices)} samples successfully")
    logger.info(f"Video embeddings shape: {video_embeddings.shape}")
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    logger.info(f"Adverb negative samples: {len(adverb_negative_examples)}")

    similarity = compute_similarity(video_embeddings, text_embeddings)

    logger.info("\n" + "="*60)
    logger.info("STANDARD VIDEO-TO-TEXT RETRIEVAL")
    logger.info("(Each video queries ALL captions in dataset)")
    logger.info("="*60)
    v2t_recalls = compute_recall(similarity)
    for metric, value in v2t_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    log_retrieval_examples(similarity, valid_texts, direction='v2t', num_examples=args.num_examples)

    logger.info("\n" + "="*60)
    logger.info("STANDARD TEXT-TO-VIDEO RETRIEVAL")
    logger.info("(Each caption queries ALL videos in dataset)")
    logger.info("="*60)
    t2v_recalls = compute_recall(similarity.T)
    for metric, value in t2v_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    log_retrieval_examples(similarity, valid_texts, direction='t2v', num_examples=args.num_examples)

    if len(adverb_negative_examples) > 0:
        logger.info("\n" + "="*60)
        logger.info("ADVERB NEGATIVE VIDEO-TO-TEXT RETRIEVAL")
        logger.info("(Each video ranks: 1 original + K negative captions)")
        logger.info("="*60)

        all_r1 = [ex['recalls']['R@1'] for ex in adverb_negative_examples]
        all_r3 = [ex['recalls']['R@3'] for ex in adverb_negative_examples]
        all_r5 = [ex['recalls']['R@5'] for ex in adverb_negative_examples]
        all_ranks = [ex['gt_rank'] for ex in adverb_negative_examples]
        all_num_neg = [ex['num_negatives'] for ex in adverb_negative_examples]

        logger.info(f"  Samples evaluated: {len(adverb_negative_examples)}")
        logger.info(f"  Avg negatives per sample: {np.mean(all_num_neg):.1f}")
        logger.info(f"  R@1: {np.mean(all_r1):.2f}%")
        logger.info(f"  R@3: {np.mean(all_r3):.2f}%")
        logger.info(f"  R@5: {np.mean(all_r5):.2f}%")
        logger.info(f"  Mean Rank: {np.mean(all_ranks):.2f}")
        logger.info(f"  Median Rank: {np.median(all_ranks):.2f}")

        log_adverb_negative_examples(adverb_negative_examples, num_examples=args.num_examples)

        logger.info("\n" + "="*60)
        logger.info("PER-ADVERB NEGATIVE RETRIEVAL RESULTS")
        logger.info("="*60)

        per_adverb = compute_per_adverb_metrics(adverb_negative_examples)
        sorted_adverbs = sorted(per_adverb.items(), key=lambda x: x[1]['R@1'], reverse=True)

        for adv, metrics in sorted_adverbs:
            logger.info(f"  {adv:20s} | R@1: {metrics['R@1']:6.2f}% | Count: {metrics['count']:4d} | Mean Rank: {metrics['mean_rank']:.2f} | Avg Candidates: {metrics['mean_candidates']:.1f}")

    logger.info("\n" + "="*60)
    logger.info("OVERALL PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Samples: {len(valid_indices)}")

    logger.info(f"\nStandard Video-to-Text Retrieval:")
    for metric, value in v2t_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    logger.info(f"\nStandard Text-to-Video Retrieval:")
    for metric, value in t2v_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    avg_r1 = (v2t_recalls['R@1'] + t2v_recalls['R@1']) / 2
    avg_r5 = (v2t_recalls['R@5'] + t2v_recalls['R@5']) / 2
    avg_r10 = (v2t_recalls['R@10'] + t2v_recalls['R@10']) / 2

    logger.info(f"\nAverage (V2T + T2V):")
    logger.info(f"  R@1: {avg_r1:.2f}%")
    logger.info(f"  R@5: {avg_r5:.2f}%")
    logger.info(f"  R@10: {avg_r10:.2f}%")

    if len(adverb_negative_examples) > 0:
        logger.info(f"\nAdverb Negative V2T Retrieval:")
        logger.info(f"  Samples: {len(adverb_negative_examples)}")
        logger.info(f"  R@1: {np.mean(all_r1):.2f}%")
        logger.info(f"  Mean Rank: {np.mean(all_ranks):.2f}")

if __name__ == '__main__':
    main()
