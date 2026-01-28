import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import XCLIPProcessor, XCLIPModel
import cv2
from PIL import Image
import logging
import argparse
import re
import json
import csv
import random
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ABSTRACT MODEL INTERFACE
# ============================================================================

class VideoTextModel(ABC):
    """Abstract base class for video-text models."""

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass

    @abstractmethod
    def encode_video(self, video_frames: List[Image.Image]) -> torch.Tensor:
        """
        Encode video frames into embeddings.

        Args:
            video_frames: List of PIL Images

        Returns:
            torch.Tensor of shape (1, embedding_dim) - normalized embeddings
        """
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text into embeddings.

        Args:
            texts: List of text strings

        Returns:
            torch.Tensor of shape (len(texts), embedding_dim) - normalized embeddings
        """
        pass

    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()

# ============================================================================
# X-CLIP MODEL WRAPPER
# ============================================================================

class XCLIPModelWrapper(VideoTextModel):
    """Wrapper for X-CLIP model."""

    def load_model(self):
        """Load X-CLIP model and processor."""
        logger.info(f"Loading X-CLIP model: {self.model_name}")
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"X-CLIP model loaded successfully")

    def encode_video(self, video_frames: List[Image.Image]) -> torch.Tensor:
        """Encode video frames using X-CLIP."""
        with torch.no_grad():
            logger.debug(f"Processing video with X-CLIP processor...")
            logger.debug(f"  Input: {len(video_frames)} frames")
            logger.debug(f"  Frame types: {[type(f) for f in video_frames[:3]]}")
            if len(video_frames) > 0:
                logger.debug(f"  First frame size: {video_frames[0].size}, mode: {video_frames[0].mode}")

            # Convert PIL Images to numpy array as expected by XCLIPProcessor
            # Processor expects: videos=list(video) where video is np.ndarray of shape (num_frames, H, W, C)
            logger.debug(f"  Converting PIL images to numpy array...")
            video_array = np.stack([np.array(frame) for frame in video_frames])
            logger.debug(f"  Video array shape: {video_array.shape}")

            # Try calling the video_processor/image_processor directly
            # In transformers 5.0, XCLIPProcessor has video_processor attribute
            logger.debug(f"  Trying direct video_processor call...")
            try:
                if hasattr(self.processor, 'video_processor'):
                    logger.debug(f"  Using processor.video_processor")
                    inputs = self.processor.video_processor(videos=list(video_array), return_tensors="pt")
                elif hasattr(self.processor, 'image_processor'):
                    logger.debug(f"  Using processor.image_processor with videos parameter")
                    inputs = self.processor.image_processor(videos=list(video_array), return_tensors="pt")
                else:
                    logger.debug(f"  Falling back to main processor")
                    inputs = self.processor(videos=list(video_array), return_tensors="pt")
            except Exception as e:
                logger.debug(f"  Direct processor call failed: {e}")
                logger.debug(f"  Trying with images parameter instead...")
                # Try processing as images instead
                inputs = self.processor.image_processor(images=video_frames, return_tensors="pt")

            logger.debug(f"  Processor returned {len(inputs)} items")
            logger.debug(f"  Processor keys: {list(inputs.keys())}")
            logger.debug(f"  Video input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            video_outputs = self.model.get_video_features(**inputs)
            logger.debug(f"  Video features type: {type(video_outputs)}")

            # Extract embeddings from BaseModelOutputWithPooling
            # get_video_features returns a BaseModelOutputWithPooling object
            # We want the pooler_output which contains the video embeddings
            if hasattr(video_outputs, 'pooler_output'):
                video_features = video_outputs.pooler_output
                logger.debug(f"  Using pooler_output, shape: {video_features.shape}")
            elif hasattr(video_outputs, 'last_hidden_state'):
                # Fallback: use mean pooling of last hidden state
                video_features = video_outputs.last_hidden_state.mean(dim=1)
                logger.debug(f"  Using mean-pooled last_hidden_state, shape: {video_features.shape}")
            else:
                # If it's already a tensor
                video_features = video_outputs
                logger.debug(f"  Using direct output, shape: {video_features.shape}")

            # Normalize
            video_embed = video_features / video_features.norm(dim=-1, keepdim=True)
            logger.debug(f"  Normalized video embed shape: {video_embed.shape}")
            return video_embed

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using X-CLIP."""
        with torch.no_grad():
            logger.debug(f"Processing {len(texts)} text(s) with X-CLIP processor...")
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            logger.debug(f"  Text input shapes: {[(k, v.shape) for k, v in text_inputs.items()]}")
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_outputs = self.model.get_text_features(**text_inputs)
            logger.debug(f"  Text features type: {type(text_outputs)}")

            # Extract embeddings from BaseModelOutputWithPooling if needed
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
                logger.debug(f"  Using pooler_output, shape: {text_features.shape}")
            elif hasattr(text_outputs, 'last_hidden_state'):
                text_features = text_outputs.last_hidden_state.mean(dim=1)
                logger.debug(f"  Using mean-pooled last_hidden_state, shape: {text_features.shape}")
            else:
                # Already a tensor
                text_features = text_outputs
                logger.debug(f"  Using direct output, shape: {text_features.shape}")

            # Normalize
            text_embed = text_features / text_features.norm(dim=-1, keepdim=True)
            logger.debug(f"  Normalized text embed shape: {text_embed.shape}")
            return text_embed

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_name: str, device: str) -> VideoTextModel:
    """
    Factory function to create model based on model name.

    Args:
        model_name: Name of the model (e.g., 'microsoft/xclip-base-patch32')
        device: Device to load model on

    Returns:
        VideoTextModel instance
    """
    # Determine model type based on name
    if 'xclip' in model_name.lower():
        model = XCLIPModelWrapper(model_name, device)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported models: X-CLIP")

    model.load_model()
    return model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_antonym_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    antonym_map = {}
    for pair in data['adverb_antonym_pairs']:
        adverb = pair['adverb'].lower()  # Normalize to lowercase
        antonym_map[adverb] = pair['replacements']
    return antonym_map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/xclip-base-patch32')
    parser.add_argument('--video_dir', type=str, default='datasets/VATEX_Adverbs/videos')
    parser.add_argument('--annotations', type=str, default='datasets/VATEX_Adverbs/annotations.csv')
    parser.add_argument('--adverbs_file', type=str, default='datasets/VATEX_Adverbs/adverbs.csv',
                        help='CSV file with adverb-antonym pairs (used if --hierarchy is not provided)')
    parser.add_argument('--hierarchy', type=str, default=None,
                        help='JSON file with adverb hierarchy and replacements (alternative to --adverbs_file)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of negative captions to generate per sample (when using --hierarchy)')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all samples)')
    parser.add_argument('--random_sample', action='store_true',
                        help='Randomly sample max_samples instead of taking first N')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save logs and metrics (CSV and JSON files). If not provided, results are only logged to console.')
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving any output files (logs, metrics). Only display results in console.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging for detailed execution trace')
    parser.add_argument('--use_time_limits', action='store_true',
                        help='Sample frames between start_time and end_time from CSV (default: sample uniformly across entire video)')
    return parser.parse_args()

def load_adverb_data(adverbs_file):
    df = pd.read_csv(adverbs_file)
    antonym_map = dict(zip(df['adverb'], df['antonym']))
    all_adverbs = list(set(df['adverb'].tolist() + df['antonym'].tolist()))
    return antonym_map, all_adverbs

def setup_logging(output_dir: Optional[str] = None, debug: bool = False, no_save: bool = False):
    """
    Setup logging to both console and file.

    Args:
        output_dir: Directory to save log file (if None, only logs to console)
        debug: Enable debug logging
        no_save: If True, disable file logging even if output_dir is specified
    """
    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if output_dir is specified and no_save is False
    if output_dir and not no_save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'retrieval_eval_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return output_dir

def replace_adverb_in_caption(caption, adverb, replacements, k=1):
    """
    Replace adverb in caption with one or more replacements, preserving case.

    Args:
        caption: Original caption text
        adverb: Adverb to replace
        replacements: Single replacement string or list of replacement strings
        k: Number of replacements to select (if replacements is a list)

    Returns:
        List of modified captions (or single caption if replacements is a string)
    """
    # Handle single replacement string
    if isinstance(replacements, str):
        replacements = [replacements]
        return_single = True
    else:
        return_single = False

    pattern = re.compile(r'\b' + re.escape(adverb) + r'\b', re.IGNORECASE)
    selected = random.sample(replacements, min(k, len(replacements)))

    results = []
    for replacement in selected:
        def match_case(match, repl=replacement):
            word = match.group()
            if word.isupper():
                return repl.upper()
            elif word[0].isupper():
                return repl.capitalize()
            return repl
        results.append(pattern.sub(match_case, caption))

    return results[0] if return_single else results

def generate_negative_captions(caption, original_adverb, all_adverbs=None, antonym_map=None, k=None, clustered_adverb=None):
    """
    Generate negative captions by replacing the adverb.

    Args:
        caption: Original caption text
        original_adverb: The adverb to replace in the caption text
        all_adverbs: List of all possible adverbs (for exhaustive replacement)
        antonym_map: Dictionary mapping adverbs to their replacements (from hierarchy)
        k: Number of negative captions to generate (when using antonym_map)
        clustered_adverb: The clustered/canonical adverb form for hierarchy lookup

    Returns:
        Tuple of (negative_captions, negative_adverbs)
    """
    negatives = []
    negative_adverbs = []

    # Use hierarchy-based approach if antonym_map is provided
    # Use clustered_adverb for lookup, but original_adverb for text replacement
    lookup_adverb = clustered_adverb if clustered_adverb is not None else original_adverb
    if antonym_map is not None and lookup_adverb in antonym_map:
        replacements = antonym_map[lookup_adverb]
        if k is not None:
            # Generate k negative captions
            negative_captions = replace_adverb_in_caption(caption, original_adverb, replacements, k)
            if not isinstance(negative_captions, list):
                negative_captions = [negative_captions]
            for neg_cap in negative_captions:
                if neg_cap != caption:
                    negatives.append(neg_cap)
                    # Extract which replacement was used (approximate)
                    for repl in replacements:
                        if repl.lower() in neg_cap.lower() and repl.lower() != original_adverb.lower():
                            negative_adverbs.append(repl)
                            break
                    else:
                        negative_adverbs.append(replacements[0])  # fallback
        else:
            # Generate negative caption for each replacement
            for repl in replacements:
                neg_caption = replace_adverb_in_caption(caption, original_adverb, repl)
                if neg_caption != caption:
                    negatives.append(neg_caption)
                    negative_adverbs.append(repl)
    # Use all-adverbs approach if provided
    elif all_adverbs is not None:
        for adv in all_adverbs:
            if adv != original_adverb:
                neg_caption = replace_adverb_in_caption(caption, original_adverb, adv)
                if neg_caption != caption:
                    negatives.append(neg_caption)
                    negative_adverbs.append(adv)

    return negatives, negative_adverbs

class NegativeCaptionDataset(Dataset):
    """Dataset for loading video annotations with negative captions based on adverb replacements."""

    def __init__(self, annotations_path, hierarchy_path, k=1):
        self.antonym_map = load_antonym_mapping(hierarchy_path)
        self.k = k

        with open(annotations_path, 'r') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        adverb = row['adverb'].lower()
        clustered_adverb = row['clustered_adverb'].lower()
        caption = row['caption']

        if clustered_adverb in self.antonym_map:
            negative_captions = replace_adverb_in_caption(caption, adverb, self.antonym_map[clustered_adverb], self.k)
        else:
            negative_captions = [caption] * self.k

        return {
            'clip_id': row['clip_id'],
            'caption': caption,
            'negative_captions': negative_captions,
            'action': row['action'],
            'adverb': adverb,
            'clustered_adverb': clustered_adverb
        }

def load_video(video_path, num_frames=8, start_time=None, end_time=None):
    """
    Load video frames from a video file using cv2 for X-CLIP.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        start_time: Start time in seconds (if None, starts from beginning)
        end_time: End time in seconds (if None, goes to end)

    Returns:
        List of PIL Images
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.debug(f"Loading video: {os.path.basename(video_path)}")
    logger.debug(f"  FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")

    if start_time is not None and end_time is not None:
        # Validate time range is within video duration
        if start_time >= duration or end_time > duration + 1.0:
            logger.warning(f"  Requested time range ({start_time:.2f}s-{end_time:.2f}s) exceeds video duration ({duration:.2f}s)")
            logger.warning(f"  Falling back to sampling entire video")
            start_frame = 0
            end_frame = total_frames
        else:
            # Clamp to valid range within the video
            start_time_clamped = max(0, min(start_time, duration))
            end_time_clamped = max(start_time_clamped + 0.1, min(end_time, duration))

            start_frame = int(start_time_clamped * fps)
            end_frame = int(end_time_clamped * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            logger.debug(f"  Time range: {start_time:.2f}s to {end_time:.2f}s (frames {start_frame}-{end_frame})")
    else:
        start_frame = 0
        end_frame = total_frames
        logger.debug(f"  Sampling entire video (0-{duration:.2f}s)")

    # Calculate which frames to extract
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    logger.debug(f"  Sampling {num_frames} frames at indices: {frame_indices.tolist()}")

    frames = []
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            logger.debug(f"    Frame {idx}: {frame_rgb.shape} -> PIL Image {pil_image.size}")

    video.release()

    # Ensure we have the right number of frames
    if len(frames) < num_frames and len(frames) > 0:
        # Duplicate last frame if needed
        original_count = len(frames)
        while len(frames) < num_frames:
            frames.append(frames[-1])
        logger.debug(f"  Padded from {original_count} to {len(frames)} frames")

    logger.debug(f"  Successfully loaded {len(frames)} frames")
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

    # Concatenate: original_embed is [1, 512], negative_embeds is [k, 512] -> [k+1, 512]
    all_embeds = torch.cat([original_embed, negative_embeds], dim=0)
    scores = (video_embed @ all_embeds.T).squeeze()

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
    """
    Compute comprehensive per-adverb metrics for negative retrieval.

    Returns:
        Dictionary mapping adverb -> metrics dict with R@1, R@3, R@5, MRR, ranks, etc.
    """
    adverb_results = {}
    for ex in examples:
        adv = ex['adverb']
        if adv not in adverb_results:
            adverb_results[adv] = {
                'r1_correct': 0,
                'r3_correct': 0,
                'r5_correct': 0,
                'total': 0,
                'ranks': [],
                'reciprocal_ranks': [],
                'num_candidates': []
            }

        adverb_results[adv]['total'] += 1
        adverb_results[adv]['ranks'].append(ex['gt_rank'])
        adverb_results[adv]['reciprocal_ranks'].append(1.0 / ex['gt_rank'])
        adverb_results[adv]['num_candidates'].append(ex['num_negatives'] + 1)

        # Track R@k metrics
        if ex['gt_rank'] == 1:
            adverb_results[adv]['r1_correct'] += 1
        if ex['gt_rank'] <= 3:
            adverb_results[adv]['r3_correct'] += 1
        if ex['gt_rank'] <= 5:
            adverb_results[adv]['r5_correct'] += 1

    per_adverb = {}
    for adv, data in adverb_results.items():
        per_adverb[adv] = {
            'R@1': data['r1_correct'] / data['total'] * 100,
            'R@3': data['r3_correct'] / data['total'] * 100,
            'R@5': data['r5_correct'] / data['total'] * 100,
            'MRR': np.mean(data['reciprocal_ranks']) * 100,
            'count': data['total'],
            'mean_rank': np.mean(data['ranks']),
            'median_rank': np.median(data['ranks']),
            'mean_candidates': np.mean(data['num_candidates'])
        }
    return per_adverb

def compute_per_adverb_standard_retrieval(similarity_matrix, adverbs, direction='v2t', k_values=[1, 5, 10]):
    """
    Compute per-adverb metrics for standard retrieval (v2t or t2v).

    Args:
        similarity_matrix: Similarity matrix of shape (n_videos, n_texts) for v2t or transposed for t2v
        adverbs: List of adverbs corresponding to each sample
        direction: 'v2t' or 't2v'
        k_values: List of k values for R@k metrics

    Returns:
        Dictionary mapping adverb -> metrics dict
    """
    n = similarity_matrix.shape[0]
    adverb_results = {}

    for idx in range(n):
        adv = adverbs[idx]
        if adv not in adverb_results:
            adverb_results[adv] = {
                'total': 0,
                'ranks': [],
                'reciprocal_ranks': []
            }
            for k in k_values:
                adverb_results[adv][f'r{k}_correct'] = 0

        # Compute rank for this sample
        if direction == 'v2t':
            rankings = similarity_matrix[idx].argsort(descending=True)
        else:  # t2v
            rankings = similarity_matrix[:, idx].argsort(descending=True)

        gt_rank = (rankings == idx).nonzero(as_tuple=True)[0].item() + 1

        adverb_results[adv]['total'] += 1
        adverb_results[adv]['ranks'].append(gt_rank)
        adverb_results[adv]['reciprocal_ranks'].append(1.0 / gt_rank)

        # Track R@k metrics
        for k in k_values:
            if k <= similarity_matrix.shape[1] and gt_rank <= k:
                adverb_results[adv][f'r{k}_correct'] += 1

    # Compute final metrics
    per_adverb = {}
    for adv, data in adverb_results.items():
        metrics = {
            'count': data['total'],
            'mean_rank': np.mean(data['ranks']),
            'median_rank': np.median(data['ranks']),
            'MRR': np.mean(data['reciprocal_ranks']) * 100
        }
        for k in k_values:
            if k <= similarity_matrix.shape[1]:
                metrics[f'R@{k}'] = data[f'r{k}_correct'] / data['total'] * 100
            else:
                metrics[f'R@{k}'] = float('nan')
        per_adverb[adv] = metrics

    return per_adverb

def main():
    args = parse_args()

    # Setup logging
    setup_logging(output_dir=args.output_dir, debug=args.debug, no_save=args.no_save)

    if args.debug:
        logger.info("Debug logging ENABLED")

    if args.no_save:
        logger.info("File saving DISABLED (--no_save flag set)")

    logger.info(f"{'='*60}")
    logger.info(f"EVALUATION CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Video dir: {args.video_dir}")
    logger.info(f"Annotations: {args.annotations}")
    logger.info(f"Num frames: {args.num_frames}")
    logger.info(f"Use time limits: {args.use_time_limits}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Random sample: {args.random_sample}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"K (negatives): {args.k}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Save files: {not args.no_save}")
    logger.debug(f"Debug logging: ENABLED")
    logger.info(f"{'='*60}\n")

    # Create model using factory
    model = create_model(args.model_name, args.device)
    model.eval_mode()

    # Load adverb data - either from hierarchy JSON or CSV
    if args.hierarchy:
        logger.info(f"Loading adverb hierarchy from {args.hierarchy}")
        hierarchy_antonym_map = load_antonym_mapping(args.hierarchy)
        logger.info(f"Loaded {len(hierarchy_antonym_map)} adverbs from hierarchy")
        logger.debug(f"Hierarchy keys: {list(hierarchy_antonym_map.keys())}")
        all_adverbs = None
        csv_antonym_map = None
    else:
        logger.info(f"Loading adverb data from {args.adverbs_file}")
        csv_antonym_map, all_adverbs = load_adverb_data(args.adverbs_file)
        logger.info(f"Loaded {len(csv_antonym_map)} antonym pairs, {len(all_adverbs)} unique adverbs")
        hierarchy_antonym_map = None

    logger.info(f"Loading annotations from {args.annotations}")
    df = pd.read_csv(args.annotations)
    logger.debug(f"Loaded {len(df)} annotations")
    logger.debug(f"Columns: {df.columns.tolist()}")

    # Sample subset if requested
    if args.max_samples:
        if args.random_sample:
            np.random.seed(args.seed)
            sample_indices = np.random.choice(len(df), min(args.max_samples, len(df)), replace=False)
            df = df.iloc[sample_indices].reset_index(drop=True)
            logger.info(f"Randomly sampled {len(df)} samples (seed={args.seed})")
        else:
            df = df.head(args.max_samples)
            logger.info(f"Using first {len(df)} samples")
    else:
        logger.info(f"Using all samples")

    logger.info(f"Total samples to evaluate: {len(df)}")

    # Check if time-based sampling is available
    has_time_info = 'start_time' in df.columns and 'end_time' in df.columns
    if has_time_info:
        logger.info("Using time-based frame sampling (start_time, end_time from annotations)")
    else:
        logger.info("Using full-video frame sampling (no time info available)")

    video_embeddings = []
    text_embeddings = []
    valid_texts = []
    valid_indices = []
    valid_adverbs = []  # Track adverbs for per-adverb metrics
    adverb_negative_examples = []

    logger.info("Computing embeddings...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(args.video_dir, f"{row['clip_id']}.mp4")

        logger.debug(f"\n{'='*60}")
        logger.debug(f"Processing sample {idx}/{len(df)}: {row['clip_id']}")

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        adverb = row.get('adverb', row.get('clustered_adverb', None))
        clustered_adverb = row.get('clustered_adverb', adverb)

        # Normalize to lowercase for hierarchy lookup
        if clustered_adverb:
            clustered_adverb_lower = clustered_adverb.lower()
        else:
            clustered_adverb_lower = None

        logger.debug(f"Adverb info: adverb='{adverb}', clustered='{clustered_adverb}', lower='{clustered_adverb_lower}'")

        try:
            # Use time-based sampling if flag is enabled and times are available
            if args.use_time_limits and 'start_time' in row and 'end_time' in row:
                start_time = float(row['start_time']) if row['start_time'] else None
                end_time = float(row['end_time']) if row['end_time'] else None
                video_frames = load_video(video_path, args.num_frames, start_time, end_time)
            else:
                # Sample uniformly across entire video (default)
                video_frames = load_video(video_path, args.num_frames)
            logger.debug(f"Loaded {len(video_frames)} video frames")

            text = row['caption'] if 'caption' in row else f"{row['action']} {row['adverb']}"
            logger.debug(f"Caption: \"{text}\"")

            # Encode video and text using model interface
            video_embed = model.encode_video(video_frames)
            text_embed = model.encode_text([text])

            video_embeddings.append(video_embed.cpu())
            text_embeddings.append(text_embed.cpu())
            valid_texts.append(text)
            valid_indices.append(idx)
            # Store adverb for per-adverb metrics (use clustered_adverb_lower for consistency)
            valid_adverbs.append(clustered_adverb_lower if clustered_adverb_lower else adverb)
            logger.debug(f"Added to valid samples. Total valid: {len(valid_indices)}")

            # Generate negative captions based on available data
            # Use clustered_adverb_lower for hierarchy lookup (case-insensitive), adverb for all_adverbs lookup
            should_generate = (
                (hierarchy_antonym_map is not None and clustered_adverb_lower in hierarchy_antonym_map) or
                (all_adverbs is not None and adverb in all_adverbs)
            )

            logger.debug(f"Negative caption check:")
            logger.debug(f"  adverb is not None: {adverb is not None}")
            logger.debug(f"  hierarchy_antonym_map is not None: {hierarchy_antonym_map is not None}")
            logger.debug(f"  clustered_adverb_lower in hierarchy: {clustered_adverb_lower in hierarchy_antonym_map if hierarchy_antonym_map and clustered_adverb_lower else False}")
            logger.debug(f"  should_generate: {should_generate}")

            if adverb is not None and should_generate:
                logger.debug(f"Generating negative captions...")
                if hierarchy_antonym_map is not None:
                    # Use hierarchy-based approach with k replacements
                    # Use adverb for replacement in text, but clustered_adverb_lower for hierarchy lookup
                    negative_texts, negative_adverbs = generate_negative_captions(
                        text, adverb, antonym_map=hierarchy_antonym_map, k=args.k, clustered_adverb=clustered_adverb_lower
                    )
                    logger.debug(f"  Generated {len(negative_texts)} negative captions (hierarchy-based)")
                else:
                    # Use all-adverbs approach
                    negative_texts, negative_adverbs = generate_negative_captions(
                        text, adverb, all_adverbs=all_adverbs
                    )
                    logger.debug(f"  Generated {len(negative_texts)} negative captions (all-adverbs)")

                if len(negative_texts) > 0:
                    logger.debug(f"  Example negatives:")
                    for i, (neg_text, neg_adv) in enumerate(zip(negative_texts[:3], negative_adverbs[:3])):
                        logger.debug(f"    {i+1}. [{neg_adv}] \"{neg_text}\"")

                    logger.debug(f"  Encoding {len(negative_texts)} negative captions...")
                    # Encode negative captions using model interface
                    neg_embeds = []
                    for i, neg_text in enumerate(negative_texts):
                        neg_embed = model.encode_text([neg_text])
                        neg_embeds.append(neg_embed)
                        if i < 3:
                            logger.debug(f"    Negative {i+1} embed shape: {neg_embed.shape}")
                    neg_embeds = torch.cat(neg_embeds, dim=0)
                    logger.debug(f"  All negative embeds shape: {neg_embeds.shape}")

                    logger.debug(f"  Computing adverb negative recall...")
                    recalls, gt_rank, scores = compute_adverb_negative_recall(
                        video_embed.cpu(), text_embed.cpu(), neg_embeds.cpu()
                    )
                    logger.debug(f"  GT Rank: {gt_rank}/{len(negative_texts)+1}")
                    logger.debug(f"  Recalls: {recalls}")

                    adverb_negative_examples.append({
                        'video_idx': len(valid_indices) - 1,
                        'adverb': clustered_adverb_lower if hierarchy_antonym_map is not None else adverb,
                        'original_text': text,
                        'negative_texts': negative_texts,
                        'negative_adverbs': negative_adverbs,
                        'num_negatives': len(negative_texts),
                        'gt_rank': gt_rank,
                        'recalls': recalls,
                        'scores': scores
                    })
                    logger.debug(f"  Added to adverb negative examples. Total: {len(adverb_negative_examples)}")
                else:
                    logger.debug(f"  No negative captions generated")

        except Exception as e:
            logger.warning(f"Failed to process {video_path}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}", exc_info=True)
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING SUMMARY")
    logger.info(f"{'='*60}")

    if len(video_embeddings) == 0:
        logger.error("No valid samples found!")
        return

    logger.debug(f"Concatenating {len(video_embeddings)} video embeddings...")
    video_embeddings = torch.cat(video_embeddings, dim=0)
    logger.debug(f"Concatenating {len(text_embeddings)} text embeddings...")
    text_embeddings = torch.cat(text_embeddings, dim=0)

    logger.info(f"Total samples in dataset: {len(df)}")
    logger.info(f"Successfully processed: {len(valid_indices)}")
    logger.info(f"Failed/Skipped: {len(df) - len(valid_indices)}")
    logger.info(f"Video embeddings shape: {video_embeddings.shape}")
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    logger.info(f"Adverb negative samples: {len(adverb_negative_examples)}")
    logger.debug(f"Valid indices: {valid_indices}")

    logger.debug(f"Computing similarity matrix...")
    similarity = compute_similarity(video_embeddings, text_embeddings)
    logger.debug(f"Similarity matrix shape: {similarity.shape}")

    logger.info("\n" + "="*60)
    logger.info("STANDARD VIDEO-TO-TEXT RETRIEVAL")
    logger.info("(Each video queries ALL captions in dataset)")
    logger.info("="*60)
    logger.debug(f"Computing V2T recalls...")
    v2t_recalls = compute_recall(similarity)
    logger.debug(f"V2T recalls: {v2t_recalls}")
    for metric, value in v2t_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    log_retrieval_examples(similarity, valid_texts, direction='v2t', num_examples=args.num_examples)

    # Compute per-adverb metrics for V2T
    logger.debug("Computing per-adverb metrics for V2T...")
    per_adverb_v2t = compute_per_adverb_standard_retrieval(
        similarity, valid_adverbs, direction='v2t', k_values=[1, 5, 10]
    )

    logger.info("\n" + "="*60)
    logger.info("STANDARD TEXT-TO-VIDEO RETRIEVAL")
    logger.info("(Each caption queries ALL videos in dataset)")
    logger.info("="*60)
    logger.debug(f"Computing T2V recalls (transposed similarity)...")
    t2v_recalls = compute_recall(similarity.T)
    logger.debug(f"T2V recalls: {t2v_recalls}")
    for metric, value in t2v_recalls.items():
        logger.info(f"  {metric}: {value:.2f}%")

    log_retrieval_examples(similarity, valid_texts, direction='t2v', num_examples=args.num_examples)

    # Compute per-adverb metrics for T2V
    logger.debug("Computing per-adverb metrics for T2V...")
    per_adverb_t2v = compute_per_adverb_standard_retrieval(
        similarity.T, valid_adverbs, direction='t2v', k_values=[1, 5, 10]
    )

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
        logger.info("PER-ADVERB NEGATIVE RETRIEVAL RESULTS (Video-to-Text with Negatives)")
        logger.info("="*60)

        per_adverb_neg = compute_per_adverb_metrics(adverb_negative_examples)
        sorted_adverbs_neg = sorted(per_adverb_neg.items(), key=lambda x: x[1]['R@1'], reverse=True)

        logger.info(f"{'Adverb':<20} | {'R@1':>7} | {'R@3':>7} | {'R@5':>7} | {'MRR':>7} | {'Count':>6} | {'Mean Rank':>10} | {'Median Rank':>12} | {'Avg Cands':>10}")
        logger.info("-" * 120)
        for adv, metrics in sorted_adverbs_neg:
            logger.info(f"{adv:<20} | {metrics['R@1']:6.2f}% | {metrics['R@3']:6.2f}% | {metrics['R@5']:6.2f}% | {metrics['MRR']:6.2f}% | {metrics['count']:6d} | {metrics['mean_rank']:10.2f} | {metrics['median_rank']:12.1f} | {metrics['mean_candidates']:10.1f}")

    # Display per-adverb metrics for V2T
    logger.info("\n" + "="*60)
    logger.info("PER-ADVERB VIDEO-TO-TEXT RETRIEVAL RESULTS")
    logger.info("="*60)
    sorted_adverbs_v2t = sorted(per_adverb_v2t.items(), key=lambda x: x[1]['R@1'], reverse=True)
    logger.info(f"{'Adverb':<20} | {'R@1':>7} | {'R@5':>7} | {'R@10':>7} | {'MRR':>7} | {'Count':>6} | {'Mean Rank':>10} | {'Median Rank':>12}")
    logger.info("-" * 100)
    for adv, metrics in sorted_adverbs_v2t:
        logger.info(f"{adv:<20} | {metrics['R@1']:6.2f}% | {metrics['R@5']:6.2f}% | {metrics.get('R@10', float('nan')):6.2f}% | {metrics['MRR']:6.2f}% | {metrics['count']:6d} | {metrics['mean_rank']:10.2f} | {metrics['median_rank']:12.1f}")

    # Display per-adverb metrics for T2V
    logger.info("\n" + "="*60)
    logger.info("PER-ADVERB TEXT-TO-VIDEO RETRIEVAL RESULTS")
    logger.info("="*60)
    sorted_adverbs_t2v = sorted(per_adverb_t2v.items(), key=lambda x: x[1]['R@1'], reverse=True)
    logger.info(f"{'Adverb':<20} | {'R@1':>7} | {'R@5':>7} | {'R@10':>7} | {'MRR':>7} | {'Count':>6} | {'Mean Rank':>10} | {'Median Rank':>12}")
    logger.info("-" * 100)
    for adv, metrics in sorted_adverbs_t2v:
        logger.info(f"{adv:<20} | {metrics['R@1']:6.2f}% | {metrics['R@5']:6.2f}% | {metrics.get('R@10', float('nan')):6.2f}% | {metrics['MRR']:6.2f}% | {metrics['count']:6d} | {metrics['mean_rank']:10.2f} | {metrics['median_rank']:12.1f}")

    # Save per-adverb metrics to CSV files if output_dir is specified and no_save is False
    if args.output_dir and not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save V2T per-adverb metrics
        v2t_csv_path = os.path.join(args.output_dir, 'per_adverb_v2t.csv')
        with open(v2t_csv_path, 'w', newline='') as f:
            fieldnames = ['adverb', 'R@1', 'R@5', 'R@10', 'MRR', 'count', 'mean_rank', 'median_rank']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for adv, metrics in sorted_adverbs_v2t:
                row = {'adverb': adv}
                row.update(metrics)
                writer.writerow(row)
        logger.info(f"\nPer-adverb V2T metrics saved to: {v2t_csv_path}")

        # Save T2V per-adverb metrics
        t2v_csv_path = os.path.join(args.output_dir, 'per_adverb_t2v.csv')
        with open(t2v_csv_path, 'w', newline='') as f:
            fieldnames = ['adverb', 'R@1', 'R@5', 'R@10', 'MRR', 'count', 'mean_rank', 'median_rank']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for adv, metrics in sorted_adverbs_t2v:
                row = {'adverb': adv}
                row.update(metrics)
                writer.writerow(row)
        logger.info(f"Per-adverb T2V metrics saved to: {t2v_csv_path}")

        # Save negative retrieval per-adverb metrics if available
        if len(adverb_negative_examples) > 0:
            neg_csv_path = os.path.join(args.output_dir, 'per_adverb_negatives.csv')
            with open(neg_csv_path, 'w', newline='') as f:
                fieldnames = ['adverb', 'R@1', 'R@3', 'R@5', 'MRR', 'count', 'mean_rank', 'median_rank', 'mean_candidates']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for adv, metrics in sorted_adverbs_neg:
                    row = {'adverb': adv}
                    row.update(metrics)
                    writer.writerow(row)
            logger.info(f"Per-adverb negative metrics saved to: {neg_csv_path}")

        # Save overall summary metrics
        summary_path = os.path.join(args.output_dir, 'summary_metrics.json')
        summary = {
            'model_name': args.model_name,
            'total_samples': len(valid_indices),
            'v2t': {k: float(v) for k, v in v2t_recalls.items()},
            't2v': {k: float(v) for k, v in t2v_recalls.items()},
            'average': {
                'R@1': float((v2t_recalls['R@1'] + t2v_recalls['R@1']) / 2),
                'R@5': float((v2t_recalls['R@5'] + t2v_recalls['R@5']) / 2),
                'R@10': float((v2t_recalls.get('R@10', float('nan')) + t2v_recalls.get('R@10', float('nan'))) / 2)
            }
        }
        if len(adverb_negative_examples) > 0:
            summary['negatives'] = {
                'samples': len(adverb_negative_examples),
                'R@1': float(np.mean(all_r1)),
                'R@3': float(np.mean(all_r3)),
                'R@5': float(np.mean(all_r5)),
                'mean_rank': float(np.mean(all_ranks))
            }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary metrics saved to: {summary_path}")
    elif args.no_save:
        logger.info("\nFile saving disabled. All results displayed in console only.")

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
    avg_r10 = (v2t_recalls.get('R@10', float('nan')) + t2v_recalls.get('R@10', float('nan'))) / 2

    logger.info(f"\nAverage (V2T + T2V):")
    logger.info(f"  R@1: {avg_r1:.2f}%")
    logger.info(f"  R@5: {avg_r5:.2f}%")
    if not np.isnan(avg_r10):
        logger.info(f"  R@10: {avg_r10:.2f}%")
    else:
        logger.info(f"  R@10: nan%")

    if len(adverb_negative_examples) > 0:
        logger.info(f"\nAdverb Negative V2T Retrieval:")
        logger.info(f"  Samples: {len(adverb_negative_examples)}")
        logger.info(f"  R@1: {np.mean(all_r1):.2f}%")
        logger.info(f"  Mean Rank: {np.mean(all_ranks):.2f}")

if __name__ == '__main__':
    main()
