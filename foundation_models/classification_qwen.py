import os
import sys
import math
import hashlib
import argparse
import requests
from qwen_vl_utils import process_vision_info
import numpy as np
import pandas as pd
from PIL import Image
import decord
from decord import VideoReader, cpu
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperbolic_helpers.hierarchy import Hierarchy



def load_or_download_model(save_directory="/ivi/zfs/s0/original_homes/gmago/models/Qwen/Qwen3-VL-8B-Instruct"):
    """
    Load model from local directory if exists, otherwise download and save.
    
    Args:
        save_directory: Path to save/load the model
        
    Returns:
        model, processor
    """
    if os.path.exists(save_directory):
        print(f"Loading model from {save_directory}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            save_directory, 
            dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(save_directory)
    else:
        print(f"Downloading model and saving to {save_directory}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", 
            dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        
        # Create directory and save
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")
    
    return model, processor


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def inference(video, prompt, model, processor, max_new_tokens=2048, total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32, max_frames= 2048, sample_fps = 2):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url). 
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.

    Returns:
        str: Generated text response from the model.

    Notes:
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """

    messages = [
        {"role": "user", "content": [
                {"video": video,
                 "type": "video",
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels, 
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]



def parse_action_adverb_output(output: str) -> Optional[Dict[str, str]]:
    """
    Parse the JSON output from Qwen VL action and adverb classification.
    
    Args:
        output: String output from Qwen VL model (expected to be JSON)
        
    Returns:
        Dictionary containing action, action_category, adverb, adverb_category
        Returns None if parsing fails
        
    Example:
        >>> output = '{"action": "run", "action_category": "locomotion", "adverb": "quickly", "adverb_category": "speed"}'
        >>> result = parse_action_adverb_output(output)
        >>> print(result)
        {'action': 'run', 'action_category': 'locomotion', 'adverb': 'quickly', 'adverb_category': 'speed'}
    """
    try:
        # Remove any markdown code block formatting if present
        cleaned_output = output.strip()
        if cleaned_output.startswith('```json'):
            cleaned_output = cleaned_output.replace('```json', '').replace('```', '').strip()
        elif cleaned_output.startswith('```'):
            cleaned_output = cleaned_output.replace('```', '').strip()
        elif cleaned_output.lower().startswith('json'):                                            
            cleaned_output = cleaned_output[4:].strip()
        
        # Parse JSON
        parsed = json.loads(cleaned_output)
        
        # Validate required fields
        required_fields = ['action', 'action_category', 'adverb', 'adverb_category']
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")
        
        return {
            'action': parsed['action'],
            'action_category': parsed['action_category'],
            'adverb': parsed['adverb'],
            'adverb_category': parsed['adverb_category']
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except ValueError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def validate_classification(parsed_output: Dict[str, str], 
                           action_hierarchy: Dict[str, list],
                           adverb_hierarchy: Dict[str, list]) -> Tuple[bool, list]:
    """
    Validate that the parsed output contains valid actions and adverbs.
    
    Args:
        parsed_output: Dictionary from parse_action_adverb_output
        action_hierarchy: Dictionary of action categories and their actions
        adverb_hierarchy: Dictionary of adverb categories and their adverbs
        
    Returns:
        Tuple of (is_valid: bool, errors: list of error messages)
    """
    errors = []
    
    if not parsed_output:
        return False, ["Parsed output is None or empty"]
    
    # Validate action
    action_category = parsed_output.get('action_category')
    action = parsed_output.get('action')
    
    if action_category not in action_hierarchy:
        errors.append(f"Invalid action_category: {action_category}")
    elif action not in action_hierarchy[action_category]:
        errors.append(f"Action '{action}' not found in category '{action_category}'")
    
    # Validate adverb
    adverb_category = parsed_output.get('adverb_category')
    adverb = parsed_output.get('adverb')
    
    if adverb_category not in adverb_hierarchy:
        errors.append(f"Invalid adverb_category: {adverb_category}")
    elif adverb not in adverb_hierarchy[adverb_category]:
        errors.append(f"Adverb '{adverb}' not found in category '{adverb_category}'")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL action/adverb classification")
    parser.add_argument("--csv_path", type=str, help="Path to annotations CSV", default='/home/gmago/AA/PseudoAdverbs/datasets/VATEX_Adverbs/annotations.csv')
    parser.add_argument('--video_dir', type=str, default='/ivi/zfs/s0/original_homes/gmago/AA/VaTeX/videos/vatex-dataset', help='Directory containing the video files')
    parser.add_argument('--hierarchy_json_path', type=str, default='datasets/VATEX_Adverbs/action_adverb_hierarchy.json')
    parser.add_argument("--system_prompt_file", type=str, help="Path to text file containing the system prompt", default='foundation_models/prompt.txt')
    parser.add_argument("--output_path", type=str, default="results_qwen.csv", help="Output CSV path")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Qwen model name or path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process (for testing)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries on parse/hierarchy failure")
    parser.add_argument("--num_frames", type=int, default=64, help="Number of frames to sample per video")
    args = parser.parse_args()

    # Load system prompt
    with open(args.system_prompt_file, "r") as f:
        system_prompt = f.read()

    # Load hierarchy
    hierarchy = Hierarchy(args.hierarchy_json_path)
    action_hierarchy = dict(hierarchy.parent_to_actions)
    adverb_hierarchy = dict(hierarchy.parent_to_adverbs)

    # Load CSV
    df = pd.read_csv(args.csv_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    print(f"Processing {len(df)} samples")

    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor = load_or_download_model()

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        clip_id = row["clip_id"]
        video_path = os.path.join(args.video_dir, f"{clip_id}.mp4")
        gt_action = row.get("clustered_action", row.get("action", None))
        gt_adverb = row.get("clustered_adverb", row.get("adverb", None))

        if not os.path.exists(video_path):
            print(f"[WARN] Video not found, skipping: {video_path}", file=sys.stderr)
            results.append({
                "clip_id": clip_id,
                "predicted_action": None,
                "predicted_action_category": None,
                "predicted_adverb": None,
                "predicted_adverb_category": None,
                "gt_action": gt_action,
                "gt_adverb": gt_adverb,
                "success": False,
            })
            continue

        parsed = None
        prompt = system_prompt
        for attempt in range(args.max_retries):
            try:
                with torch.no_grad():
                    output = inference(video_path, prompt, model, processor, max_new_tokens=512)
                print(f"[DEBUG] clip={clip_id} attempt={attempt+1}: model output: {output}")
                if attempt > 0:
                    print(f"[DEBUG] prompt: {prompt[:]}")
            except Exception as e:
                print(
                    f"[WARN] clip={clip_id} attempt={attempt+1}: inference error ({e}), retrying",
                    file=sys.stderr,
                )
                continue
            parsed = parse_action_adverb_output(output)
            if parsed is None:
                print(
                    f"[WARN] clip={clip_id} attempt={attempt+1}: parse failed, output was: {output}, retrying",
                    file=sys.stderr,
                )
                prompt = (
                    system_prompt
                    + "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
                    "Respond with ONLY a valid JSON object and nothing else. "
                    'Example: {"action": "run", "action_category": "locomotion", '
                    '"adverb": "quickly", "adverb_category": "speed"}'
                )
                continue
            is_valid, errors = validate_classification(parsed, action_hierarchy, adverb_hierarchy)
            if not is_valid:
                print(
                    f"[WARN] clip={clip_id} attempt={attempt+1}: hierarchy validation failed "
                    f"({errors}), retrying",
                    file=sys.stderr,
                )
                all_actions_flat = [a for acts in action_hierarchy.values() for a in acts]
                all_adverbs_flat = [a for advs in adverb_hierarchy.values() for a in advs]
                prompt = (
                    system_prompt
                    + f"\n\nIMPORTANT: Your previous answer was invalid — {errors}. "
                    f"You MUST choose the action from this exact list: {all_actions_flat}. "
                    f"You MUST choose the adverb from this exact list: {all_adverbs_flat}."
                )
                parsed = None
                continue
            break  # success

        if parsed is not None:
            results.append({
                "clip_id": clip_id,
                "predicted_action": parsed["action"],
                "predicted_action_category": parsed["action_category"],
                "predicted_adverb": parsed["adverb"],
                "predicted_adverb_category": parsed["adverb_category"],
                "gt_action": gt_action,
                "gt_adverb": gt_adverb,
                "success": True,
            })
        else:
            print(
                f"[ERROR] clip={clip_id}: failed after {args.max_retries} retries",
                file=sys.stderr,
            )
            results.append({
                "clip_id": clip_id,
                "predicted_action": None,
                "predicted_action_category": None,
                "predicted_adverb": None,
                "predicted_adverb_category": None,
                "gt_action": gt_action,
                "gt_adverb": gt_adverb,
                "success": False,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")
    success_count = results_df["success"].sum()
    print(f"Successful: {success_count}/{len(results_df)}")


