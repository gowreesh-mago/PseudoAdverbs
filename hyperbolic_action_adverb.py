import os
import argparse
import time
from tqdm import tqdm
import torch.nn as nn
import hypll.nn as hnn
import torch.optim as optim
import hypll.optim as hoptim
import torch
from torch.utils.tensorboard import SummaryWriter

def create_hierarchy_mappings(hierarchy, dataset_info, device):
    """
    Create hierarchy mapping tensors for losses.

    Args:
        hierarchy: Hierarchy object
        dataset_info: Dataset information dictionary
        device: torch device

    Returns:
        Dictionary of hierarchy mappings
    """
    # Get list of actions and adverbs
    action_list = list(dataset_info['action_to_idx'].keys())
    adverb_list = list(dataset_info['adverb_to_idx'].keys())

    # Build parent category indices
    parent_action_names = hierarchy.get_all_parent_actions()
    parent_adverb_names = hierarchy.get_all_parent_adverbs()

    parent_action_to_idx = {name: idx for idx, name in enumerate(parent_action_names)}
    parent_adverb_to_idx = {name: idx for idx, name in enumerate(parent_adverb_names)}

    # Build mappings for semantic hierarchy (primitives → parent categories)
    action_to_parent_action = []
    adverb_to_parent_adverb = []

    # Actions → Parent actions
    for action in action_list:
        parent = hierarchy.get_action_parent(action)
        if parent and parent in parent_action_to_idx:
            action_idx = dataset_info['action_to_idx'][action]
            parent_idx = parent_action_to_idx[parent]
            action_to_parent_action.append([action_idx, parent_idx])

    # Adverbs → Parent adverbs
    for adverb in adverb_list:
        parent = hierarchy.get_adverb_parent(adverb)
        if parent and parent in parent_adverb_to_idx:
            adverb_idx = dataset_info['adverb_to_idx'][adverb]
            parent_idx = parent_adverb_to_idx[parent]
            adverb_to_parent_adverb.append([adverb_idx, parent_idx])

    # Convert to tensors [child_idx, parent_idx]
    action_to_parent_action = torch.tensor(
        action_to_parent_action,
        dtype=torch.long,
        device=device
    ) if len(action_to_parent_action) > 0 else torch.empty(0, 2, dtype=torch.long, device=device)

    adverb_to_parent_adverb = torch.tensor(
        adverb_to_parent_adverb,
        dtype=torch.long,
        device=device
    ) if len(adverb_to_parent_adverb) > 0 else torch.empty(0, 2, dtype=torch.long, device=device)

    # Create composition to primitive mappings (conceptual hierarchy)
    # For each composition, map to its action and adverb
    composition_to_action = []
    composition_to_adverb = []

    for comp_str, comp_idx in dataset_info['composition_to_idx'].items():
        # Parse composition (format: "adverb, action")
        parts = comp_str.split(', ')
        if len(parts) == 2:
            adverb_name, action_name = parts
            if action_name in dataset_info['action_to_idx']:
                action_idx = dataset_info['action_to_idx'][action_name]
                composition_to_action.append([comp_idx, action_idx])
            if adverb_name in dataset_info['adverb_to_idx']:
                adverb_idx = dataset_info['adverb_to_idx'][adverb_name]
                composition_to_adverb.append([comp_idx, adverb_idx])

    composition_to_action = torch.tensor(
        composition_to_action,
        dtype=torch.long,
        device=device
    ) if len(composition_to_action) > 0 else torch.empty(0, 2, dtype=torch.long, device=device)

    composition_to_adverb = torch.tensor(
        composition_to_adverb,
        dtype=torch.long,
        device=device
    ) if len(composition_to_adverb) > 0 else torch.empty(0, 2, dtype=torch.long, device=device)

    return {
        'action_to_parent_action': action_to_parent_action,
        'adverb_to_parent_adverb': adverb_to_parent_adverb,
        'composition_to_action': composition_to_action,
        'composition_to_adverb': composition_to_adverb
    }


class Trainer:
    def __init__(self,):
        pass
    def train(self,):
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for idx, batch in enumerate(pbar):
            video_features = batch['video_features'].to(self.device)
            action_idx = batch['action_idx'].to(self.device)
            adverb_idx = batch['adverb_idx'].to(self.device)



if __name__ == "__main__":
    create_hierarchy_mappings()