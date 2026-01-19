import json
import csv
import re
import random
import torch
from torch.utils.data import Dataset


def load_antonym_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    antonym_map = {}
    for pair in data['adverb_antonym_pairs']:
        adverb = pair['adverb']
        antonym_map[adverb] = pair['replacements']
    return antonym_map


def replace_adverb_in_caption(caption, adverb, replacements, k=1):
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

    return results


class NegativeCaptionDataset(Dataset):
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', default='datasets/VATEX_Adverbs/annotations.csv')
    parser.add_argument('--hierarchy', default='datasets/VATEX_Adverbs/action_adverb_hierarchy.json')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    dataset = NegativeCaptionDataset(args.annotations, args.hierarchy, k=args.k)

    samples_by_adverb = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        adv = sample['clustered_adverb']
        if adv not in samples_by_adverb:
            samples_by_adverb[adv] = sample

    print(f"Dataset size: {len(dataset)}")
    print(f"Unique adverbs: {len(samples_by_adverb)}")
    print(f"Generating {args.k} negative captions per sample")
    print("=" * 80)

    for adv in sorted(samples_by_adverb.keys()):
        sample = samples_by_adverb[adv]
        print(f"\nAdverb: {adv}")
        print(f"  Original:  {sample['caption']}")
        for j, neg in enumerate(sample['negative_captions']):
            print(f"  Negative {j+1}: {neg}")
