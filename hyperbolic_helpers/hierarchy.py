

import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools


class Hierarchy:

    def __init__(self, action_adverb_path: str):
        self.action_adverb_path = action_adverb_path

        # Load hierarchies
        self.action_to_parent = {}  # action_name → parent_action_name
        self.adverb_to_parent = {}  # adverb_name → parent_adverb_name
        self.parent_to_actions = defaultdict(list)  # parent_action_name → [actions]
        self.parent_to_adverbs = defaultdict(list)  # parent_adverb_name → [adverbs]

        self._load_hierarchies()

    def _load_hierarchies(self):
        with open(self.action_adverb_path, 'r') as f:
            action_hierarchy = json.load(f)

        for parent, children in action_hierarchy['action_hierarchy'].items():
            self.parent_to_actions[parent] = children
            for child in children:
                self.action_to_parent[child] = parent

        for parent, children in action_hierarchy['adverb_hierarchy'].items():
            self.parent_to_adverbs[parent] = children
            for child in children:
                self.adverb_to_parent[child] = parent    

    def get_action_parent(self, action: str) -> str:
        return self.action_to_parent.get(action, None)

    def get_adverb_parent(self, adverb: str) -> str:
        return self.adverb_to_parent.get(adverb, None)

    def get_parent_actions(self, parent: str) -> List[str]:
        return self.parent_to_actions.get(parent, [])

    def get_parent_adverbs(self, parent: str) -> List[str]:
        return self.parent_to_adverbs.get(parent, [])

    def get_all_parent_actions(self) -> List[str]:
        return list(self.parent_to_actions.keys())

    def get_all_parent_adverbs(self) -> List[str]:
        return list(self.parent_to_adverbs.keys())

    def get_all_actions(self) -> List[str]:
        """Get all child actions (primitives with parents)."""
        return list(self.action_to_parent.keys())

    def get_all_adverbs(self) -> List[str]:
        """Get all child adverbs (primitives with parents)."""
        return list(self.adverb_to_parent.keys())

    def get_all_actions_with_parents(self) -> Dict[str, str]:
        """Get all actions mapped to their parent categories."""
        return dict(self.action_to_parent)

    def get_all_adverbs_with_parents(self) -> Dict[str, str]:
        """Get all adverbs mapped to their parent categories."""
        return dict(self.adverb_to_parent)

    def get_semantic_hierarchy_pairs(
        self,
        action_list: List[str],
        adverb_list: List[str]
    ) -> List[Tuple[str, str]]:
        pairs = []

        # Action hierarchy pairs
        for action in action_list:
            parent = self.get_action_parent(action)
            if parent:
                pairs.append((parent, action))

        # Adverb hierarchy pairs
        for adverb in adverb_list:
            parent = self.get_adverb_parent(adverb)
            if parent:
                pairs.append((parent, adverb))

        return pairs

    def get_conceptual_hierarchy_pairs(
        self,
        compositions: List[Tuple[str, str]]
    ) -> List[Tuple[Tuple[str, str], str]]:
        pairs = []

        for adverb, action in compositions:
            # Composition ⊂ adverb
            pairs.append(((adverb, action), adverb))
            # Composition ⊂ action
            pairs.append(((adverb, action), action))

        return pairs

    def get_hard_negatives(
        self,
        composition: Tuple[str, str],
        all_compositions: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        adverb, action = composition
        hard_negatives = []

        for comp_adv, comp_act in all_compositions:
            # Skip the query composition itself
            if comp_adv == adverb and comp_act == action:
                continue

            # Include if shares same action OR same adverb (but not both)
            if comp_adv == adverb or comp_act == action:
                hard_negatives.append((comp_adv, comp_act))

        return hard_negatives

    def get_easy_negatives(
        self,
        composition: Tuple[str, str],
        all_compositions: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Get easy negative compositions (share no primitives).

        Args:
            composition: Query composition (adverb, action) tuple
            all_compositions: List of all possible compositions

        Returns:
            List of easy negative compositions
        """
        adverb, action = composition
        easy_negatives = []

        for comp_adv, comp_act in all_compositions:
            # Skip the query composition itself
            if comp_adv == adverb and comp_act == action:
                continue

            # Include if shares neither action nor adverb
            if comp_adv != adverb and comp_act != action:
                easy_negatives.append((comp_adv, comp_act))

        return easy_negatives

    def validate_hierarchy(
        self,
        action_list: List[str],
        adverb_list: List[str]
    ) -> Dict[str, any]:
        """
        Validate that all primitives have parent categories.

        Args:
            action_list: List of all actions
            adverb_list: List of all adverbs

        Returns:
            Dictionary with validation statistics
        """
        # Check actions
        actions_with_parent = [a for a in action_list if self.get_action_parent(a)]
        actions_without_parent = [a for a in action_list if not self.get_action_parent(a)]

        # Check adverbs
        adverbs_with_parent = [a for a in adverb_list if self.get_adverb_parent(a)]
        adverbs_without_parent = [a for a in adverb_list if not self.get_adverb_parent(a)]

        stats = {
            'total_actions': len(action_list),
            'actions_with_parent': len(actions_with_parent),
            'actions_without_parent': len(actions_without_parent),
            'actions_missing': actions_without_parent,
            'total_adverbs': len(adverb_list),
            'adverbs_with_parent': len(adverbs_with_parent),
            'adverbs_without_parent': len(adverbs_without_parent),
            'adverbs_missing': adverbs_without_parent,
            'parent_action_categories': len(self.get_all_parent_actions()),
            'parent_adverb_categories': len(self.get_all_parent_adverbs()),
        }

        return stats

    def __repr__(self):
        return (
            f"Hierarchy(\n"
            f"  Action categories: {len(self.get_all_parent_actions())}\n"
            f"  Adverb categories: {len(self.get_all_parent_adverbs())}\n"
            f"  Total actions: {len(self.action_to_parent)}\n"
            f"  Total adverbs: {len(self.adverb_to_parent)}\n"
            f")"
        )



if __name__ == "__main__":
    # Test hierarchy functionality
    print("Testing Hierarchy Module")
    print("=" * 50)
    action_adverb_path = "datasets/VATEX_Adverbs/action_adverb_hierarchy.json"

    # Load hierarchy
    hierarchy = Hierarchy(action_adverb_path)
    print(f"\n{hierarchy}")

    # Test parent lookup
    print("\nParent lookups:")
    print(f"  'run' → '{hierarchy.get_action_parent('run')}'")
    print(f"  'quickly' → '{hierarchy.get_adverb_parent('quickly')}'")

    # Test hierarchy pairs
    all_actions = ['put', 'lower', 'look', 'smile', 'climb', 'mount', 'move', 'carry',
       'take', 'walk', 'tilt', 'fall', 'stand', 'talk', 'run', 'pull',
       'turn', 'bend', 'fold', 'hold', 'adjust', 'jump', 'peel', 'shake',
       'hit', 'sit', 'play', 'dance', 'stick', 'open', 'eat', 'laugh',
       'drive', 'clap', 'spread', 'knead', 'throw', 'use', 'scoop',
       'measure', 'breathe', 'raise', 'shoot', 'nod', 'cry', 'lunge',
       'arrange', 'make', 'wash', 'set', 'install', 'attach', 'separate',
       'press', 'fill', 'land', 'cook', 'cover', 'extend', 'touch',
       'step', 'mix', 'pour', 'juggle', 'fly', 'ride', 'sprinkle',
       'spray', 'swing', 'gesture', 'squat', 'sing', 'follow', 'stretch',
       'roll', 'point', 'dip', 'ski', 'pedal', 'secure', 'dry',
       'decorate', 'stop', 'swim', 'surf', 'twist', 'weave', 'stack',
       'feed', 'cut', 'insert', 'release', 'start', 'brush', 'kick',
       'close', 'paddle', 'scrape', 'draw', 'sail', 'count', 'tuck',
       'flail', 'drain', 'inflate', 'pass', 'crawl', 'hug', 'burn',
       'smoke', 'drink', 'enter', 'approach', 'fight', 'wax', 'stab',
       'skate', 'peck', 'add', 'grind', 'pump', 'kiss', 'cool', 'exit',
       'mark', 'skateboard', 'click', 'strum', 'sound', 'sharpen',
       'blink', 'drill', 'sneeze', 'park', 'tickle']
    all_adverbs = ['indoor', 'quickly', 'downwards', 'upwards', 'carefully', 'slowly',
       'outdoor', 'gently', 'properly', 'backwards', 'partially', 'off',
       'quietly', 'loudly', 'out', 'forwards', 'on', 'continuously',
       'completely', 'in', 'periodically', 'firmly', 'carelessly',
       'horizontally', 'evenly', 'instantly', 'vertically', 'unevenly',
       'accidently', 'neatly', 'improperly', 'gradually', 'purposefully',
       'messily']
    all_compositions = itertools.product(all_adverbs, all_actions)
    all_compositions = list(all_compositions)

    semantic_pairs = hierarchy.get_semantic_hierarchy_pairs(all_actions, all_adverbs)
    print(f"\nSemantic hierarchy pairs: {len(semantic_pairs)}")
    for parent, child in semantic_pairs:
        print(f"  {child} ⊂ {parent}")

    # Test hard negatives
    query_comp = ("quickly", "run")
    hard_negs = hierarchy.get_hard_negatives(query_comp, all_compositions)
    print(f"\nHard negatives for {query_comp}:")
    for neg in hard_negs:
        print(f"  {neg}")

    # Validate
    stats = hierarchy.validate_hierarchy(all_actions, all_adverbs)
    print(f"\nHierarchy validation:")
    print(f"  Actions with parents: {stats['actions_with_parent']}/{stats['total_actions']}")
    print(f"  Adverbs with parents: {stats['adverbs_with_parent']}/{stats['total_adverbs']}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
