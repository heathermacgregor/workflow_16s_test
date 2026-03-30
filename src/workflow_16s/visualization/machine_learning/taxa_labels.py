# workflow_16s/visualization/machine_learning/taxa_labels.py
from typing import List

def simplify_feature_name(taxon: str) -> str:
    """Simplifies a taxon string to its most specific informative label."""
    parts = taxon.split(";")
    last = parts[-1].strip().lower()
    if last in {"__unclassified", "__uncultured", "__"}:
        return ";".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return parts[-1]

def generate_unique_simplified_labels(feature_names: List[str]) -> List[str]:
    """Generates simplified, unique labels for a list of feature names (taxa)."""
    simplified_labels = []; used_labels = set()
    for f in feature_names:
        label = simplify_feature_name(f); base_label = label; suffix = 1
        while label in used_labels:
            label = f"{base_label}_{suffix}"; suffix += 1
        simplified_labels.append(label); used_labels.add(label)
    return simplified_labels