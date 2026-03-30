# src/workflow_16s/downstream/machine_learning/visualization/taxa_labels.py
from typing import List
import re

def extract_asv_id(feature_name: str) -> str:
    """Extracts ASV ID from feature name (e.g., 'ASV_2996' from full name)."""
    match = re.search(r'ASV_\d+', feature_name)
    return match.group(0) if match else feature_name[:10]

def simplify_feature_name(taxon: str, max_length: int = 25) -> str:
    """Aggressively simplifies taxon string for plot readability.
    
    Priority:
    1. If has genus (e.g., 'g__Bacillus'), use ASV + genus only
    2. If unclassified, use ASV + family
    3. Default: ASV + first specific level
    """
    # Extract ASV ID
    asv_id = extract_asv_id(taxon)
    
    # Parse taxonomic levels
    parts = [p.strip() for p in taxon.split(";")]
    
    # Find genus level (g__)
    genus = None
    family = None
    
    for part in parts:
        if part.startswith('g__'):
            genus = part.replace('g__', '').replace('_', ' ')
            break
        elif part.startswith('f__'):
            family = part.replace('f__', '').replace('_', ' ')
    
    # Build label
    if genus and 'Unclassified' not in genus:
        label = f"{asv_id} {genus}"
    elif family and 'Unclassified' not in family:
        label = f"{asv_id} {family}"
    else:
        # Fallback: use last non-unclassified part
        for part in reversed(parts):
            clean = part.replace('__', ' ').replace('_', ' ').strip()
            if clean and 'unclassified' not in clean.lower() and 'uncultured' not in clean.lower():
                label = f"{asv_id} {clean}"
                break
        else:
            label = asv_id
    
    # Enforce max length with ellipsis if needed
    if len(label) > max_length:
        label = label[:max_length-1] + '…'
    
    return label

def generate_unique_simplified_labels(feature_names: List[str]) -> List[str]:
    """Generates simplified, unique labels for a list of feature names (taxa)."""
    simplified_labels = []
    used_labels = set()
    
    for f in feature_names:
        label = simplify_feature_name(f)
        base_label = label
        suffix = 1
        
        # Handle duplicates
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        
        simplified_labels.append(label)
        used_labels.add(label)
    
    return simplified_labels