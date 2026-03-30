# src/workflow_16s/downstream/machine_learning/validation/utils.py

import pandas as pd
import numpy as np
import re
from typing import List, Any, Dict, Optional

def resolve_feature_names(
    adata_agg: Any, 
    level: str
) -> List[str]:
    """
    Extracts high-quality taxonomic names from AnnData var columns.
    Tries Genus/Species columns first, then parses from Tax/Taxaname columns.
    For ASV level, creates short names like: ASV_0_Bacteroides
    """
    names = []
    
    # Try to get from explicit genus column first
    if level.upper() in ['ASV', 'ASVS', 'GENUS']:
        if 'Genus' in adata_agg.var.columns:
            genus_col = adata_agg.var['Genus'].tolist()
            # Only use if not all empty/NaN
            if any(pd.notna(g) and str(g).strip() and str(g).strip() != 'nan' for g in genus_col):
                names = [str(g).strip() if pd.notna(g) and str(g).strip() != 'nan' else None for g in genus_col]
    
    # If genus failed or empty, try parsing from Tax column (full taxonomy)
    if not names or all(n is None for n in names):
        if 'Tax' in adata_agg.var.columns:
            raw_names = adata_agg.var['Tax'].tolist()
        elif 'Taxaname' in adata_agg.var.columns:
            raw_names = adata_agg.var['Taxaname'].tolist()
        else:
            raw_names = adata_agg.var_names.tolist()
        
        names = []
        for name in raw_names:
            if pd.isna(name):
                names.append(None)
                continue
                
            name_str = str(name).strip()
            
            # Split by semicolon for delimited taxonomy strings
            if ';' in name_str:
                parts = [p.strip() for p in name_str.split(';') if p.strip()]
                if len(parts) > 0:
                    # For ASV level, try to find genus (usually 2nd level after kingdom)
                    # Otherwise get deepest assignment
                    if level.upper() in ['ASV', 'ASVS']:
                        # Look for genus-level (g__) or use second-to-last or last
                        genus_part = None
                        for part in parts:
                            if part.startswith('g__'):
                                genus_part = part.replace('g__', '').strip()
                                break
                        if not genus_part and len(parts) >= 2:
                            # Try second to last (genus often there)
                            genus_part = parts[-2].replace('g__', '').replace('f__', '').replace('s__', '').strip()
                        if not genus_part:
                            # Fall back to last
                            genus_part = parts[-1].replace('g__', '').replace('f__', '').replace('s__', '').strip()
                        names.append(genus_part if genus_part else None)
                    else:
                        # For non-ASV, just get deepest
                        deepest = parts[-1].replace('g__', '').replace('f__', '').replace('s__', '').strip()
                        names.append(deepest if deepest else None)
                else:
                    names.append(None)
            else:
                # No semicolons, clean and use as-is
                cleaned = name_str.replace('g__', '').replace('f__', '').replace('s__', '').strip()
                names.append(cleaned if cleaned else None)
    
    # For ASV level, create enumerated names with taxonomy
    if level.upper() in ['ASV', 'ASVS']:
        enriched_names = []
        for asv_idx, taxonomy in enumerate(names):
            if taxonomy and pd.notna(taxonomy) and str(taxonomy).strip() and str(taxonomy).strip() != 'nan':
                enriched_name = f"ASV_{asv_idx}_{taxonomy}"
            else:
                enriched_name = f"ASV_{asv_idx}"
            enriched_names.append(enriched_name)
        return enriched_names
    
    # Filter out None/NaN for other levels
    return [str(n).strip() if n and pd.notna(n) and str(n).strip() != 'nan' else "unknown_feature" for n in names]

def clean_feature_names(df: pd.DataFrame, adata: Optional[Any] = None) -> pd.DataFrame:
    """
    Removes problematic characters from feature names for ML models (CatBoost, LightGBM) and LaTeX.
    Preserves ASV numbers and taxonomy assignments.
    
    If adata is provided and level is ASV, creates richly-named features: ASV_<ID>_<Taxonomy>
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with original column names
    adata : Optional[Any]
        AnnData object with var information. Used to enrich ASV names.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned, deduplicated column names
    """
    df = df.copy()
    
    def _clean_string(col_name: str) -> str:
        """Sanitize individual feature name."""
        name = str(col_name)
        # 1. Replace spaces with underscores
        name = name.replace(' ', '_')
        
        # 2. Remove problematic punctuation: [, ], <, >, ., and ,
        name = re.sub(r'[\[\]<>\.,]', '', name)
        
        # 3. If name looks like "ASV_123_Taxonomy", keep it as-is
        # Otherwise, strip leading digits if they're NOT part of ASV pattern
        if not re.match(r'^ASV_', name):
            name = re.sub(r'^[\d_]+', '', name)
        
        # 4. Fallback if the string becomes completely empty
        return name if name else "unnamed_feature"

    # Apply the cleaning function
    cleaned_cols = [_clean_string(col) for col in df.columns]
    
    # 5. Deduplicate names (e.g. if '1_Bacteroides' and '2_Bacteroides' both become 'Bacteroides')
    seen = {}
    deduplicated_cols = []
    for col in cleaned_cols:
        if col not in seen:
            deduplicated_cols.append(col)
            seen[col] = 1
        else:
            seen[col] += 1
            deduplicated_cols.append(f"{col}_{seen[col]}")
    
    df.columns = deduplicated_cols
    return df

def format_audit_results(train_score: float, test_score: float, target: str) -> Dict[str, Any]:
    """Calculates the scientific 'Generalization Gap' for forensic records."""
    gap = train_score - test_score
    return {
        "target": target,
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "overfitting_gap": round(gap, 4),
        "status": "PASS" if gap < 0.15 else "FAIL"
    }