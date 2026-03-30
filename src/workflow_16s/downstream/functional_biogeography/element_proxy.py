"""
Element Composition Proxies for Metal Enrichment Inference

Uses SoilGrids soil element composition to infer potential for metal
enrichment. Maps soil chemistry patterns to likely heavy metal presence.

Key insight: Certain soil compositions indicate metal-bearing parent material:
- High clay + low sand = stronger weathering, metal preservation
- Organic carbon = chelation/bioavailability indicator  
- pH patterns = metal solubility patterns
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ElementProxyScore:
    """Score for a single heavy metal based on element proxies"""
    metal: str
    proxy_score: float  # 0-1 confidence
    component_scores: Dict[str, float]  # Contributing factors
    soil_composition: Dict[str, float]  # Measured soil elements
    interpretation: str


class SoilElementProxyMapping:
    """
    Maps SoilGrids element composition to heavy metal enrichment proxies.
    
    Integrates knowledge about:
    - Weathering rates (clay content = more chemical weathering)
    - Metal preservation in soil profile (pH, organic carbon)
    - Mineral composition indicators (sand/silt/clay ratios)
    - Water holding capacity (relates to bioavailability)
    """
    
    # SoilGrids columns expected in adata.obs after enrichment
    SOILGRIDS_COMPOSITION = {
        'SoilGrids_clay': 'Clay content (percent)',
        'SoilGrids_silt': 'Silt content (percent)',
        'SoilGrids_sand': 'Sand content (percent)',
        'SoilGrids_soc': 'Soil organic carbon (dg/kg)',
        'SoilGrids_nitrogen': 'Total nitrogen (cg/kg)',
        'SoilGrids_phh2o': 'Soil pH in water',
        'SoilGrids_cec': 'Cation exchange capacity (cmol+/kg)',
        'SoilGrids_bdod': 'Bulk density fine earth (cg/cm³)',
    }
    
    # Metal proxy formulas based on soil composition
    METAL_PROXIES = {
        'uranium': {
            'description': 'Uranium accumulation proxy',
            'indicators': {
                'clay_content': 0.40,      # Clay preserves uranium
                'acidic_ph': 0.25,         # Acidic dissolves uranium
                'low_organic_carbon': 0.20, # Low OC = less chelation
                'high_cec': 0.15           # High CEC = retention
            },
            'optimal_conditions': "Moderately acidic, clayey soils with low organic matter"
        },
        'arsenic': {
            'description': 'Arsenic enrichment proxy',
            'indicators': {
                'clay_content': 0.30,
                'acidic_ph': 0.20,
                'moderate_organic_carbon': 0.30,  # Organic matter can mobilize arsenic
                'high_cec': 0.20
            },
            'optimal_conditions': "Acidic to neutral, fine-textured soils with organic matter"
        },
        'copper': {
            'description': 'Copper enrichment proxy',
            'indicators': {
                'clay_content': 0.35,
                'neutral_to_alkaline_ph': 0.25,
                'high_organic_carbon': 0.25,  # Copper binds to organics
                'high_cec': 0.15
            },
            'optimal_conditions': "Neutral pH, clayey soils with organic matter"
        },
        'lead': {
            'description': 'Lead accumulation proxy',
            'indicators': {
                'clay_content': 0.35,
                'neutral_to_alkaline_ph': 0.20,
                'high_organic_carbon': 0.30,  # Lead strongly sorbs to organics
                'high_cec': 0.15
            },
            'optimal_conditions': "Neutral to alkaline, fine-textured soils with high OM"
        },
        'zinc': {
            'description': 'Zinc enrichment proxy',
            'indicators': {
                'clay_content': 0.30,
                'neutral_to_alkaline_ph': 0.25,
                'high_organic_carbon': 0.30,
                'high_cec': 0.15
            },
            'optimal_conditions': "Neutral pH, clayey soils with organic matter"
        },
        'cadmium': {
            'description': 'Cadmium mobilization proxy',
            'indicators': {
                'acidic_ph': 0.35,           # Cadmium more mobile in acidic
                'high_organic_carbon': 0.30,
                'low_cec': 0.20,             # Lower retention
                'sandy_loam': 0.15
            },
            'optimal_conditions': "Acidic, sandy soils with organic matter"
        },
        'nickel': {
            'description': 'Nickel accumulation proxy',
            'indicators': {
                'clay_content': 0.30,
                'moderate_ph': 0.25,
                'low_organic_carbon': 0.25,
                'neutral_ph': 0.20
            },
            'optimal_conditions': "Fine-textured soils with neutral pH"
        },
        'rare_earth': {
            'description': 'Rare earth element accumulation proxy',
            'indicators': {
                'clay_content': 0.40,
                'acidic_ph': 0.30,
                'high_organic_carbon': 0.20,
                'high_cec': 0.10
            },
            'optimal_conditions': "Acidic, clayey soils (REE preserved in weathering resistant minerals)"
        },
    }
    
    @classmethod
    def calculate_proxy_score(
        cls,
        soil_data: Dict[str, float],
        metal: str
    ) -> ElementProxyScore:
        """
        Calculate metal enrichment proxy score from soil composition.
        
        Args:
            soil_data: Dict with keys like 'clay', 'soc', 'phh2o', etc.
            metal: Metal name from METAL_PROXIES.keys()
        
        Returns:
            ElementProxyScore with normalized 0-1 confidence
        """
        if metal not in cls.METAL_PROXIES:
            raise ValueError(f"Unknown metal: {metal}")
        
        proxy_def = cls.METAL_PROXIES[metal]
        component_scores = {}
        
        # Evaluate each indicator
        for indicator, weight in proxy_def['indicators'].items():
            score = cls._evaluate_indicator(
                indicator,
                soil_data,
                metal
            )
            component_scores[indicator] = score * weight
        
        # Weighted average
        total_weight = sum(proxy_def['indicators'].values())
        proxy_score = sum(component_scores.values()) / total_weight if total_weight > 0 else 0.5
        
        # Clamp to 0-1
        proxy_score = max(0, min(1, proxy_score))
        
        return ElementProxyScore(
            metal=metal,
            proxy_score=proxy_score,
            component_scores=component_scores,
            soil_composition=soil_data,
            interpretation=proxy_def['optimal_conditions']
        )
    
    @classmethod
    def _evaluate_indicator(
        cls,
        indicator: str,
        soil_data: Dict[str, float],
        metal: str
    ) -> float:
        """
        Evaluate how well a soil indicator matches metal enrichment conditions.
        
        Returns:
            Score 0-1 where 1 = optimal condition for metal
        """
        # Extract values from soil_data
        clay = soil_data.get('clay', 30)      # Default 30%
        ph = soil_data.get('phh2o', 6.5)
        soc = soil_data.get('soc', 50)        # Default 50 dg/kg (~5%)
        cec = soil_data.get('cec', 150)       # Default 150 cmol+/kg
        sand = soil_data.get('sand', 40)
        
        # Evaluate specific indicators
        if indicator == 'clay_content':
            # Higher clay is generally better for metal retention (except very high)
            if clay < 10:
                return 0.3
            elif clay < 25:
                return 0.6
            elif clay < 40:
                return 0.9
            else:
                return 0.8  # Very high clay may be less optimal
        
        elif indicator == 'acidic_ph':
            # Optimal pH 4.5-6.5 for most heavy metals
            if ph < 4.5:
                return 0.8
            elif ph < 6.5:
                return 0.9
            elif ph < 7.5:
                return 0.6
            else:
                return 0.3
        
        elif indicator == 'neutral_to_alkaline_ph':
            # Neutral to slightly alkaline optimal
            if ph < 6.5:
                return 0.4
            elif ph < 7.5:
                return 0.95
            elif ph < 8.5:
                return 0.8
            else:
                return 0.5
        
        elif indicator == 'moderate_ph':
            # pH 6.5-7.5 optimal
            if ph < 6:
                return 0.6
            elif ph < 7.5:
                return 0.95
            elif ph < 8:
                return 0.7
            else:
                return 0.4
        
        elif indicator == 'neutral_ph':
            # Exactly neutral (pH ~7)
            pH_diff = abs(ph - 7.0)
            return max(0, 1 - (pH_diff / 3))  # Decline 0.33 per pH unit
        
        elif indicator == 'high_organic_carbon':
            # High OC favors metal binding (>75 dg/kg = >7.5%)
            if soc < 30:
                return 0.3
            elif soc < 60:
                return 0.6
            elif soc < 100:
                return 0.9
            else:
                return 0.95
        
        elif indicator == 'moderate_organic_carbon':
            # Moderate OC 50-100 dg/kg
            if soc < 40:
                return 0.5
            elif soc < 100:
                return 0.95
            else:
                return 0.7
        
        elif indicator == 'low_organic_carbon':
            # Low OC <40 dg/kg
            if soc < 30:
                return 0.95
            elif soc < 60:
                return 0.7
            else:
                return 0.3
        
        elif indicator == 'high_cec':
            # High CEC (>200 cmol+/kg) indicates more sorption capacity
            if cec < 80:
                return 0.3
            elif cec < 150:
                return 0.6
            elif cec < 250:
                return 0.9
            else:
                return 0.95
        
        elif indicator == 'low_cec':
            # Low CEC (<80 cmol+/kg)
            if cec < 80:
                return 0.95
            elif cec < 150:
                return 0.6
            else:
                return 0.2
        
        elif indicator == 'sandy_loam':
            # Sand % 50-70%, silt+clay 30-50%
            silt_clay = 100 - sand
            if 50 <= sand <= 70 and 30 <= silt_clay <= 50:
                return 0.9
            elif 40 <= sand <= 80:
                return 0.7
            else:
                return 0.4
        
        else:
            return 0.5  # Unknown indicator


class MetalProxyAnalyzer:
    """
    Orchestrates metal proxy calculation across samples.
    
    Produces:
    - Per-sample metal enrichment scores (0-1)
    - Correlation between traits and proxy scores
    - Interpretation of metal selection pressure
    """
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def calculate_sample_proxies(
        self,
        adata: Any,  # AnnData object
        metals: Optional[List[str]] = None,
        gee_soil_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Calculate metal proxies for all samples.
        
        Args:
            adata: AnnData object with SoilGrids metadata in .obs
            metals: List of metals to calculate (default: all)
            gee_soil_data: Optional GEE-sourced soil data dict
        
        Returns:
            DataFrame with samples × metals proxy scores
        """
        if metals is None:
            metals = list(SoilElementProxyMapping.METAL_PROXIES.keys())
        
        scores = []
        
        for sample_id in adata.obs_names:
            # Try GEE data first, then fall back to adata.obs
            soil_data = {}
            
            if gee_soil_data and sample_id in gee_soil_data:
                soil_data = gee_soil_data[sample_id]
            else:
                # Extract from adata.obs
                sample_data = adata.obs.loc[sample_id]
                for scikit_col, description in SoilElementProxyMapping.SOILGRIDS_COMPOSITION.items():
                    if scikit_col in sample_data.index:
                        try:
                            value = float(sample_data[scikit_col])
                            # Map column to simple key
                            simple_key = scikit_col.replace('SoilGrids_', '')
                            soil_data[simple_key] = value
                        except (ValueError, TypeError):
                            pass
            
            # Calculate proxy for each metal
            sample_scores = {'sample_id': sample_id}
            for metal in metals:
                try:
                    proxy = SoilElementProxyMapping.calculate_proxy_score(soil_data, metal)
                    sample_scores[f'{metal}_proxy'] = proxy.proxy_score
                except Exception as e:
                    logger.warning(f"Could not calculate {metal} proxy for {sample_id}: {e}")
                    sample_scores[f'{metal}_proxy'] = np.nan
            
            scores.append(sample_scores)
        
        return pd.DataFrame(scores).set_index('sample_id')
    
    def correlate_traits_with_proxies(
        self,
        trait_matrix: pd.DataFrame,
        proxy_scores: pd.DataFrame,
        metal: str
    ) -> Dict[str, Any]:
        """
        Correlate trait presence with metal proxy scores.
        
        Args:
            trait_matrix: Features × Samples trait abundance matrix
            proxy_scores: Samples × Metals database
            metal: Metal to analyze
        
        Returns:
            Dict with correlations and statistics
        """
        metal_col = f'{metal}_proxy'
        if metal_col not in proxy_scores.columns:
            raise ValueError(f"No proxy data for {metal}")
        
        # Align indices
        common_samples = list(set(trait_matrix.index) & set(proxy_scores.index))
        if not common_samples:
            raise ValueError("No overlapping samples between traits and proxies")
        
        trait_subset = trait_matrix.loc[common_samples]
        proxy_subset = proxy_scores.loc[common_samples, metal_col]
        
        correlations = {}
        for trait in trait_subset.columns:
            trait_abund = trait_subset[trait]
            
            # Skip if no variance
            if trait_abund.std() == 0:
                continue
            
            # Spearman correlation (handles non-normal distributions)
            corr, pval = stats.spearmanr(trait_abund, proxy_subset, nan_policy='omit')
            
            correlations[trait] = {
                'correlation': corr,
                'pvalue': pval,
                'significant': pval < 0.05
            }
        
        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        
        return {
            'metal': metal,
            'correlations': dict(sorted_corr),
            'n_samples': len(common_samples),
            'n_traits': len(trait_subset.columns),
            'n_significant': sum(1 for c in correlations.values() if c['significant'])
        }
