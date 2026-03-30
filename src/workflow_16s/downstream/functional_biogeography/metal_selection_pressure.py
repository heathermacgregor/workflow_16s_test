"""
Module 3: Metal Selection Pressure Analysis

Quantifies the extent to which metal enrichment (inferred from geology and
soil chemistry proxies) correlates with metal-resistance trait distribution.

Tests hypothesis: "Is metal the primary environmental driver of trait selection?"

Workflow:
1. Build metal enrichment proxies from geologic data + soil composition
2. Calculate trait presence across samples
3. Correlate trait distribution with metal proxy scores
4. Generate visualizations and statistical summary
5. Report evidence strength for metal-driven selection
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from .geologic_data import USGSGeologicMapClient, MetalBearingFormations, get_geologic_client
from .earth_engine_geology import EarthEngineGeologyClient, get_gee_client
from .element_proxy import MetalProxyAnalyzer, SoilElementProxyMapping
from .functional_trait_mapping import create_trait_matrix

logger = logging.getLogger(__name__)


@dataclass
class MetalSelectionResult:
    """Results from metal selection pressure analysis"""
    metal: str
    n_samples: int
    proxy_mean: float
    proxy_std: float
    n_traits_analyzed: int
    n_traits_correlated: int
    top_trait: Optional[str]
    top_correlation: Optional[float]
    avg_correlation: Optional[float]
    selection_pressure_strength: float  # 0-1, combined evidence
    interpretation: str


class MetalSelectionPressureAnalyzer:
    """
    Analyzes metal selection pressure using proxy indicators.
    
    Uses two parallel approaches:
    1. **Geologic Proxy**: Infer metals from rock formations + rock type
    2. **Element Proxy**: Infer metals from soil composition patterns
    
    Combines both to generate metal enrichment score for each sample.
    """
    
    def __init__(
        self,
        adata: Any,  # AnnData object
        otu_metadata_path: Optional[str] = None,
        user_email: str = "macgregor@berkeley.edu",
        use_jgi: bool = True,
        cache_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        use_gee: bool = True
    ):
        """
        Initialize metal selection pressure analyzer.
        
        Args:
            adata: AnnData object with sample metadata
            otu_metadata_path: Path to OTU metadata file (for trait definitions)
            user_email: Email for database queries
            use_jgi: Use JGI database for trait definitions
            cache_dir: Directory for caching results
            config: Config dict with GEE credentials
            use_gee: Use Google Earth Engine for geospatial data (preferred)
        """
        self.adata = adata
        self.otu_metadata_path = otu_metadata_path
        self.user_email = user_email
        self.use_jgi = use_jgi
        self.use_gee = use_gee
        self.config = config or {}
        
        self.cache_dir = cache_dir or Path.home() / ".cache" / "workflow_16s_metal"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GEE client (primary data source)
        self.gee_client = None
        if use_gee:
            try:
                self.gee_client = get_gee_client(config)
                logger.info("Google Earth Engine client initialized (primary source)")
            except Exception as e:
                logger.warning(f"GEE initialization failed, using USGS REST API: {e}")
                self.gee_client = None
        
        # Fallback USGS client
        self.geologic_client = get_geologic_client({'cache_dir': str(self.cache_dir)})
        self.proxy_analyzer = MetalProxyAnalyzer()
        
        logger.info("Metal Selection Pressure Analyzer initialized")
    
    def run_analysis(
        self,
        metals: Optional[List[str]] = None,
        min_metal_proxy: float = 0.3,
        corr_threshold: float = 0.05  # p-value threshold
    ) -> Dict[str, MetalSelectionResult]:
        """
        Run complete metal selection pressure analysis.
        
        Args:
            metals: Metals to analyze (default: all available)
            min_metal_proxy: Minimum proxy score to consider sample "metal-enriched"
            corr_threshold: P-value threshold for significant correlations
        
        Returns:
            Dict mapping metal → MetalSelectionResult
        """
        if metals is None:
            metals = list(SoilElementProxyMapping.METAL_PROXIES.keys())
        
        logger.info(f"Running metal selection analysis for {len(metals)} metals")
        
        # Step 1: Calculate trait matrix
        logger.info("Building trait matrix...")
        trait_matrix = self._build_trait_matrix()
        
        # Step 2: Calculate geologic proxies for all samples
        logger.info("Calculating geologic proxies...")
        geologic_proxies = self._calculate_geologic_proxies(metals)
        
        # Step 3: Calculate element composition proxies
        logger.info("Calculating element composition proxies...")
        element_proxies = self._calculate_element_proxies(metals)
        
        # Step 4: Combine proxies
        logger.info("Combining proxy indicators...")
        combined_proxies = self._combine_proxies(geologic_proxies, element_proxies)
        
        # Step 5: Correlate traits with metal proxies
        logger.info("Correlating traits with metal proxies...")
        logger.debug(f"Trait matrix shape: {trait_matrix.shape}, Proxy shape: {combined_proxies.shape}")
        
        results = {}
        for metal in metals:
            try:
                result = self._analyze_metal_traits(
                    metal,
                    trait_matrix,
                    combined_proxies,
                    corr_threshold
                )
                results[metal] = result
            except Exception as e:
                logger.warning(f"Analysis failed for {metal}: {e}")
                # Store None as placeholder - don't skip entirely
                results[metal] = None
                continue
        
        # Step 6: Save results
        self._save_results(results)
        
        return results
    
    def _build_trait_matrix(self) -> pd.DataFrame:
        """
        Build trait abundance matrix (features × samples).
        
        Returns:
            DataFrame with trait counts across samples
        """
        try:
            # Use functional_trait_mapping module
            trait_result = create_trait_matrix(
                self.adata,
                otu_metadata_path=self.otu_metadata_path,
                user_email=self.user_email,
                use_jgi=self.use_jgi
            )
            
            # create_trait_matrix returns (DataFrame, database) tuple
            if isinstance(trait_result, tuple):
                trait_matrix, _ = trait_result
            elif isinstance(trait_result, pd.DataFrame):
                trait_matrix = trait_result
            else:
                raise ValueError(f"Unexpected trait result type: {type(trait_result)}")
            
            if trait_matrix is None or trait_matrix.empty:
                logger.warning("Trait matrix is empty - will generate aggregated traits from OTUs")
                # Aggregate OTU traits to samples (sum across OTUs for each sample)
                trait_matrix = self.adata.X.sum(axis=1)
                trait_matrix = pd.DataFrame(
                    trait_matrix,
                    index=self.adata.obs_names,
                    columns=['OTU_abundance']
                )
                return trait_matrix
            
            # Check if trait matrix has sample or OTU indices
            if len(trait_matrix.index) == len(self.adata.obs_names):
                # Already has sample indices
                logger.debug(f"Trait matrix has {len(trait_matrix)} samples")
                return trait_matrix
            elif len(trait_matrix.index) == len(self.adata.var_names):
                # Has OTU indices, need to aggregate
                logger.debug(f"Aggregating {len(trait_matrix)} OTU traits to {len(self.adata.obs_names)} samples")
                # Matrix multiplication: samples x OTUs @ OTUs x traits = samples x traits
                # self.adata.X is (samples, OTUs)
                # trait_matrix is (OTUs, traits)
                # Result is (samples, traits)
                
                # Ensure indices align: reindex trait matrix to match adata.var_names order
                trait_matrix = trait_matrix.reindex(self.adata.var_names, fill_value=0)
                
                aggregated = self.adata.X @ trait_matrix.values
                trait_matrix = pd.DataFrame(
                    aggregated,
                    index=self.adata.obs_names,
                    columns=trait_matrix.columns
                )
                return trait_matrix
            else:
                raise ValueError(f"Trait matrix index length ({len(trait_matrix.index)}) doesn't match samples ({len(self.adata.obs_names)}) or OTUs ({len(self.adata.var_names)})")
        
        except Exception as e:
            logger.error(f"Could not build trait matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return minimal matrix as fallback (samples × 1 column of zeros)
            # Shape: (num_samples, 1) NOT (0, num_samples)
            fallback_data = np.zeros((len(self.adata.obs_names), 1))
            return pd.DataFrame(
                fallback_data,
                index=self.adata.obs_names,
                columns=['fallback_trait']
            )
    
    def _calculate_geologic_proxies(
        self,
        metals: List[str]
    ) -> pd.DataFrame:
        """
        Calculate metal proxies from geologic data.
        
        Priority:
        1. Google Earth Engine (if authenticated) - highest resolution, global coverage
        2. USGS REST API (fallback) - REST-based queries
        3. Default values - if no geographic data available
        """
        proxies = []
        
        # Try to find coordinate columns
        lat_col = self._find_coordinate_column('latitude')
        lon_col = self._find_coordinate_column('longitude')
        
        if not (lat_col and lon_col):
            logger.info("No geographic data found; using default geologic proxies")
            return self._default_geologic_proxies(metals)
        
        for sample_id in self.adata.obs_names:
            # Initialize outside try block to avoid UnboundLocalError in except
            # Preload all metals to ensure they exist in all code paths
            metal_scores = {f'{metal}_geo': 0.5 for metal in metals}
            data_source = 'default'
            
            try:
                # Get coordinate values
                lat_val = self.adata.obs.loc[sample_id, lat_col]
                lon_val = self.adata.obs.loc[sample_id, lon_col]
                
                # Check for invalid coordinate strings
                if isinstance(lat_val, str) and lat_val.lower() in ['unknown', 'na', 'nan', '']:
                    logger.debug(f"Skipping {sample_id}: latitude is '{lat_val}'")
                    metal_scores['data_source'] = 'skipped_invalid_coords'
                    proxies.append({**metal_scores, 'sample_id': sample_id})
                    continue
                
                if isinstance(lon_val, str) and lon_val.lower() in ['unknown', 'na', 'nan', '']:
                    logger.debug(f"Skipping {sample_id}: longitude is '{lon_val}'")
                    metal_scores['data_source'] = 'skipped_invalid_coords'
                    proxies.append({**metal_scores, 'sample_id': sample_id})
                    continue
                
                lat = float(lat_val)
                lon = float(lon_val)
                
                # Validate coordinate ranges
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    logger.debug(f"Skipping {sample_id}: invalid coordinate ranges (lat={lat}, lon={lon})")
                    metal_scores['data_source'] = 'skipped_out_of_range'
                    proxies.append({**metal_scores, 'sample_id': sample_id})
                    continue
                
                # Try GEE first
                if self.gee_client:
                    try:
                        gee_result = self.gee_client.query_geology_by_point(lat, lon)
                        if gee_result:
                            # Infer metal proxies from GEE geology result
                            rock_type = gee_result.rock_type
                            if rock_type:
                                metal_proxies = MetalBearingFormations.get_rock_type_metals(rock_type)
                                for metal in metals:
                                    metal_scores[f'{metal}_geo'] = metal_proxies.get(metal, 0.5)
                            else:
                                # Use baseline from lithology confidence
                                for metal in metals:
                                    metal_scores[f'{metal}_geo'] = gee_result.metal_bearing_confidence
                            
                            data_source = 'GEE'
                    except Exception as e:
                        logger.debug(f"GEE query failed for {sample_id}: {e}")
                
                # Fallback to USGS if GEE didn't work
                if data_source == 'default' and not metal_scores:
                    try:
                        usgs_geology = self.geologic_client.query_geology_by_coordinates(lat, lon)
                        if usgs_geology:
                            metal_proxies = self.geologic_client.infer_metal_proxy(
                                usgs_geology.get('rock_type'),
                                usgs_geology.get('formation')
                            )
                            for metal in metals:
                                metal_scores[f'{metal}_geo'] = metal_proxies.get(metal, 0.5)
                            data_source = 'USGS'
                    except Exception as e:
                        logger.debug(f"USGS query failed for {sample_id}: {e}")
                
                # Use neutral scores if all sources failed
                if not metal_scores:
                    for metal in metals:
                        metal_scores[f'{metal}_geo'] = 0.5
            
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Could not process coordinates for {sample_id}: {e}")
                for metal in metals:
                    metal_scores[f'{metal}_geo'] = 0.5
            
            metal_scores['sample_id'] = sample_id
            proxies.append(metal_scores)
        
        result_df = pd.DataFrame(proxies).set_index('sample_id')
        logger.info(f"Geologic proxies calculated ({data_source} source)")
        return result_df
    
    def _calculate_element_proxies(
        self,
        metals: List[str]
    ) -> pd.DataFrame:
        """
        Calculate metal proxies from soil element composition.
        
        Priority:
        1. Google Earth Engine SoilGrids (250m resolution, global)
        2. Existing SoilGrids data in adata.obs (if available)
        3. Neutral proxies if unavailable
        """
        # Collect soil data from GEE if available with coordinates
        soil_data_gee = {}
        lat_col = self._find_coordinate_column('latitude')
        lon_col = self._find_coordinate_column('longitude')
        
        if self.gee_client and lat_col and lon_col:
            try:
                logger.info("Fetching SoilGrids from Google Earth Engine...")
                for sample_id in self.adata.obs_names:
                    try:
                        lat = float(self.adata.obs.loc[sample_id, lat_col])
                        lon = float(self.adata.obs.loc[sample_id, lon_col])
                        
                        gee_soil = self.gee_client.query_soil_elements_by_point(lat, lon)
                        if gee_soil:
                            soil_data_gee[sample_id] = {
                                'clay': gee_soil.clay,
                                'silt': gee_soil.silt,
                                'sand': gee_soil.sand,
                                'soc': gee_soil.organic_carbon,
                                'cec': gee_soil.cation_exchange_capacity,
                                'phh2o': gee_soil.ph_water,
                                'bdod': gee_soil.bulk_density
                            }
                    except Exception as e:
                        logger.debug(f"GEE soil fetch failed for {sample_id}: {e}")
                
                if soil_data_gee:
                    logger.info(f"Fetched SoilGrids from GEE for {len(soil_data_gee)} samples")
            except Exception as e:
                logger.warning(f"GEE SoilGrids fetch failed: {e}, falling back to adata")
        
        try:
            # Calculate proxies from available soil data
            element_proxies = self.proxy_analyzer.calculate_sample_proxies(
                self.adata,
                metals=metals,
                gee_soil_data=soil_data_gee if soil_data_gee else None
            )
            # Rename columns to match naming convention
            element_proxies.columns = [f"{col.replace('_proxy', '')}_elem" 
                                      for col in element_proxies.columns]
            return element_proxies
        except Exception as e:
            logger.warning(f"Element proxy calculation failed: {e}")
            # Return neutral proxies
            return self._default_element_proxies(metals)
    
    def _combine_proxies(
        self,
        geologic: pd.DataFrame,
        element: pd.DataFrame,
        geo_weight: float = 0.4,
        elem_weight: float = 0.6
    ) -> pd.DataFrame:
        """
        Combine geologic and element proxies into unified score.
        
        Uses weighted average with optional weights.
        """
        combined = pd.DataFrame(index=self.adata.obs_names)
        
        metals = [col.split('_')[0] for col in geologic.columns]
        unique_metals = list(set(metals))
        
        for metal in unique_metals:
            geo_col = f'{metal}_geo'
            elem_col = f'{metal}_elem'
            
            geo_scores = geologic.get(geo_col, 0.5)
            elem_scores = element.get(elem_col, 0.5)
            
            # Weighted average
            combined[f'{metal}_proxy'] = (
                geo_scores * geo_weight + elem_scores * elem_weight
            )
        
        return combined
    
    def _analyze_metal_traits(
        self,
        metal: str,
        trait_matrix: pd.DataFrame,
        proxy_scores: pd.DataFrame,
        corr_threshold: float
    ) -> MetalSelectionResult:
        """
        Analyze trait distribution relative to a single metal proxy.
        """
        proxy_col = f'{metal}_proxy'
        
        if proxy_col not in proxy_scores.columns:
            raise ValueError(f"No proxy data for {metal}")
        
        # Correlate traits with metal proxy
        corr_dict = self.proxy_analyzer.correlate_traits_with_proxies(
            trait_matrix,
            proxy_scores[[proxy_col]].rename(columns={proxy_col: f'{metal}_proxy'}),
            metal
        )
        
        # Extract statistics
        correlations = corr_dict['correlations']
        proxy_data = proxy_scores[proxy_col].dropna()
        
        significant_traits = [
            t for t, c in correlations.items() if c['significant']
        ]
        
        # Identify top trait
        top_trait = None
        top_corr = None
        if correlations:
            top_trait = max(correlations.items(), 
                          key=lambda x: abs(x[1]['correlation']))[0]
            top_corr = correlations[top_trait]['correlation']
        
        # Calculate average absolute correlation
        all_corrs = [c['correlation'] for c in correlations.values() 
                   if not np.isnan(c['correlation'])]
        avg_corr = np.mean(np.abs(all_corrs)) if all_corrs else 0
        
        # Assess selection pressure strength
        # Combine: number of significant correlations + strength of top correlation
        n_sig_ratio = len(significant_traits) / max(1, len(correlations))
        top_corr_strength = abs(top_corr) if top_corr else 0
        
        selection_strength = (n_sig_ratio * 0.4 + top_corr_strength * 0.6)
        
        # Generate interpretation
        if selection_strength > 0.7:
            interp = f"STRONG evidence: {metal} enrichment strongly correlates with metal-resistance traits"
        elif selection_strength > 0.5:
            interp = f"MODERATE evidence: {metal} enrichment shows correlation with some traits"
        elif selection_strength > 0.3:
            interp = f"WEAK evidence: {metal} enrichment weakly correlates with trait distribution"
        else:
            interp = f"MINIMAL evidence: {metal} enrichment does not predict trait distribution"
        
        return MetalSelectionResult(
            metal=metal,
            n_samples=len(proxy_data),
            proxy_mean=float(proxy_data.mean()),
            proxy_std=float(proxy_data.std()),
            n_traits_analyzed=len(correlations),
            n_traits_correlated=len(significant_traits),
            top_trait=top_trait,
            top_correlation=float(top_corr) if top_corr else None,
            avg_correlation=float(avg_corr),
            selection_pressure_strength=float(selection_strength),
            interpretation=interp
        )
    
    def _find_coordinate_column(self, coord_type: str) -> Optional[str]:
        """Find latitude or longitude column in metadata"""
        patterns = {
            'latitude': ['latitude', 'lat', 'Latitude', 'LAT'],
            'longitude': ['longitude', 'lon', 'Longitude', 'LON', 'lng']
        }
        
        for pattern in patterns.get(coord_type, []):
            matching = [c for c in self.adata.obs.columns if pattern.lower() in c.lower()]
            if matching:
                return matching[0]
        
        return None
    
    def _default_geologic_proxies(self, metals: List[str]) -> pd.DataFrame:
        """Return neutral geologic proxies when no data available"""
        data = {'sample_id': list(self.adata.obs_names)}
        for metal in metals:
            data[f'{metal}_geo'] = [0.5] * len(self.adata.obs_names)
        return pd.DataFrame(data).set_index('sample_id')
    
    def _default_element_proxies(self, metals: List[str]) -> pd.DataFrame:
        """Return neutral element proxies when calculation fails"""
        data = {}
        for metal in metals:
            data[f'{metal}_elem'] = [0.5] * len(self.adata.obs_names)
        return pd.DataFrame(data, index=self.adata.obs_names)
    
    def _save_results(self, results: Dict[str, MetalSelectionResult]) -> None:
        """Save analysis results to JSON"""
        output_file = self.cache_dir / "metal_selection_results.json"
        
        # Filter out None results (from failed analyses) and convert dataclasses to dicts
        valid_results = {}
        for metal, result in results.items():
            if result is not None:
                try:
                    valid_results[metal] = asdict(result)
                except TypeError:
                    # If not a dataclass, skip it
                    logger.warning(f"Result for {metal} is not a dataclass, skipping save")
                    continue
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(self.adata),
            'metals_analyzed': list(valid_results.keys()),
            'results': valid_results
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
