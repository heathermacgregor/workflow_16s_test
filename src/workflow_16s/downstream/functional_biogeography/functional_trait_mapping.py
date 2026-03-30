"""
Functional Trait Mapping

Maps gene functions and metabolic capabilities to OTUs based on RAST annotations.
Focuses on metal-resistance, uranium-reduction, and other environmental adaptations.

This module translates genomic data into discrete functional traits that can be
analyzed for phylogenetic signal (Pagel's lambda) and ecological distribution.

Data Integration:
- Primary: JGI/IMG database (via KEGG, InterPro REST APIs)
- Secondary: User-provided RAST annotations (otus.97.allinfo, etc.)
- Fallback: Curated trait definitions from literature
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import json

try:
    from .jgi_database import JGIDatabaseClient, get_jgi_client
    JGI_AVAILABLE = True
except ImportError:
    JGI_AVAILABLE = False
    get_jgi_client = None

logger = logging.getLogger(__name__)


@dataclass
class FunctionalTrait:
    """Represents a single functional trait with gene signatures."""
    name: str
    description: str
    gene_keywords: List[str]  # Keywords to search for in gene annotations
    ec_numbers: List[str] = field(default_factory=list)  # EC numbers to search for
    kegg_kos: List[str] = field(default_factory=list)  # KEGG KO identifiers
    pfam_domains: List[str] = field(default_factory=list)  # PFAM domain signatures
    confidence: float = 0.8  # How confident are we in this prediction?


class MetalResistanceGeneDatabase:
    """
    Database of functional traits for microbial metabolism.
    
    Data sources (in priority order):
    1. JGI/IMG via KEGG REST API (authoritative)
    2. InterPro (protein domain signatures)
    3. User-provided RAST annotations
    4. Fallback: Curated trait definitions from literature
    """
    
    def __init__(self, user_email: str = "macgregor@berkeley.edu", use_jgi: bool = True):
        """
        Initialize the trait database.
        
        Parameters
        ----------
        user_email : str
            Berkeley email for JGI database access
        use_jgi : bool
            Whether to try loading from JGI/KEGG (requires internet)
        """
        self.traits: Dict[str, FunctionalTrait] = {}
        self.user_email = user_email
        self.use_jgi = use_jgi and JGI_AVAILABLE
        
        # Initialize JGI client if available
        self.jgi_client = None
        if self.use_jgi:
            try:
                self.jgi_client = get_jgi_client({'email': user_email})
                logger.info(f"✓ JGI Database Client initialized for {user_email}")
            except Exception as e:
                logger.warning(f"Failed to initialize JGI client: {e}. Using fallback traits.")
                self.jgi_client = None
        
        # Build database from JGI or fallback
        self._build_database()
    
    def _build_database(self):
        """Build trait database from JGI or fallback to curated definitions."""
        
        trait_list = [
            # Metal Resistance & Cycling
            'uranium_reduction',
            'heavy_metal_efflux',
            'iron_manganese_reduction',
            'arsenic_metabolism',
            'mercury_resistance',
            'copper_resistance',
            'chromium_resistance',
            'cobalt_nickel_resistance',
            'cadmium_zinc_resistance',
            'metal_bioaccumulation',
            
            # Nitrogen Cycling
            'nitrate_reduction',
            'nitrification',
            'nitrogen_fixation',
            'ammonia_oxidation',
            
            # Sulfur & Carbon Metabolism
            'sulfur_metabolism',
            'hydrogen_metabolism',
            'carbon_fixation',
            'photosynthesis',
            
            # Stress Response & Virulence
            'biofilm_formation',
            'oxidative_stress_response',
            'heat_shock_response',
            'desiccation_resistance',
            'motility_chemotaxis',
            'virulence_factors'
        ]
        
        for trait_name in trait_list:
            trait = self._load_trait(trait_name)
            if trait:
                self.traits[trait_name] = trait
        
        logger.info(f"Loaded {len(self.traits)} functional trait definitions from "
                   f"{'JGI/KEGG' if self.jgi_client else 'fallback curated database'}")
    
    def _load_trait(self, trait_name: str) -> Optional[FunctionalTrait]:
        """Load trait from JGI or fallback database."""
        
        # Try JGI first
        if self.jgi_client:
            try:
                jgi_data = self.jgi_client.get_functional_trait_genes(trait_name)
                if jgi_data.get('status') == 'success':
                    logger.debug(f"  Loaded {trait_name} from JGI/KEGG")
                    return FunctionalTrait(
                        name=trait_name,
                        description=jgi_data.get('description', trait_name),
                        gene_keywords=jgi_data.get('gene_keywords', []),
                        ec_numbers=jgi_data.get('ec_numbers', []),
                        kegg_kos=jgi_data.get('kegg_kos', []),
                        pfam_domains=jgi_data.get('interpro', []),
                    )
            except Exception as e:
                logger.debug(f"JGI query failed for {trait_name}: {e}")
        
        # Fallback to curated definitions
        logger.debug(f"  Loading {trait_name} from fallback curated database")
        return self._get_fallback_trait(trait_name)
    
    def _get_fallback_trait(self, trait_name: str) -> Optional[FunctionalTrait]:
        """Get trait from curated fallback database (literature-based)."""
        
        fallback_traits = {
            'uranium_reduction': FunctionalTrait(
                name='uranium_reduction',
                description='Uranium reduction capability via c-type cytochromes (JGI/Lovley et al.)',
                gene_keywords=[
                    'cytochrome', 'c-type cytochrome', 'omcB', 'omcC', 'omcE', 'omcZ',
                    'geobacter', 'dissimilatory', 'metal reducing', 
                    'outer membrane cytochrome', 'multiheme cytochrome'
                ],
                ec_numbers=['1.7.2.8', '1.7.2.9', '1.7.99.-'],
                kegg_kos=['K05301', 'K05302']
            ),
            'heavy_metal_efflux': FunctionalTrait(
                name='heavy_metal_efflux',
                description='Active efflux pumps for heavy metals (KEGG pathway M00412)',
                gene_keywords=[
                    'efflux pump', 'RND transporter', 'copper resistance', 
                    'heavy metal ATPase', 'P-type ATPase', 'cus system', 'czc',
                    'copA', 'copB', 'cusA', 'cusB', 'czrA', 'zntA'
                ],
                ec_numbers=['3.6.3.6', '3.6.3.8'],
                kegg_kos=['K01537', 'K01538', 'K07799']
            ),
            'iron_manganese_reduction': FunctionalTrait(
                name='iron_manganese_reduction',
                description='Dissimilatory iron/manganese reduction',
                gene_keywords=[
                    'dissimilatory', 'iron reduction', 'manganese reduction',
                    'mtr', 'decahemin', 'geobacter', 'shewanella',
                    'Fe(III) reductase', 'Mn(IV) reductase'
                ],
                ec_numbers=['1.7.2.3', '1.6.6.2'],
                kegg_kos=['K00405']
            ),
            'arsenic_metabolism': FunctionalTrait(
                name='arsenic_metabolism',
                description='Arsenic oxidation and reduction (KEGG pathway M00558)',
                gene_keywords=[
                    'arsenic', 'aioA', 'arrA', 'arsTj', 'arsenic oxidation',
                    'arsenic reduction', 'arsenic-3 oxidase', 'arsenate respiration'
                ],
                ec_numbers=['1.20.9.1', '1.20.99.-'],
                kegg_kos=['K10670', 'K10676', 'K10677']
            ),
            'mercury_resistance': FunctionalTrait(
                name='mercury_resistance',
                description='Mercury detoxification via merA and merB (mer operon)',
                gene_keywords=[
                    'mercuric reductase', 'merA', 'merB', 'mercury resistance', 'organomercurial lyase',
                    'mercury detoxification', 'merE', 'merF', 'merG', 'merT'
                ],
                ec_numbers=['1.16.1.2', '4.99.1.3'],
                kegg_kos=['K07730', 'K07731']
            ),
            'copper_resistance': FunctionalTrait(
                name='copper_resistance',
                description='Copper tolerance via CopA/CopB pumps and multicopper oxidases',
                gene_keywords=[
                    'copA', 'copB', 'copC', 'copD', 'copper resistance', 'copper ATPase',
                    'multicopper oxidase', 'laccase', 'cytochrome c oxidase', 'cusC', 'cusS'
                ],
                ec_numbers=['3.6.3.6', '1.9.3.2'],
                kegg_kos=['K07800', 'K07801']
            ),
            'chromium_resistance': FunctionalTrait(
                name='chromium_resistance',
                description='Chromium reduction and detoxification',
                gene_keywords=[
                    'chromium reduction', 'chrA', 'chrB', 'chromate reductase', 'Cr(VI) reduction',
                    'chromium ATPase', 'PglA', 'YieF'
                ],
                ec_numbers=['1.5.1.-', '1.97.1.-'],
                kegg_kos=['K09015']
            ),
            'cobalt_nickel_resistance': FunctionalTrait(
                name='cobalt_nickel_resistance',
                description='Cobalt and nickel resistance via RcnA and NreB exporters',
                gene_keywords=[
                    'rcnA', 'nreB', 'cobalt resistance', 'nickel resistance', 'cobalt efflux',
                    'nickel efflux', 'CNT transporter', 'metal ion exporter'
                ],
                ec_numbers=['3.6.3.1'],
                kegg_kos=['K07804']
            ),
            'cadmium_zinc_resistance': FunctionalTrait(
                name='cadmium_zinc_resistance',
                description='Cadmium and zinc tolerance via ZntA and CdtA',
                gene_keywords=[
                    'zntA', 'cdtA', 'cadmium resistance', 'zinc resistance', 'cation ATPase',
                    'P-type ATPase', 'metal ion exporter', 'HMA'
                ],
                ec_numbers=['3.6.3.8'],
                kegg_kos=['K01538']
            ),
            'metal_bioaccumulation': FunctionalTrait(
                name='metal_bioaccumulation',
                description='Metal uptake and intracellular accumulation capability',
                gene_keywords=[
                    'bioaccumulation', 'sequestration', 'metallothionein', 'metal binding protein',
                    'iron uptake', 'iron transporter', 'siderophore', 'enterobactin', 'transport'
                ],
                ec_numbers=['1.16.1.1', '3.6.3.14'],
                kegg_kos=['K02010', 'K02012']
            ),
            'nitrate_reduction': FunctionalTrait(
                name='nitrate_reduction',
                description='Nitrate reduction and denitrification (KEGG pathway M00529)',
                gene_keywords=[
                    'nitrate reductase', 'nas', 'nar', 'nir', 'nor', 'nos',
                    'denitrification', 'dissimilatory nitrate reduction', 'nitrite reductase'
                ],
                ec_numbers=['1.7.5.2', '1.7.2.1', '1.7.2.4', '1.7.2.5'],
                kegg_kos=['K00370', 'K00371', 'K15876']
            ),
            'nitrification': FunctionalTrait(
                name='nitrification',
                description='Ammonia and nitrite oxidation (assimilatory & dissimilatory)',
                gene_keywords=[
                    'ammonia monooxygenase', 'amoA', 'amoB', 'amoC', 'nitrite oxidase',
                    'nxrA', 'nxrB', 'nitrification', 'ammonia oxidation', 'comammox'
                ],
                ec_numbers=['1.14.99.39', '1.7.2.3'],
                kegg_kos=['K10944', 'K10945']
            ),
            'nitrogen_fixation': FunctionalTrait(
                name='nitrogen_fixation',
                description='Nitrogen fixation via nitrogenase (nif cluster)',
                gene_keywords=[
                    'nitrogenase', 'nifA', 'nifB', 'nifD', 'nifE', 'nifH', 'nifK',
                    'nitrogen fixation', 'dinitrogen reduction', 'molybdenum-iron cluster'
                ],
                ec_numbers=['1.19.6.1'],
                kegg_kos=['K02588', 'K02591']
            ),
            'ammonia_oxidation': FunctionalTrait(
                name='ammonia_oxidation',
                description='Ammonia oxidation for energy (ammonia-oxidizing archaea/bacteria)',
                gene_keywords=[
                    'ammonia oxidation', 'AOA', 'AOB', 'hydroxylamine oxidoreductase',
                    'HAO', 'ammonia monooxygenase', 'amoA', 'Nitrosomonas', 'Nitrososphaera'
                ],
                ec_numbers=['1.14.99.39'],
                kegg_kos=['K10944']
            ),
            'sulfur_metabolism': FunctionalTrait(
                name='sulfur_metabolism',
                description='Sulfate reduction and sulfur oxidation',
                gene_keywords=[
                    'sulfate reductase', 'aprA', 'aprB', 'dsrA', 'dsrB',
                    'sulfite reductase', 'sulfur oxidation', 'thiosulfate', 'soxA', 'soxB'
                ],
                ec_numbers=['1.8.99.2', '1.8.99.5'],
                kegg_kos=['K00394', 'K00395']
            ),
            'hydrogen_metabolism': FunctionalTrait(
                name='hydrogen_metabolism',
                description='Hydrogen oxidation and production (energy metabolism)',
                gene_keywords=[
                    'hydrogenase', 'hyd', 'hydrogen oxidation', 'hydrogen evolution',
                    'nickel-iron hydrogenase', 'uptake hydrogenase', 'FeFe-hydrogenase', 'H2'
                ],
                ec_numbers=['1.12.1.6', '1.12.1.2', '1.12.2.1'],
                kegg_kos=['K00532', 'K00533']
            ),
            'carbon_fixation': FunctionalTrait(
                name='carbon_fixation',
                description='CO2 fixation pathways (Calvin, RuBisCO, 3-HP cycle)',
                gene_keywords=[
                    'RuBisCO', 'ribulose-1,5-bisphosphate carboxylase', 'rbcL', 'rbcS',
                    'carbon fixation', 'CO2 fixation', 'Calvin cycle', '3-HP'
                ],
                ec_numbers=['4.1.1.39'],
                kegg_kos=['K01601', 'K01602']
            ),
            'photosynthesis': FunctionalTrait(
                name='photosynthesis',
                description='Photosynthetic capability via photosystem complexes',
                gene_keywords=[
                    'photosystem', 'photosynthesis', 'chlorophyll', 'bacteriochlorophyll',
                    'reaction center', 'photosynthetic membrane', 'pufL', 'pufM', 'RC'
                ],
                ec_numbers=['1.97.-.-'],
                kegg_kos=['K08901', 'K08902']
            ),
            'biofilm_formation': FunctionalTrait(
                name='biofilm_formation',
                description='Biofilm and EPS production (survival strategy)',
                gene_keywords=[
                    'polysaccharide dehydratase', 'capsular', 'exopolysaccharide',
                    'glycosyl transferase', 'biofilm', 'psl', 'pga', 'pel',
                    'alginate', 'cellulose synthesis', 'adhesin'
                ],
                ec_numbers=['2.4.-.-'],
                kegg_kos=['K13010', 'K14633']
            ),
            'oxidative_stress_response': FunctionalTrait(
                name='oxidative_stress_response',
                description='Antioxidant defense: catalase, peroxidase, SOD',
                gene_keywords=[
                    'catalase', 'peroxidase', 'superoxide dismutase', 'SOD', 'katA', 'katG',
                    'ahpC', 'glutathione', 'oxidative stress', 'antioxidant', 'ROS detoxification'
                ],
                ec_numbers=['1.11.1.6', '1.11.1.21', '1.15.1.1'],
                kegg_kos=['K03564', 'K03782']
            ),
            'heat_shock_response': FunctionalTrait(
                name='heat_shock_response',
                description='Heat shock proteins and stress response (Hsp60, Hsp70, GroEL)',
                gene_keywords=[
                    'heat shock protein', 'hsp', 'groEL', 'groES', 'dnaK', 'dnaJ',
                    'chaperone', 'stress response', 'protease', 'Lon', 'ClpX'
                ],
                ec_numbers=['3.4.21.-'],
                kegg_kos=['K04043', 'K04044']
            ),
            'desiccation_resistance': FunctionalTrait(
                name='desiccation_resistance',
                description='Osmolyte accumulation and desiccation tolerance',
                gene_keywords=[
                    'trehalose', 'glycine betaine', 'proline', 'osmolyte', 'osmoprotectant',
                    'desiccation', 'drought resistance', 'compatible solute', 'glycerol'
                ],
                ec_numbers=['2.4.1.-', '4.3.1.1'],
                kegg_kos=['K00947', 'K00948']
            ),
            'motility_chemotaxis': FunctionalTrait(
                name='motility_chemotaxis',
                description='Flagellar motility and chemotaxis',
                gene_keywords=[
                    'flagella', 'flagellar', 'motility', 'chemotaxis', 'cheA', 'cheB', 'cheW',
                    'fliA', 'fliC', 'fliD', 'flagellin', 'motor protein'
                ],
                ec_numbers=['3.6.4.12'],
                kegg_kos=['K02409', 'K02410']
            ),
            'virulence_factors': FunctionalTrait(
                name='virulence_factors',
                description='Pathogenicity and virulence-associated genes',
                gene_keywords=[
                    'virulence', 'pathogenicity', 'toxin', 'toxin-antitoxin', 'effector',
                    'adhesion', 'invasion', 'secretion', 'Type III secretion', 'pili'
                ],
                ec_numbers=['3.4.21.-', '3.4.22.-'],
                kegg_kos=['K04769']
            ),
        }
        
        return fallback_traits.get(trait_name)
    
    def get_trait(self, trait_name: str) -> Optional[FunctionalTrait]:
        """Get a specific trait by name."""
        return self.traits.get(trait_name)
    
    def list_traits(self) -> List[str]:
        """List all available traits."""
        return list(self.traits.keys())
    
    def get_all_traits(self) -> Dict[str, FunctionalTrait]:
        """Get all traits."""
        return self.traits


def extract_traits_from_otu_metadata(
    otu_metadata: Dict[str, Any],
    trait_db: MetalResistanceGeneDatabase,
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Extract functional traits from OTU metadata (RAST annotations).
    
    Parameters
    ----------
    otu_metadata : Dict[str, Any]
        OTU metadata including annotations, gene names, etc.
    trait_db : MetalResistanceGeneDatabase
        Database of traits to search for
    confidence_threshold : float
        Only return traits with confidence >= threshold
    
    Returns
    -------
    Dict[str, float]
        Mapping of trait names to confidence scores (0-1)
    """
    detected_traits = {}
    
    # Extract text fields that might contain gene annotations
    annotation_text = str(otu_metadata).lower()
    
    for trait_name, trait in trait_db.get_all_traits().items():
        confidence_score = 0.0
        
        # Search for gene keywords
        keyword_matches = 0
        for keyword in trait.gene_keywords:
            if keyword.lower() in annotation_text:
                keyword_matches += 1
                confidence_score = min(1.0, keyword_matches * 0.2)  # Each match adds 0.2
        
        # If we found keywords and meet threshold, add it
        if confidence_score >= confidence_threshold:
            detected_traits[trait_name] = min(confidence_score, trait.confidence)
            logger.debug(f"Detected trait {trait_name} with confidence {detected_traits[trait_name]:.2f}")
    
    return detected_traits


def map_traits_to_otus(
    otu_data_path: str,
    trait_db: Optional[MetalResistanceGeneDatabase] = None,
    confidence_threshold: float = 0.5,
    sample_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Map functional traits to all OTUs in the dataset.
    
    Parameters
    ----------
    otu_data_path : str
        Path to the OTU metadata file (e.g., otus.97.allinfo)
    trait_db : MetalResistanceGeneDatabase, optional
        Trait database (creates new if not provided)
    confidence_threshold : float
        Minimum confidence for trait assignment
    sample_n : int, optional
        If provided, only process first N OTUs (for testing)
    
    Returns
    -------
    pd.DataFrame
        OTU x Trait matrix with confidence scores
    """
    if trait_db is None:
        trait_db = MetalResistanceGeneDatabase()
    
    otu_traits = {}
    trait_names = list(trait_db.list_traits())
    
    logger.info(f"Mapping {len(trait_names)} traits to OTUs from {otu_data_path}")
    
    try:
        # Try to parse the OTU allinfo file
        with open(otu_data_path, 'r') as f:
            line_count = 0
            for line in f:
                if sample_n and line_count >= sample_n:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse line (format depends on allinfo structure)
                # Typically: OTU_ID, sequence, annotations, etc.
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                otu_id = parts[0]
                metadata_text = '\t'.join(parts[1:])  # Rest is annotation
                
                # Extract traits for this OTU
                traits = extract_traits_from_otu_metadata(
                    metadata_text, 
                    trait_db, 
                    confidence_threshold
                )
                
                if traits:
                    otu_traits[otu_id] = traits
                
                line_count += 1
        
        logger.info(f"Processed {line_count} OTUs, found traits in {len(otu_traits)} OTUs")
        
    except Exception as e:
        logger.error(f"Error reading OTU metadata file: {e}")
        # Return empty matrix
        return pd.DataFrame()
    
    # Convert to DataFrame
    trait_matrix = pd.DataFrame.from_dict(otu_traits, orient='index', columns=trait_names)
    trait_matrix = trait_matrix.fillna(0.0)
    
    logger.info(f"Created trait matrix: {trait_matrix.shape[0]} OTUs x {trait_matrix.shape[1]} traits")
    
    return trait_matrix


def create_trait_matrix(
    adata,
    otu_metadata_path: Optional[str] = None,
    user_email: str = "macgregor@berkeley.edu",
    use_jgi: bool = True,
    trait_db: Optional[MetalResistanceGeneDatabase] = None,
    use_existing_annot: bool = True
) -> Tuple[pd.DataFrame, MetalResistanceGeneDatabase]:
    """
    Create a trait matrix from AnnData object and optional OTU metadata file.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with OTU data
    otu_metadata_path : str, optional
        Path to external OTU metadata file for enrichment
    user_email : str
        Berkeley email for JGI database access
    use_jgi : bool
        Whether to use JGI/KEGG database for trait definitions
    trait_db : MetalResistanceGeneDatabase, optional
        Trait database (creates new if not provided)
    use_existing_annot : bool
        Try to extract traits from existing adata.var metadata
    
    Returns
    -------
    Tuple[pd.DataFrame, MetalResistanceGeneDatabase]
        (OTU x Trait matrix, trait database used)
    """
    if trait_db is None:
        trait_db = MetalResistanceGeneDatabase(user_email=user_email, use_jgi=use_jgi)
    
    trait_matrix = None
    
    # If external metadata file provided, use it
    if otu_metadata_path and Path(otu_metadata_path).exists():
        logger.info(f"Loading traits from external OTU metadata: {otu_metadata_path}")
        trait_matrix = map_traits_to_otus(otu_metadata_path, trait_db)
    
    # If no external file or want to supplement, try existing annotations
    if use_existing_annot and hasattr(adata, 'var'):
        logger.info("Extracting traits from existing adata.var annotations")
        
        existing_traits = {}
        for otu_id in adata.var_names:
            if otu_id in adata.var.index:
                row = adata.var.loc[otu_id]
                # Search all columns for trait keywords
                row_dict = row.to_dict()
                traits = extract_traits_from_otu_metadata(row_dict, trait_db)
                if traits:
                    existing_traits[otu_id] = traits
        
        if existing_traits:
            existing_df = pd.DataFrame.from_dict(
                existing_traits, 
                orient='index',
                columns=list(trait_db.list_traits())
            ).fillna(0.0)
            
            # Merge with existing trait matrix
            if trait_matrix is not None:
                trait_matrix = trait_matrix.combine_first(existing_df)
            else:
                trait_matrix = existing_df
    
    # If still empty, create zero matrix
    if trait_matrix is None or trait_matrix.empty:
        logger.warning("No traits detected. Creating zero matrix.")
        trait_matrix = pd.DataFrame(
            0.0,
            index=adata.var_names,
            columns=list(trait_db.list_traits())
        )
    
    # Ensure all OTUs are included (missing ones get 0)
    full_matrix = pd.DataFrame(
        0.0,
        index=adata.var_names,
        columns=trait_matrix.columns
    )
    full_matrix.update(trait_matrix)
    
    logger.info(f"Final trait matrix: {full_matrix.shape[0]} OTUs x {full_matrix.shape[1]} traits")
    
    return full_matrix, trait_db
