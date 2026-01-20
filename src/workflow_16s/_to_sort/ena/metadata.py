# ==================================== IMPORTS ======================================= #

from typing import Dict, List, Optional, Set, Tuple, Any
import re
import pandas as pd
from functools import partial
from Bio import Entrez

import logging

from workflow_16s.ena.api import MetadataFetcher
import workflow_16s.custom_tmp_config

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_EMAIL = "macgregor@berkeley.edu"
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

# ==================================== FUNCTIONS ===================================== #

class ENAMetadata:
    """Process ENA datasets with metadata parsing and taxonomic analysis."""
    # Class constants
    COLUMNS_TO_DISPLAY = [
        "sequencing_tech",
        "library_type",
        "sample_type",
        "target_gene",
        "primary_taxonomy",
        "environmental_keywords",
    ]
    FILTER_CRITERIA = {
        "sequencing_tech": "illumina",
        "library_type": "amplicon",
        "sample_type": "environmental",
        "min_confidence": 0.7,
    }
    ENV_KEYWORD_PATTERN = re.compile(
        r"\b(soil|water|aquatic|sediment|environmental|metagenom)\w*\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, email: str = None):
        self.email = email or DEFAULT_EMAIL
        self.tax_cache: Dict[int, Optional[str]] = {}
        self.analysis_rules = self._compile_rules()
        self.fetcher = MetadataFetcher()

        self.df: pd.DataFrame() = None
        self.parsed_df: pd.DataFrame() = None
        self.characteristics = None
        self.runs = None
        self.diagnostics = Dict = None

    @staticmethod
    def _parse_ncbi_taxid(tax_id: str) -> List[int]:
        """Extract unique NCBI taxonomy IDs from complex strings."""
        if pd.isna(tax_id) or not str(tax_id).strip():
            return []
        return list({int(tid) for tid in re.findall(r"\d+", str(tax_id))})

    def _get_taxonomy_string(self, tax_id: int) -> Optional[str]:
        """Retrieve taxonomic lineage from NCBI Taxonomy database."""
        try:
            Entrez.email = self.email
            with Entrez.efetch(db="taxonomy", id=str(tax_id), retmode="xml") as handle:
                record = Entrez.read(handle)
                lineage = record[0]["LineageEx"]
                return "; ".join([taxon["ScientificName"] for taxon in lineage])
        except Exception as e:
            logger.warning(f"Failed to fetch taxonomy for {tax_id}: {str(e)}")
            return None

    def _compile_rules(
        self,
    ) -> Dict[str, Dict[str, List[Tuple[re.Pattern, re.Pattern, float]]]]:
        """Precompile regex patterns for metadata analysis rules."""
        rule_definitions = {
            "sample_type": {
                "human": [
                    (r".*", r"\b(homo\s+sapiens|human)\b", 0.9),
                    (r".*", r"\b(blood|saliva|stool)\b", 0.7),
                ],
                "microbial": [
                    (r".*", r"\b(bacteria|archaea|fungi|virus)\b", 0.8),
                    (r".*", r"\b(strain|isolate|culture)\b", 0.6),
                ],
                "environmental": [
                    (r".*", r"\b(soil|water|air|sediment|environmental)\b", 0.8),
                    (r".*", r"\bmetagenom", 0.9),
                ],
            },
            "sequencing_tech": {
                "illumina": [
                    (r".*", r"\b(Illumina|MiSeq|HiSeq|NovaSeq|NextSeq)\b", 0.95),
                    (r".*", r"\b(sequencing\s+by\s+synthesis|SBS)\b", 0.7),
                ],
                "nanopore": [
                    (r".*", r"\b(Nanopore|MinION|GridION|PromethION)\b", 0.95),
                    (r".*", r"\b(Oxford\s+Nanopore)\b", 0.9),
                ],
                "pacbio": [
                    (r".*", r"\b(PacBio|SMRT|Sequel)\b", 0.95),
                    (r".*", r"\b(single\s+molecule\s+real-time)\b", 0.8),
                ],
            },
            "library_type": {
                "amplicon": [
                    (r".*", r"\b(amplicon|16S|ITS|PCR)\b", 0.95),
                    (r".*", r"\b(V[1-9]-V[1-9]|hypervariable)\b", 0.8),
                ],
                "wgs": [
                    (r".*", r"\b(WGS|whole\s+genome)\b", 0.95),
                    (r".*", r"\b(shotgun|metagenomic)\b", 0.8),
                ],
                "rna": [
                    (r".*", r"\b(RNA|transcriptom)\b", 0.95),
                    (r".*", r"\b(mRNA|ribo-depletion)\b", 0.8),
                ],
            },
            "target_gene": {
                "16S": [
                    (r".*", r"\b16S\b", 0.95),
                    (r".*", r"\b(v[1-9]-v[1-9]|515f|806r)\b", 0.8),
                ],
                "18S": [(r".*", r"\b18S\b", 0.95)],
                "23S": [(r".*", r"\b23S\b", 0.95)],
                "ITS": [
                    (r".*", r"\bITS\b", 0.95),
                    (r".*", r"\b(ITS1|ITS2|ITS4)\b", 0.8),
                ],
            },
        }

        compiled_rules = {}
        for category, types in rule_definitions.items():
            compiled_rules[category] = {}
            for type_name, rules in types.items():
                compiled = [
                    (
                        re.compile(col_pat, re.IGNORECASE),
                        re.compile(val_pat, re.IGNORECASE),
                        weight,
                    )
                    for col_pat, val_pat, weight in rules
                ]
                compiled_rules[category][type_name] = compiled
        return compiled_rules

    def _parse_taxonomy(self, row: pd.Series) -> Dict:
        """Process taxonomy data with environmental keyword detection."""
        tax_data = {
            "tax_ids": [],
            "taxonomy_strings": [],
            "primary_taxid": None,
            "primary_taxonomy": None,
            "environmental_keywords": set(),
        }

        tax_id = row.get("tax_id")
        if not pd.isna(tax_id) and tax_id:
            tax_ids = self._parse_ncbi_taxid(tax_id)
            if tax_ids:
                tax_data["tax_ids"] = tax_ids
                tax_data["primary_taxid"] = tax_ids[0]

                for tid in tax_ids:
                    if tid not in self.tax_cache:
                        self.tax_cache[tid] = self._get_taxonomy_string(tid)
                    tax_str = self.tax_cache[tid] or ""
                    tax_data["taxonomy_strings"].append(tax_str)

                    matches = self.ENV_KEYWORD_PATTERN.findall(tax_str)
                    normalized_matches = {m.lower() for m in matches}
                    tax_data["environmental_keywords"].update(normalized_matches)
                tax_data["primary_taxonomy"] = self.tax_cache.get(tax_ids[0], "")
                tax_data["environmental_keywords"] = sorted(
                    tax_data["environmental_keywords"]
                )
        return tax_data

    def _analyze_category(self, row: pd.Series, category: str) -> Dict:
        """Analyze row for a specific category using precompiled regex patterns."""
        best_result = {"type": "unknown", "confidence": 0.0, "evidence": []}
        str_row = {
            col: str(val) if not pd.isna(val) else "" for col, val in row.items()
        }

        for type_name, rules in self.analysis_rules[category].items():
            total_score = max_possible = 0.0
            evidence = []

            for col_re, val_re, weight in rules:
                max_possible += weight
                for col, val_str in str_row.items():
                    if col_re.fullmatch(col) and val_re.search(val_str):
                        total_score += weight
                        evidence.append(f"{col}={val_str}")
                        break  # One match per rule
            if max_possible > 0:
                confidence = total_score / max_possible
                if confidence > best_result["confidence"] and confidence >= 0.5:
                    best_result = {
                        "type": type_name,
                        "confidence": round(confidence, 2),
                        "evidence": evidence,
                    }
        return best_result

    def parse_sample(self, row: pd.Series) -> Optional[Dict]:
        """Process a single sample row with enhanced error handling."""
        try:
            sample_acc = row.get("sample_accession")
            if pd.isna(sample_acc) or sample_acc == "":
                raise ValueError("Missing sample_accession")
            taxonomy = self._parse_taxonomy(row)
            enhanced_row = row.copy()
            enhanced_row["parsed_taxonomy"] = taxonomy.get("primary_taxonomy", "")

            results = {
                "run_accession": enhanced_row.get("run_accession"),
                "sample_accession": sample_acc,
                "taxonomy": taxonomy,
                "raw_metadata": enhanced_row.to_dict(),
            }

            for category in self.analysis_rules:
                results[category] = self._analyze_category(enhanced_row, category)
            return results
        except Exception as e:
            logger.error(
                f"Error processing {row.get('sample_accession', 'unknown')}: {str(e)}"
            )
            return None

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame with vectorized operations and result consolidation."""
        analyses = df.apply(self.parse_sample, axis=1).dropna()
        if analyses.empty:
            return pd.DataFrame()
        processed = []
        for analysis in analyses:
            tax = analysis["taxonomy"]
            entry = {
                "run_accession": analysis["run_accession"],
                "sample_accession": analysis["sample_accession"],
                **{cat: analysis[cat]["type"] for cat in self.analysis_rules},
                "primary_taxid": tax.get("primary_taxid"),
                "primary_taxonomy": tax.get("primary_taxonomy"),
                "environmental_keywords": ", ".join(
                    tax.get("environmental_keywords", [])
                ),
                "confidence_score": analysis["sample_type"]["confidence"],
            }
            processed.append(entry)
        return pd.DataFrame(processed).set_index("sample_accession")

    @staticmethod
    def _list_of_tuples_format(items: List[Tuple[str, int]]) -> str:
        """Format value-count tuples into standardized string representation."""
        return "; ".join(
            f"{k} ({v})" for k, v in sorted(items, key=lambda x: (-x[1], x[0]))
        )

    def _get_filtered_runs(self, df: pd.DataFrame, target_gene_value: str) -> Set[str]:
        """Filter runs using configurable criteria with confidence threshold."""
        mask = (
            (df["sequencing_tech"] == self.FILTER_CRITERIA["sequencing_tech"])
            & (df["library_type"] == self.FILTER_CRITERIA["library_type"])
            & (df["sample_type"] == self.FILTER_CRITERIA["sample_type"])
            & (df["target_gene"] == target_gene_value)
            & (df["confidence_score"] >= self.FILTER_CRITERIA["min_confidence"])
        )
        return set(df.loc[mask, "run_accession"].dropna().unique())

    def _library_layout_from_fastq_ftp(
        self, metadata: pd.DataFrame, info: Dict
    ) -> pd.DataFrame:
        # Handle NaN/empty values in 'fastq_ftp'
        metadata["fastq_ftp"] = metadata["fastq_ftp"].fillna("")

        # Calculate URL counts safely
        url_counts = [
            len([url for url in ftp_urls.strip().split(";") if url.strip() != ""])
            for ftp_urls in metadata["fastq_ftp"]
        ]

        # Determine library layout
        library_layout = [
            "paired" if count == 2 else "single" if count == 1 else "unknown"
            for count in url_counts
        ]

        # Convert to a pandas Series for alignment
        new_layout = pd.Series(
            library_layout, index=metadata.index, name="library_layout"
        )

        # Check for mismatches using case-insensitive comparison
        original_lower = metadata["library_layout"].str.lower()
        if not original_lower.equals(new_layout):
            # Log differences
            mismatches = metadata[original_lower != new_layout]
            if len(mismatches) > 0:
                logger.debug(
                    f"\nLibrary layout mismatch in {len(mismatches)} rows.\n"
                    f"Differences:\n{mismatches[['library_layout']].join(new_layout.rename('new_layout'))}"
                )
            # Update the DataFrame
            metadata["library_layout"] = new_layout
        return metadata

    def process_dataset(self, dataset: str, info: Dict) -> Optional[Dict[str, Any]]:
        """Process ENA dataset metadata with comprehensive validation and filtering."""
        if not ENA_PATTERN.match(dataset):
            logger.error(f"Invalid ENA accession format: {dataset}")
            return None
        try:
            df = self.fetcher.get_study_and_sample_metadata(dataset)
            if df.empty:
                logger.warning(f"Empty metadata returned for {dataset}")
                return None
            df = df.dropna(axis=1, how="all")
            for key, value in info.items():
                df[key] = value
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.convert_dtypes()

            parsed_df = self.analyze_dataframe(df)

            characteristics = {
                col: dict(parsed_df[col].value_counts().items())
                for col in self.COLUMNS_TO_DISPLAY
            }

            target_genes = parsed_df["target_gene"].unique().tolist() + ["unknown"]
            runs = {
                gene: self._get_filtered_runs(parsed_df, gene)
                for gene in target_genes
                if gene in parsed_df["target_gene"].values or gene == "unknown"
            }

            df = self._library_layout_from_fastq_ftp(df, info)
            

            self.df = df
            self.parsed_df = parsed_df
            self.characteristics = characteristics
            self.runs = runs
            self.diagnostics = {
                "accession": dataset,
                "processed_at": pd.Timestamp.now(),
                "record_count": len(self.df),
                "filter_criteria": self.FILTER_CRITERIA,
            }

            return 1
        except Exception as e:
            logger.error(
                f"Critical failure processing {dataset}: {str(e)}", exc_info=True
            )
            self.df = pd.DataFrame()
            return None
