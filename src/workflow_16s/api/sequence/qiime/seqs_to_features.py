#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QIIME 2 Self-Contained Amplicon Analysis Workflow (Optimized & Robust)
Fully Checkpointed Version
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from qiime2 import Artifact, Metadata, Visualization
from qiime2.plugins import demux, taxa, feature_table, fragment_insertion
from qiime2.plugins.cutadapt.methods import trim_paired, trim_single
from qiime2.plugins.dada2.methods import denoise_paired, denoise_single
from qiime2.plugins.diversity.pipelines import core_metrics_phylogenetic
from qiime2.plugins.feature_classifier.methods import classify_sklearn
from qiime2.plugins.feature_table.methods import filter_features, filter_seqs
from qiime2.plugins.phylogeny.pipelines import align_to_tree_mafft_fasttree
from qiime2.plugins.greengenes2.actions import taxonomy_from_table, non_v4_16s
from qiime2.plugins.rescript.methods import orient_seqs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_N_THREADS = max(1, (os.cpu_count() or 1) - 1)
DEFAULT_MIN_FREQUENCY = 150
DEFAULT_COLLAPSE_LEVEL = 6

class QIIMEWorkflow:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.qiime_dir = Path(self.params['qiime_dir']).resolve()
        self._setup_file_registry()

    def run(self) -> None:
        self._check_inputs()
        self.qiime_dir.mkdir(parents=True, exist_ok=True)

        if self.params.get("hard_rerun", False):
            self._clean_qiime_dir()
        
        mode = self.params.get('dada2_mode', 'auto')
        if mode == 'inspect':
            self._run_inspection_workflow()
        else:
            self._run_execution_workflow(mode)

    def _clean_qiime_dir(self):
        for f in self.qiime_dir.glob("*.qza"): f.unlink(missing_ok=True)
        for f in self.qiime_dir.glob("*.qzv"): f.unlink(missing_ok=True)

    def _determine_resume_state(self) -> Tuple[str, Dict[str, Artifact]]:
        artifacts = {}
        stage = "IMPORT"
        if self.files["table"].exists() and self.files["rep_seqs"].exists():
            stage = "DADA2"
            artifacts["table"] = Artifact.load(self.files["table"])
            artifacts["rep_seqs"] = Artifact.load(self.files["rep_seqs"])
        if self.files["taxonomy"].exists():
            stage = "TAXONOMY"
            artifacts["taxonomy"] = Artifact.load(self.files["taxonomy"])
        return stage, artifacts

    def _run_inspection_workflow(self) -> None:
        logging.info("[INSPECT MODE] Starting DADA2 parameter inspection.")
        seqs = self._import_sequences()
        if self.params.get("trim_sequences", False):
            seqs = self._trim_sequences(seqs)
        
        filtered_seqs = self._filter_sequences(seqs)
        self._summarize_sequences(filtered_seqs, "03_filtered-summary")
        logging.info("[INSPECT MODE] Inspection complete.")

    def _run_execution_workflow(self, mode: str) -> None:
        logging.info(f"[{mode.upper()} MODE] Starting full QIIME 2 artifact generation.")
        
        resume_stage, loaded_artifacts = self._determine_resume_state()
        needs_export = not self._skip("table", "rep_seqs", "taxonomy")

        if resume_stage == "TAXONOMY" and not needs_export:
            logging.info("✅ Project verified. All artifacts and physical exports exist.")
            return
        
        if "table" not in loaded_artifacts or "rep_seqs" not in loaded_artifacts:
            seqs = self._import_sequences()
            if self.params.get("trim_sequences", False):
                seqs = self._trim_sequences(seqs)
            
            filtered_seqs = self._filter_sequences(seqs)

            if mode == 'auto':
                summary_key = "03_filtered-summary"
                if not self.files[summary_key].exists():
                    self._summarize_sequences(filtered_seqs, summary_key)
                dada2_params = self._estimate_dada2_params(summary_key)
                logging.info(f"Automatically determined DADA2 parameters: {dada2_params}")
            else: 
                dada2_params = self._get_locked_dada2_params()
            
            rep_seqs, table, stats = self._denoise_sequences(filtered_seqs, **dada2_params)
            
            logging.info("Filtering low-abundance features (singletons)...")
            try:
                table = filter_features(table=table, min_frequency=10, min_samples=2).filtered_table
                rep_seqs = filter_seqs(data=rep_seqs, table=table).filtered_data
            except ValueError:
                logging.warning("⚠️ Filtering resulted in an empty table. Proceeding with unfiltered data.")
            
            table.save(str(self.files["table"]))
            rep_seqs.save(str(self.files["rep_seqs"]))
        else:
            logging.info("⚡ [RESUME] DADA2 artifacts loaded from disk.")
            table = loaded_artifacts["table"]
            rep_seqs = loaded_artifacts["rep_seqs"]

        if "taxonomy" not in loaded_artifacts:
            taxonomy_strategy = self.params.get('taxonomy_strategy', 'gg2')
            if taxonomy_strategy == "gg2":
                taxonomy, table, rep_seqs = self._classify_taxonomy_gg2(table, rep_seqs)
            elif taxonomy_strategy == "sepp":
                taxonomy, table, _ = self._classify_taxonomy_sepp(table, rep_seqs)
            else:
                taxonomy = self._classify_taxonomy(rep_seqs)
        else:
            logging.info("⚡ [RESUME] Taxonomy artifact loaded from disk.")
            taxonomy = loaded_artifacts["taxonomy"]
            
        rooted_tree = None
        if self.params.get("diversity", False) and self.params.get('reference_tree_path'):
            if Path(self.params['reference_tree_path']).exists():
                rooted_tree = Artifact.load(self.params['reference_tree_path'])
                self._calculate_diversity(table, rooted_tree)
        
        self._collapse_table(table, taxonomy)
        self._export_final_artifacts(table, taxonomy, rep_seqs, tree=rooted_tree)
        logging.info(f"[{mode.upper()} MODE] Pipeline completed successfully.")

    def _verify_fastq_files(self, manifest_path: Path):
        import subprocess
        manifest = pd.read_csv(manifest_path, sep='\t')
        for col in ['forward-absolute-filepath', 'reverse-absolute-filepath']:
            if col in manifest.columns:
                for fpath in manifest[col].dropna():
                    try:
                        subprocess.run(["gzip", "-t", fpath], check=True, capture_output=True)
                    except subprocess.CalledProcessError:
                        logging.error(f"❌ Corrupted FASTQ detected: {fpath}")
                        raise EOFError(f"Corrupted FASTQ: {fpath}")
                    
    def _import_sequences(self) -> Artifact:
        if self._skip("seqs"): return Artifact.load(self.files['seqs'])
        self._verify_fastq_files(self.params['manifest_tsv'])
        logging.info("Importing sequences...")
        try:
            layout = self.params['library_layout']
            types = {
                "single": ("SampleData[SequencesWithQuality]", "SingleEndFastqManifestPhred33V2"),
                "paired": ("SampleData[PairedEndSequencesWithQuality]", "PairedEndFastqManifestPhred33V2"),
            }
            import_type, view_type = types[layout]
            seqs = Artifact.import_data(import_type, str(self.params['manifest_tsv']), view_type=view_type)
            seqs.save(str(self.files["seqs"]))
            return seqs
        except Exception as e:
            logging.error(f"Error importing sequences: {e}")
            raise RuntimeError(f"FAIL_IMPORT: {str(e)}")

    def _summarize_sequences(self, seqs: Artifact, base_name: str) -> None:
        if self._skip(base_name): return
        logging.info(f"Generating summary visualization for {base_name}...")
        summary_viz = demux.visualizers.summarize(data=seqs).visualization
        summary_viz.save(str(self.files[base_name]))

    def _trim_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("trimmed_seqs"): return Artifact.load(self.files['trimmed_seqs'])
        
        fwd_primer = str(self.params.get('fwd_primer_seq', "N/A"))
        rev_primer = str(self.params.get('rev_primer_seq', "N/A"))
        detected_adapters = self.params.get('detected_adapters', [])
        has_primers = fwd_primer.upper() not in ["N/A", "NONE", "NULL"]
        
        if not has_primers and not detected_adapters:
            logging.info("No biological primers or read-through adapters detected. Skipping Cutadapt.")
            seqs.save(str(self.files["trimmed_seqs"]))
            return seqs

        cores = self.params.get('n_threads', 1)
        is_paired = self.params['library_layout'] == 'paired'

        if has_primers:
            logging.info(f"Trimming biological primers (5'): {fwd_primer}")
            rev_primer_rc = self._reverse_complement(rev_primer)
            if is_paired:
                trimmed = trim_paired(demultiplexed_sequences=seqs, front_f=[fwd_primer], front_r=[rev_primer_rc], cores=cores).trimmed_sequences
            else:
                trimmed = trim_single(demultiplexed_sequences=seqs, front=[fwd_primer], cores=cores).trimmed_sequences
        else:
            logging.info(f"Initiating Cutadapt in 3' Scavenger Mode for adapters: {detected_adapters}...")
            adapter_seq_f = "CTGTCTCTTATACACATCT" if "Nextera" in detected_adapters else "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
            adapter_seq_r = "CTGTCTCTTATACACATCT" if "Nextera" in detected_adapters else "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
            if is_paired:
                trimmed = trim_paired(demultiplexed_sequences=seqs, adapter_f=[adapter_seq_f], adapter_r=[adapter_seq_r], cores=cores).trimmed_sequences
            else:
                trimmed = trim_single(demultiplexed_sequences=seqs, adapter=[adapter_seq_f], cores=cores).trimmed_sequences

        trimmed.save(str(self.files["trimmed_seqs"]))
        return trimmed

    def _filter_sequences(self, seqs: Artifact) -> Artifact:
        if self._skip("filtered_seqs"): return Artifact.load(self.files["filtered_seqs"])
        logging.info("Applying quality filtering...")
        seqs.save(str(self.files["filtered_seqs"]))
        return seqs

    def _run_dada2_engine(self, seqs, **params):
        is_paired = str(seqs.type) == 'SampleData[PairedEndSequencesWithQuality]'
        n_threads = self.params.get('n_threads', 16)
    
        if is_paired:
            logging.info("🧬 Running DADA2 Paired-End Denoising...")
            res = denoise_paired(demultiplexed_seqs=seqs, n_threads=n_threads, **params)
        else:
            logging.info("🧬 Running DADA2 Single-End Denoising...")
            single_params = {k: v for k, v in params.items() if '_r' not in k and '_reverse' not in k}
            # Map paired-end names to single-end names for DADA2 compatibility
            single_compat_params = {
                'trim_left': params.get('trim_left_f', 0),
                'trunc_len': params.get('trunc_len_f', 0)
            }
            # Remove the paired-end specific keys if they exist
            for k in ['trim_left_f', 'trunc_len_f', 'trim_left_r', 'trunc_len_r']:
                single_params.pop(k, None)
            
            single_params.update(single_compat_params)
            res = denoise_single(demultiplexed_seqs=seqs, n_threads=n_threads, **single_params)
        
        return res.representative_sequences, res.table, res.denoising_stats
    
    def _denoise_sequences(self, seqs, **params):
        try:
            return self._run_dada2_engine(seqs, **params)
        except Exception as e:
            raise RuntimeError(f"FAIL_DENOISE: {str(e)}")

    def _classify_taxonomy_gg2(self, table: Artifact, rep_seqs: Artifact) -> Tuple[Artifact, Artifact, Artifact]:
        if self._skip("taxonomy"): 
            return (Artifact.load(self.files["taxonomy"]), table, rep_seqs)
        
        logging.info("Deriving Greengenes2 taxonomy directly from the phylogenetic tree branches...")
        try:
            backbone = Artifact.load(self.params['backbone_path'])
            ref_tax = Artifact.load(self.params['reference_taxonomy_path'])
            
            logging.info("🔄 Orienting sequences against Greengenes2 backbone...")
            oriented_results = orient_seqs(sequences=rep_seqs, reference_sequences=backbone)
            rep_seqs = oriented_results.oriented_seqs
            
            logging.info("🔍 Synchronizing feature table with sequences...")
            # Convert to DataFrame and ensure column names are strings for QIIME 2
            seq_df = rep_seqs.view(pd.Series).to_frame()
            seq_df.columns = ['sequence']  # Explicitly set column name to be string
            seq_df.index.name = 'feature-id'  # QIIME 2 metadata requires index to be feature IDs
            filter_res = filter_features(table=table, metadata=Metadata(seq_df))
            table = filter_res.filtered_table
            
            logging.info("Mapping to Greengenes2 backbone...")
            mapping_results = non_v4_16s(
                table=table, 
                sequences=rep_seqs, 
                backbone=backbone,
                threads=self.params.get('n_threads', 16)
            )
            
            try:
                mapped_table = mapping_results.mapped_table
                mapped_seqs = mapping_results.representatives
            except AttributeError:
                sync_results = filter_features(
                    table=mapping_results.mapped_table,
                    metadata=Metadata(mapping_results.mapped_seqs.view(pd.Series).to_frame())
                )
                mapped_table = sync_results.filtered_table
                mapped_seqs = mapping_results.mapped_seqs
            
            logging.info("Deriving Greengenes2 taxonomy from mapped table...")
            taxonomy = taxonomy_from_table(reference_taxonomy=ref_tax, table=mapped_table).classification
            
            taxonomy.save(str(self.files["taxonomy"]))
            taxonomy.export_data(self.qiime_dir)
            
            return taxonomy, mapped_table, mapped_seqs

        except AttributeError as e:
            if "str accessor" in str(e):
                logging.error(f"Metadata column type error - ensure all DataFrame columns are strings: {e}")
            else:
                logging.error(f"Error in Greengenes2 taxonomy classification: {e}")
            raise RuntimeError(f"FAIL_TAXONOMY: {str(e)}")
        except Exception as e:
            logging.error(f"Error in Greengenes2 taxonomy classification: {e}")
            raise RuntimeError(f"FAIL_TAXONOMY: {str(e)}")

    def _classify_taxonomy_sepp(self, table: Artifact, rep_seqs: Artifact) -> Tuple[Artifact, Artifact, Artifact]:
        logging.info("Starting SEPP Fragment Insertion...")
        insertion_results = fragment_insertion.methods.sepp(
            representative_sequences=rep_seqs,
            reference_alignment=self.params['backbone_path'],
            reference_phylogeny=self.params['reference_tree_path'],
            threads=self.params.get('n_threads', 16),
            debug=False
        )
        placed_tree = insertion_results.tree
        placements = insertion_results.placements
        taxonomy_results = fragment_insertion.methods.classify_otus_experimental(
            taxonomy=self.params['reference_taxonomy_path'], view=placements
        )
        filter_results = fragment_insertion.methods.filter_features(table=table, tree=placed_tree)
        taxonomy_results.classification.save(str(self.files["taxonomy"]))
        placed_tree.save(str(self.files["rooted_tree"]))
        return taxonomy_results.classification, filter_results.filtered_table, placed_tree

    def _classify_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        if self._skip("taxonomy"): return Artifact.load(self.files["taxonomy"])
        logging.info("Classifying taxonomy...")
        classifier = Artifact.load(self.params['classifier_path'])
        safe_jobs = min(self.params.get('n_threads', 16), 12) 
        
        taxonomy = classify_sklearn(
            reads=rep_seqs, classifier=classifier, confidence=self.params.get('confidence', 0.7), 
            n_jobs=safe_jobs, reads_per_batch=20000, pre_dispatch='2*n_jobs'
        ).classification
        
        taxonomy.save(str(self.files["taxonomy"]))
        return taxonomy
    
    def _calculate_diversity(self, table: Artifact, rooted_tree: Artifact) -> None:
        if self._skip("core_metrics"): return
        logging.info("Diversity metrics calculation skipped or manually disabled.")

    def _collapse_table(self, table: Artifact, taxonomy: Artifact) -> Artifact:
        if self._skip("collapsed_table"): return Artifact.load(self.files["collapsed_table"])
        level = self.params.get('collapse_level', DEFAULT_COLLAPSE_LEVEL)
        logging.info(f"Collapsing feature table to level {level}...")
        collapsed = taxa.actions.collapse(table=table, taxonomy=taxonomy, level=level).collapsed_table
        collapsed.save(str(self.files["collapsed_table"]))
        return collapsed
    
    def _export_final_artifacts(self, table: Artifact, taxonomy: Artifact, rep_seqs: Artifact, tree: Artifact = None):
        logging.info("Exporting final artifacts to plain formats...")
        export_map = {
            "exported_biom": (table, "feature-table.biom"),
            "exported_taxonomy": (taxonomy, "taxonomy.tsv"),
            "exported_rep_seqs": (rep_seqs, "dna-sequences.fasta")
        }
        if tree:
            export_map["exported_tree"] = (tree, "tree.nwk")

        for key, (artifact, expected_filename) in export_map.items():
            if self._skip(key): continue
            try:
                artifact.export_data(self.qiime_dir)
            except Exception as e:
                logging.error(f"Export error for {expected_filename}: {e}")

    def _extract_file_from_qzv(self, qzv_path: Path, target_filename: str, reader_func):
        try:
            with zipfile.ZipFile(qzv_path, 'r') as z:
                target_path = next((name for name in z.namelist() if name.endswith(target_filename) and not name.startswith('__MACOSX')), None)
                if not target_path: return None
                with z.open(target_path) as f:
                    return reader_func(f)
        except Exception as e:
            logging.warning(f"Error reading artifact {qzv_path}: {e}")
            return None

    def _estimate_dada2_params(self, summary_key: str) -> Dict[str, int]:
        is_paired = self.params['library_layout'] == 'paired'
        fwd_primer = str(self.params.get('fwd_primer_seq', ""))
        rev_primer = str(self.params.get('rev_primer_seq', ""))
        
        trim_f = len(fwd_primer) if self.params.get("trim_sequences") and fwd_primer.upper() not in ["N/A", "NONE", "NULL"] else 0
        trim_r = len(rev_primer) if is_paired and self.params.get("trim_sequences") and rev_primer.upper() not in ["N/A", "NONE", "NULL"] else 0

        summary_qzv_path = self.files[summary_key]
        
        try:
            df_fwd = self._extract_file_from_qzv(summary_qzv_path, "forward-seven-number-summary.csv", lambda f: pd.read_csv(f, index_col=0))
        except Exception:
            df_fwd = self._extract_file_from_qzv(summary_qzv_path, "forward-seven-number-summary.tsv", lambda f: pd.read_csv(f, sep='\t', index_col=0))
        
        df_rev = None
        if is_paired:
            try:
                df_rev = self._extract_file_from_qzv(summary_qzv_path, "reverse-seven-number-summary.csv", lambda f: pd.read_csv(f, index_col=0))
            except Exception:
                df_rev = self._extract_file_from_qzv(summary_qzv_path, "reverse-seven-number-summary.tsv", lambda f: pd.read_csv(f, sep='\t', index_col=0))

        if df_fwd is None or (is_paired and df_rev is None):
            logging.warning("Quality data missing from artifact. Using safe defaults.")
            return self._get_safe_defaults(is_paired)

        def find_trunc_pos(df, threshold=25):
            bad_qual_indices = df[df['50%'] < threshold].index
            if len(bad_qual_indices) > 0: return int(bad_qual_indices[0])
            return int(df.index.max())

        trunc_f = find_trunc_pos(df_fwd)
        trunc_r = find_trunc_pos(df_rev) if is_paired else 0

        if is_paired:
            TARGET_AMPLICON_SIZE = self.params.get('expected_amplicon_size') or 253
            REQUIRED_TOTAL_LEN = TARGET_AMPLICON_SIZE + 20
            
            effective_f = trunc_f - trim_f
            effective_r = trunc_r - trim_r
            total_len = effective_f + effective_r
            
            if total_len < REQUIRED_TOTAL_LEN:
                logging.warning(f"Calculated truncation ({trunc_f}, {trunc_r}) risks poor overlap.")
                shortfall = REQUIRED_TOTAL_LEN - total_len
                max_f = int(df_fwd.index.max())
                max_r = int(df_rev.index.max())
                trunc_f = min(max_f, trunc_f + (shortfall // 2) + 5)
                trunc_r = min(max_r, trunc_r + (shortfall // 2) + 5)
                logging.warning(f"Adjusted to ({trunc_f}, {trunc_r}) to enforce overlap.")

        params = {'trim_left_f': trim_f, 'trunc_len_f': trunc_f}
        if is_paired:
            params.update({'trim_left_r': trim_r, 'trunc_len_r': trunc_r})
            
        return params

    def _get_safe_defaults(self, is_paired: bool) -> Dict[str, int]:
        defaults = {'trim_left_f': 0, 'trunc_len_f': 220} 
        if is_paired: defaults.update({'trim_left_r': 0, 'trunc_len_r': 180}) 
        return defaults

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        if not seq: return ""
        complement_map = str.maketrans("ATGCatgcRYrySWswKMkmBDHVbdhvNn", "TACGtacgYRyrWSwsMKmkVHDBvhdbNn")
        return seq.translate(complement_map)[::-1]

    def _get_locked_dada2_params(self) -> Dict[str, int]:
        user_params = self.params.get('dada2_params', [])
        if not user_params: return self._get_safe_defaults(self.params['library_layout'] == 'paired')
        is_paired = self.params['library_layout'] == 'paired'
        try:
            if is_paired:
                return {'trunc_len_f': user_params[0], 'trunc_len_r': user_params[1], 'trim_left_f': user_params[2], 'trim_left_r': user_params[3]}
            else:
                return {'trunc_len_f': user_params[0], 'trim_left_f': user_params[1]}
        except IndexError:
            logging.warning("Invalid DADA2 params length. Using safe defaults.")
            return self._get_safe_defaults(is_paired)

    def _setup_file_registry(self) -> None:
        d = self.qiime_dir
        level = self.params.get('collapse_level', DEFAULT_COLLAPSE_LEVEL)
        self.files: Dict[str, Path] = {
            "seqs": d / "01_demux-sequences.qza",
            "01_imported-summary": d / "01_imported-summary.qzv",
            "trimmed_seqs": d / "02_trimmed-sequences.qza",
            "02_trimmed-summary": d / "02_trimmed-summary.qzv",
            "filtered_seqs": d / "03_filtered-sequences.qza",
            "03_filtered-summary": d / "03_filtered-summary.qzv",
            "rep_seqs": d / "04_representative-sequences.qza",
            "table": d / "04_feature-table.qza",
            "stats": d / "04_denoising-stats.qza",
            "collapsed_table": d / f"07_collapsed-table-L{level}.qza",
            "exported_biom": d / "feature-table.biom",
            "exported_rep_seqs": d / "dna-sequences.fasta",
        }
        
        taxonomy_strategy = self.params.get('taxonomy_strategy', 'gg2')
        if taxonomy_strategy == "gg2":
            self.files.update({
                "taxonomy": d / "05_taxonomy.qza",
                "exported_taxonomy": d / "taxonomy.tsv",
                "rooted_tree": d / "06_rooted-tree.qza",
                "exported_tree": d / "tree.nwk"
            })
        elif taxonomy_strategy == "sklearn":
            self.files.update({
                "taxonomy": d / "05_taxonomy_denovo.qza",
                "exported_taxonomy": d / "taxonomy_denovo.tsv",
                "rooted_tree": d / "06_rooted-tree_denovo.qza",
                "exported_tree": d / "tree_denovo.nwk"
            })
        elif taxonomy_strategy == "sepp":
            self.files.update({
                "taxonomy": d / "05_taxonomy_sepp.qza",
                "exported_taxonomy": d / "taxonomy_sepp.tsv",
                "rooted_tree": d / "06_rooted-tree_sepp.qza",
                "exported_tree": d / "tree_sepp.nwk"
            })
            
    def _skip(self, *keys: str) -> bool:
        if self.params.get("hard_rerun", False): return False
        export_targets = {
            "table": "feature-table.biom",
            "rep_seqs": "dna-sequences.fasta",
            "taxonomy": "taxonomy.tsv",
            "rooted_tree": "tree.nwk",
            "collapsed_table": f"07_collapsed-table-L{self.params.get('collapse_level', 6)}.qza"
        }
        for key in keys:
            qza_path = self.files.get(key)
            if not qza_path or not qza_path.exists(): return False
            if key in export_targets:
                target_file = self.qiime_dir / export_targets[key]
                if not target_file.exists(): return False
        return True

    def _check_inputs(self) -> None:
        manifest = self.params.get('manifest_tsv')
        strategy = self.params.get('taxonomy_strategy', 'gg2')
        
        if not manifest or not Path(manifest).is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest}")
            
        if self.params.get('dada2_mode') != 'inspect':
            if strategy == 'sklearn':
                classifier = self.params.get('classifier_path')
                if not classifier or not Path(classifier).is_file():
                    raise FileNotFoundError(f"Classifier file not found: {classifier}")
            elif strategy in ['sepp', 'gg2']:
                back = self.params.get('backbone_path')
                ref_tax = self.params.get('reference_taxonomy_path')
                if not back or not Path(back).is_file():
                    raise FileNotFoundError(f"Backbone file not found: {back}")
                if not ref_tax or not Path(ref_tax).is_file():
                    raise FileNotFoundError(f"Reference taxonomy file not found: {ref_tax}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIIME 2 Amplicon Workflow")
    parser.add_argument("--dada2-mode", default="auto")
    parser.add_argument("--expected_amplicon_size", default="253")
    parser.add_argument("--qiime_dir", required=True)
    parser.add_argument("--metadata_tsv", required=True)
    parser.add_argument("--manifest_tsv", required=True)
    parser.add_argument("--library_layout", required=True)
    parser.add_argument("--fwd_primer_seq", default="N/A")
    parser.add_argument("--rev_primer_seq", default="N/A")
    parser.add_argument("--detected_adapters", nargs='*', default=[])
    parser.add_argument("--chimera_method", default="consensus")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--n_threads", type=int, default=16)
    parser.add_argument("--min_frequency", type=int, default=150)
    parser.add_argument("--collapse_level", type=int, default=6)
    parser.add_argument("--taxonomy_strategy", default="gg2")
    parser.add_argument("--backbone_path", default=None)
    parser.add_argument("--reference_taxonomy_path", default=None)
    parser.add_argument("--reference_tree_path", default=None)
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--dada2-params", nargs='*', type=int, default=[])
    parser.add_argument("--hard_rerun", action="store_true")
    parser.add_argument("--trim_sequences", action="store_true")
    parser.add_argument("--diversity", action="store_true")

    args = parser.parse_args()

    if str(args.expected_amplicon_size).lower() == "none":
        args.expected_amplicon_size = None
    elif args.expected_amplicon_size is not None:
        args.expected_amplicon_size = int(args.expected_amplicon_size)

    workflow = QIIMEWorkflow(**vars(args))
    workflow.run()
