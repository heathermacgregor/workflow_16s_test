# workflow_16s/api/qiime/execute_seqs_to_features.py

import asyncio
import shlex
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, List

import biom
import re
from workflow_16s.config import AppConfig
from workflow_16s.utils.logger import get_logger


def _pick_first(mapping: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for key in candidates:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _validate_and_check_artifacts(artifact_paths: Dict[str, Path]) -> bool:
    """
    Checks if all artifact files exist, are non-empty, and are not corrupted.
    """
    logger = get_logger("workflow_16s")
    logger.debug("Checking for existing and valid QIIME 2 artifacts...")
    for name, path in artifact_paths.items():
        if not path.exists():
            logger.info(f"Artifact check failed: '{name}' does not exist at {path}.")
            return False

        if path.stat().st_size == 0:
            logger.info(f"Artifact check failed: '{name}' is an empty file at {path}.")
            return False

        try:
            if path.suffix == ".qza":
                if not zipfile.is_zipfile(path):
                    logger.info(f"Artifact check failed: '{name}' is not a valid zip file.")
                    return False
                with zipfile.ZipFile(path, 'r') as zf:
                    if zf.testzip() is not None:
                        logger.info(f"Artifact check failed: '{name}' has a bad CRC checksum.")
                        return False
            elif path.suffix == ".biom":
                biom.load_table(path)
        except Exception as e:
            logger.warning(f"Artifact check failed for '{name}' at {path}: {e}")
            return False
            
    logger.info("✅ All required QIIME 2 artifacts exist and appear valid.")
    return True

async def get_conda_env_path(env_name_substring: str) -> str:
    """Finds the full path to a Conda environment asynchronously."""
    try:
        process = await asyncio.create_subprocess_exec(
            "conda", "env", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        for line in stdout.decode().splitlines():
            if line.startswith('#') or not line.strip():
                continue
            if env_name_substring in line:
                return line.split()[-1]
        raise ValueError(f"Conda environment with substring '{env_name_substring}' not found.")
    except Exception as e:
        raise RuntimeError("Could not execute 'conda env list'.") from e

async def execute_per_dataset_qiime_workflow(
    app_config: AppConfig,
    subset: Dict[str, Any],
    qiime_dir: Path,
    metadata_path: Path,
    manifest_path: Path,
    anndata_dir: Path,
    subset_id: str,
    expected_amplicon_size: Optional[int] = None,
    progress_obj: Any = None,
    detected_adapters: Optional[List[str]] = None
) -> Dict[str, Path]:
    logger = get_logger("workflow_16s")
    q2_config = app_config.qiime2.per_dataset
    level = q2_config.collapse_level
    
    # Validate taxonomy strategy early with enum check
    taxonomy_strategy = q2_config.taxonomy.classify_method
    valid_strategies = {"sepp", "sklearn", "gg2"}
    if taxonomy_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid taxonomy_strategy '{taxonomy_strategy}'. "
            f"Must be one of: {', '.join(sorted(valid_strategies))}"
        )
    logger.info(f"Using taxonomy classification method: {taxonomy_strategy}")
    
    # Validate script path exists before subprocess
    script_path = Path(q2_config.script_path)
    if not script_path.exists():
        raise FileNotFoundError(
            f"QIIME2 script not found at {script_path}. "
            f"Check config paths.qiime2.per_dataset.script_path"
        )
    logger.debug(f"QIIME2 script validated at: {script_path}")
    
    if taxonomy_strategy == "sepp":
        logger.info("SEPP-based classification selected. Ensuring reference tree and taxonomy paths are set.")
        backbone_path = app_config.paths.backbone
        ref_tree_path = app_config.paths.reference_tree 
        ref_tax_path = app_config.paths.reference_taxonomy 
    
    elif taxonomy_strategy == "sklearn":
        logger.info("SKLearn-based classification selected. Ensuring classifier path is set.")
        classifier_path = app_config.paths.classifier 
    
    elif taxonomy_strategy == "gg2":
        logger.info("Greengenes 2 classification selected. Ensuring reference tree and taxonomy paths are set.")
        backbone_path = app_config.paths.backbone
        ref_tree_path = app_config.paths.reference_tree
        ref_tax_path = app_config.paths.reference_taxonomy
    
    final_artifact_paths = {
        "table_qza": qiime_dir / "04_feature-table.qza",
        "rep_seqs_qza": qiime_dir / "04_representative-sequences.qza",
        "collapsed_table_qza": qiime_dir / f"07_collapsed-table-L{level}.qza",
        "feature_table_biom": qiime_dir / "feature-table.biom",
        "rep_seqs_fasta": qiime_dir / "dna-sequences.fasta",
    }
    if taxonomy_strategy == "sepp":
        final_artifact_paths.update({
            "taxonomy_qza": qiime_dir / "05_taxonomy_sepp.qza",
            "taxonomy_tsv": qiime_dir / "taxonomy_sepp.tsv",
            "rooted_tree_qza": qiime_dir / "06_rooted-tree_sepp.qza",
            "rooted_tree_nwk": qiime_dir / "tree_sepp.nwk"
        })
    elif taxonomy_strategy == "sklearn":
        final_artifact_paths.update({
            "taxonomy_qza": qiime_dir / "05_taxonomy_denovo.qza",
            "taxonomy_tsv": qiime_dir / "taxonomy_denovo.tsv",
            "rooted_tree_qza": qiime_dir / "06_rooted-tree_denovo.qza",
            "rooted_tree_nwk": qiime_dir / "tree_denovo.nwk"
        })
    elif taxonomy_strategy == "gg2":
        final_artifact_paths.update({
            "taxonomy_qza": qiime_dir / "05_taxonomy.qza",
            "taxonomy_tsv": qiime_dir / "taxonomy.tsv",
            "rooted_tree_qza": qiime_dir / "06_rooted-tree.qza",
            "rooted_tree_nwk": qiime_dir / "tree.nwk"
        })

    if not q2_config.hard_rerun and _validate_and_check_artifacts(final_artifact_paths):
        logger.info(f"Skipping QIIME 2 for {subset_id}: valid outputs found.")
        return final_artifact_paths

    # Resolve conda environment: try config value, fall back to hardcoded default
    default_conda_env = "qiime2-amplicon-2025"
    conda_env_substring = getattr(
        app_config.qiime2, "conda_env_name", default_conda_env
    )
    if not conda_env_substring:
        conda_env_substring = default_conda_env
    
    try:
        conda_env_path = await get_conda_env_path(conda_env_substring)
        logger.info(f"Resolved conda environment: {conda_env_path}")
    except (ValueError, RuntimeError) as e:
        logger.warning(
            f"Could not resolve conda env '{conda_env_substring}'. "
            f"Will attempt to use default '{default_conda_env}' as fallback."
        )
        try:
            conda_env_path = await get_conda_env_path(default_conda_env)
            logger.info(f"Using fallback conda environment: {conda_env_path}")
        except (ValueError, RuntimeError) as fallback_error:
            raise RuntimeError(
                f"Could not locate QIIME2 conda environment. "
                f"Tried '{conda_env_substring}' and fallback '{default_conda_env}'. "
                f"Ensure one is installed and visible to 'conda env list'."
            ) from fallback_error
    dada2_mode = getattr(q2_config.denoise, 'dada2_mode', 'auto')
    
    library_layout = _pick_first(subset, ["library_layout", "layout"])
    if library_layout is None:
        raise KeyError(
            "Missing required subset key 'library_layout'. "
            "Expected one of: library_layout, layout"
        )
    library_layout = str(library_layout).lower()

    # Forward primer is optional (can run without trimming). Warn if missing.
    fwd_primer_seq = _pick_first(subset, ["pcr_primer_fwd_seq", "fwd_primer_seq", "forward_primer"])
    if fwd_primer_seq is None:
        logger.warning(
            f"No forward primer found for {subset_id}. "
            "Proceeding without adapter trimming. "
            "Expected one of: pcr_primer_fwd_seq, fwd_primer_seq, forward_primer"
        )
        fwd_primer_seq = "NONE"

    is_paired = library_layout == 'paired'
    rev_primer_seq = None
    if is_paired:
        rev_primer_seq = _pick_first(subset, ["pcr_primer_rev_seq", "rev_primer_seq", "reverse_primer"])
        if rev_primer_seq is None:
                logger.warning(
                    f"No reverse primer found for {subset_id}. "
                    "Proceeding without adapter trimming. "
                    "Expected one of: pcr_primer_rev_seq, rev_primer_seq, reverse_primer"
                )
                rev_primer_seq = "NONE"

    command = [
        "nice", "-n", "15",
        "conda", "run", "--prefix", conda_env_path,
        "--no-capture-output", "--live-stream",
        "python", "-u", str(script_path),
        "--dada2-mode", str(dada2_mode),
        "--expected_amplicon_size", str(expected_amplicon_size) if expected_amplicon_size is not None else "None",
        "--qiime_dir", str(qiime_dir),
        "--metadata_tsv", str(metadata_path),
        "--manifest_tsv", str(manifest_path),
        "--library_layout", library_layout,
        "--fwd_primer_seq", str(fwd_primer_seq),
    ]

    if is_paired:
        command.extend([
            "--rev_primer_seq", str(rev_primer_seq)
        ])

    if detected_adapters:
        command.extend(["--detected_adapters"] + detected_adapters)

    dynamic_threads = getattr(app_config.sequences.validate_16s, 'n_threads', 16)
    
    command.extend([
        "--chimera_method", q2_config.denoise.chimera_method,
        "--confidence", str(q2_config.taxonomy.confidence),
        "--n_threads", str(dynamic_threads),
        "--min_frequency", str(q2_config.filter.retain_threshold),
        "--collapse_level", str(q2_config.collapse_level)
    ])
    
    if taxonomy_strategy == "sepp":
        command.extend([
            "--taxonomy_strategy", "sepp",
            "--backbone_path", str(backbone_path),
            "--reference_taxonomy_path", str(ref_tax_path),
            "--reference_tree_path", str(ref_tree_path)
        ])
    elif taxonomy_strategy == "sklearn":
        command.extend([
            "--taxonomy_strategy", "sklearn",
            "--classifier_path", str(classifier_path)
        ])  
    elif taxonomy_strategy == "gg2":
        command.extend([
            "--taxonomy_strategy", "gg2",
            "--backbone_path", str(backbone_path),
            "--reference_taxonomy_path", str(ref_tax_path),
            "--reference_tree_path", str(ref_tree_path)
        ])
        
    if dada2_mode == 'manual':
        trunc_len_f = subset.get("trunc_len_f")
        if trunc_len_f is None:
            raise KeyError("Manual DADA2 mode requires subset['trunc_len_f'].")

        dada2_params = [str(trunc_len_f)]
        if is_paired:
            trunc_len_r = subset.get("trunc_len_r")
            if trunc_len_r is None:
                raise KeyError("Manual DADA2 mode for paired-end requires subset['trunc_len_r'].")
            dada2_params.extend([
                str(trunc_len_r),
                str(len(str(fwd_primer_seq))),
                str(len(str(rev_primer_seq)))
            ])
        command.extend(["--dada2-params", *dada2_params])

    if q2_config.hard_rerun: command.append("--hard_rerun")
    if app_config.sequences.trim.cutadapt.enabled: command.append("--trim_sequences")
    if q2_config.diversity.enabled: command.append("--diversity")

    p = progress_obj
    task_id = p.add_task(f"[bold yellow]QIIME 2: {subset_id}", total=None) if p else None
    stderr_logs = []
    
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        while True:
            line = await process.stdout.readline()
            if not line: break

            decoded = line.decode().strip()
            if not decoded: continue

            stderr_logs.append(decoded) 

            if p and task_id:
                trigger_words = ["DADA2", "CLASSIFY", "IMPORT", "FILTER", "CUTADAPT", "RUNNING", "SAVED"]
                if any(kw in decoded.upper() for kw in trigger_words):
                    short_log = decoded[:55] + "..." if len(decoded) > 55 else decoded
                    p.update(task_id, description=f"[yellow]QIIME2 ({subset_id}): {short_log}")
                    p.advance(task_id)

        await process.wait()

        # 🚀 OPTION 2: WRITE FORENSIC LOG TO DISK
        log_file_path = qiime_dir / "qiime2_execution.log"
        try:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"--- QIIME 2 Execution Log for {subset_id} ---\n")
                ansi_escape = re.compile(r'\x1B\[[0-?]*[ -\/]*[@-~]')
                clean_logs = [ansi_escape.sub("", line) for line in stderr_logs]
                log_file.write("\n".join(clean_logs))
                log_file.write(f"\n--- End of Log (Exit Code: {process.returncode}) ---\n")
        except Exception as e:
            logger.warning(f"Could not save execution log for {subset_id}: {e}")

        if process.returncode != 0:
            # Extract meaningful error message from logs
            error_lines = [l for l in stderr_logs if any(kw in l for kw in ["ERROR", "Error", "FAIL_", "Exception", "Traceback", "ValueError", "RuntimeError"])]
            
            # Find the key error message (usually the last meaningful one)
            key_error = "Unknown error occurred"
            if error_lines:
                # Look for FAIL_ prefix which we use
                fail_lines = [l for l in error_lines if "FAIL_" in l]
                if fail_lines:
                    key_error = fail_lines[-1].strip()
                else:
                    key_error = error_lines[-1].strip()
            
            created = [name for name, path in final_artifact_paths.items() if path.exists()]
            missing = [name for name, path in final_artifact_paths.items() if not path.exists()]
            
            # Provide helpful suggestions based on error type
            suggestions = ""
            if "Missing filepath" in key_error or "absolute-filepath" in key_error:
                suggestions = "\n💡 SUGGESTED FIX: Check manifest.tsv format. For single-end, use 'absolute-filepath' column. For paired-end, use 'forward-absolute-filepath' and 'reverse-absolute-filepath'."
            elif "No reads passed the filter" in key_error or "truncLen longer than" in key_error:
                suggestions = "\n💡 SUGGESTED FIX: Read quality is poor or truncation length is too aggressive. Check quality_control.txt in this directory and adjust dada2_params or expected_amplicon_size."
            elif "Can only use .str accessor with string values" in key_error or "FAIL_TAXONOMY" in key_error:
                suggestions = "\n💡 SUGGESTED FIX: Metadata column dtype issue. Ensure all DataFrame columns are strings before passing to QIIME 2 Metadata."
            elif "Connection refused" in key_error or "Failed to import" in key_error:
                suggestions = "\n💡 SUGGESTED FIX: Ensure all sequence files exist and are readable. Check file paths in manifest.tsv."
            
            error_report = (
                f"❌ QIIME 2 failed for {subset_id} (Exit Code {process.returncode})\n"
                f"📋 Key Error: {key_error}\n"
                f"📁 Files Created: {', '.join(created) if created else 'None'}\n"
                f"⚠️  Files Missing: {', '.join(missing)}\n"
                f"{suggestions}\n"
                f"\n📖 Full execution log saved to: {log_file_path}"
            )
            
            logger.error(error_report)
            raise RuntimeError(f"QIIME 2 workflow failure for {subset_id}\n{key_error}")

    finally:
        if p and task_id is not None: p.remove_task(task_id)

    return final_artifact_paths
