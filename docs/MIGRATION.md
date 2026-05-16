# Migration Plan: Reorganize workflow_16s

This document records the initial migration plan and file-level mapping for
reorganizing the `workflow_16s` package. It emphasizes preserving the existing
ability to organically find samples (ENA/SRA discovery and fallback behavior)
and outlines compatibility shims and verification steps.

## Goals
- Reorganize into clear packages: `core/`, `ingest/`, `processing/`, `analysis/`, `cli/`.
- Preserve existing public/internal APIs during migration via shims.
- Keep the organic sample-discovery behavior intact (ENA finder, hierarchical
  fetchers, coordinate fallback) with minimal interface changes.
- Split very large files (>500 LOC) into focused modules.

## High-level mapping

- `workflow_16s/utils/*`  -> `workflow_16s/core/*`
- `workflow_16s/api/sequence/*` -> `workflow_16s/ingest/sequence/*`
  - `api/sequence/ena/*` -> `ingest/sequence/ena/*` (preserve `finder.py`,
    `metadata_fetcher.py`, `sample_parser.py`, `hierarchical_fetcher.py`,
    `coordinate_fallback.py` under the new path). These modules implement the
    organic sample discovery flow and must retain function/class names.
- `workflow_16s/api/environmental_data/*` -> `workflow_16s/ingest/environmental/*`
- `workflow_16s/upstream/sequences/*` -> `workflow_16s/processing/seqs/*`
- `workflow_16s/downstream/*` -> `workflow_16s/analysis/*` (ML, stats, ecotype)

## Preserve organic sample finding (key modules)

The following ENA modules implement sample discovery logic and should be
preserved with minimal API changes. During migration create a shim at the old
location that re-exports from the new location and emits a `DeprecationWarning`.

- `api/sequence/ena/finder.py` -> `ingest/sequence/ena/finder.py`
- `api/sequence/ena/metadata_fetcher.py` -> `ingest/sequence/ena/metadata_fetcher.py`
- `api/sequence/ena/sample_parser.py` -> `ingest/sequence/ena/sample_parser.py`
- `api/sequence/ena/hierarchical_fetcher.py` -> `ingest/sequence/ena/hierarchical_fetcher.py`
- `api/sequence/ena/coordinate_fallback.py` -> `ingest/sequence/ena/coordinate_fallback.py`

Keep these constraints:
- Do not change public function/class names in these files during Phase 1.
- Convert intra-repo imports to absolute imports when updating callers.
- Add unit tests for finder/metadata_fetcher to ensure identical behavior.

## File splits (high-priority)

- `config/config_schema.py` -> `core/config/schema.py`, `core/config/validators.py`
- `api/environmental_data/gee/mega_image.py` -> `ingest/environmental/gee/client.py`, `ingest/environmental/gee/builder.py`
- `utils/publication_fetcher.py` -> `core/io/publication_client.py`, `core/parsers/publication_parsers.py`
- `downstream/qc/metadata_profiler.py` -> `analysis/qc/profile.py`, `analysis/qc/reports.py`

## Compatibility shims

For each moved module create a shim at the original path like:

```py
# Old path: workflow_16s/api/sequence/ena/finder.py
from workflow_16s.ingest.sequence.ena.finder import *
import warnings
warnings.warn("workflow_16s.api.sequence.ena.finder is deprecated; use workflow_16s.ingest.sequence.ena.finder", DeprecationWarning)
```

Shims should be small, import-tested, and included in the test matrix.

## Verification steps

1. After each move: run unit tests that touch moved modules.
2. Run `pytest workflow_16s/tests/` and smoke-run examples that use ENA finder.
3. Run linters and `black --check`.

## Git & release process

- Work in feature branches per phase (e.g., `reorg/phase1-core`).
- Merge only when tests pass in CI.
- Release with deprecation notes; remove shims in next major release.

## Migration Progress

### Completed Phase 1 Moves

**Proof-of-Concept (POC):**
- ✅ `utils/io/anndata.py` → `core/io/anndata.py` (commit 6f31e7c)
  - Test: `test_anndata_migration.py` (2 tests, PASSING)

**Phase 1 Batch 1 Utilities:**
- ✅ `utils/progress.py` → `core/progress.py` (commit 5b04be8)
  - Test: `test_phase1_moves.py::test_progress_compat_shim` (PASSING)
  
- ✅ `utils/dir_utils.py` → `core/dir_utils.py` (commit 5b04be8)
  - Test: `test_phase1_moves.py::test_dir_utils_compat_shim` (PASSING)

**Phase 1 Batch 2 Utilities:**
- ✅ `utils/biom_utils.py` → `core/biom_utils.py` (commit 8c9016b)
  - Test: `test_phase1_batch2.py::test_biom_utils_compat_shim` (PASSING)
  - Note: Now imports from `core.progress`, not `utils.progress`

- ✅ `utils/constants.py` → `core/constants.py` (commit 8c9016b)
  - Test: `test_phase1_batch2.py::test_constants_compat_shim` (PASSING)

**Test Suite:**
- `test_anndata_migration.py` – Validates anndata POC move (2 tests)
- `test_module_compat_layers.py` – Validates existing compat shims (3 tests)
- `test_phase1_moves.py` – Validates Phase 1 batch 1 (2 tests)
- `test_phase1_batch2.py` – Validates Phase 1 batch 2 (2 tests)
- **Total: 9 tests, all PASSING**

### Pending Phase 1 Utilities

Small to medium (100-300 LOC) candidates for next round:
- `utils/taxonomy.py` (220 LOC)
- `utils/validation.py` (318 LOC)
- `utils/auto_tune.py` (295 LOC)

Large candidates (require analysis/splitting):
- `utils/faprotax.py` (473 LOC)
- `utils/data.py` (726 LOC)
- `utils/compositional.py` (731 LOC)
- `utils/publication_fetcher.py` (1458 LOC - requires split)

## Next steps

1. Continue Phase 1 with remaining small utilities (taxonomy, validation, auto_tune).
2. Validate each move with tests following the established pattern.
3. Plan file splits for large utilities (publication_fetcher: 1458 LOC).
4. Begin Phase 2: Move api/sequence/ena/* modules with preservation constraints.

---

Document created by migration assistant on 2026-05-15.
Last updated: 2026-05-15 (Phase 1 batch 2 progress recorded).
