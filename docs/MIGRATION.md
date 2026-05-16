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
- ✅ `utils/dir_utils.py` → `core/dir_utils.py` (commit 5b04be8)
- Tests: `test_phase1_moves.py` (2 tests, PASSING)

**Phase 1 Batch 2 Utilities:**
- ✅ `utils/biom_utils.py` → `core/biom_utils.py` (commit 8c9016b)
  - Note: Now imports from `core.progress`, not `utils.progress`
- ✅ `utils/constants.py` → `core/constants.py` (commit 8c9016b)
- Tests: `test_phase1_batch2.py` (2 tests, PASSING)

**Phase 1 Batch 3 Utilities:**
- ✅ `utils/taxonomy.py` → `core/taxonomy.py` (commit c2c27f1)
  - Contains: FaprotaxDB, FaprotaxError, DownloaderError, ParserError
- ✅ `utils/validation.py` → `core/validation.py` (commit c2c27f1)
  - Contains: ResultsValidator, get_validator(), validate_results()
- Tests: `test_phase1_batch3.py` (2 tests, PASSING)

**Comprehensive Test Suite:**
- `test_anndata_migration.py` – Validates anndata POC move (2 tests)
- `test_module_compat_layers.py` – Validates existing compat shims (3 tests)
- `test_phase1_moves.py` – Validates Phase 1 batch 1 (2 tests)
- `test_phase1_batch2.py` – Validates Phase 1 batch 2 (2 tests)
- `test_phase1_batch3.py` – Validates Phase 1 batch 3 (2 tests)
- **Total: 11 tests, all PASSING**

### Remaining Phase 1 Utilities

Candidates for next rounds:
- `utils/auto_tune.py` (295 LOC) - optimization utilities
- `utils/faprotax.py` (473 LOC) - FAPROTAX database access (consider split)
- `utils/data.py` (726 LOC) - data loading (consider split)
- `utils/compositional.py` (731 LOC) - compositional analysis (consider split)
- `utils/publication_fetcher.py` (1458 LOC - requires split)

## Next steps

1. Continue Phase 1 with `auto_tune.py` (295 LOC) for batch 4.
2. For large files (473+ LOC), plan splits into focused modules before moving.
3. After Phase 1 completes, begin Phase 2: Move api/sequence/ena/* with preservation.
4. Consider Phase 3: Move environmental API modules to ingest/environmental/*.

---

Document created by migration assistant on 2026-05-15.
Last updated: 2026-05-15 (Phase 1 batch 3 progress recorded).
