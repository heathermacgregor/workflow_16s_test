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

## Next steps (immediate)

1. Create `workflow_16s/core/__init__.py` and move a small proof-of-concept
   utility (`utils/io/anndata.py`) into `core/` with imports updated.
2. Add shims for the moved module.
3. Commit and push the migration plan and proof-of-concept changes.

---

Document created by migration assistant on 2026-05-15.
