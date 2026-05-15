import importlib


def test_ena_compat_exports_sequence_symbols():
    compat = importlib.import_module("workflow_16s.api.ena")
    sequence = importlib.import_module("workflow_16s.api.sequence.ena")

    names = [
        "ENAClient",
        "ENAEnrichmentPipeline",
        "ENAFetcher",
        "SampleParser",
        "SQLiteCacheManager",
        "SequenceFetcher",
    ]

    for name in names:
        assert hasattr(compat, name), f"compat layer missing {name}"
        assert hasattr(sequence, name), f"sequence module missing {name}"
        assert getattr(compat, name) is getattr(sequence, name)


def test_qiime_execute_compat_aliases():
    compat = importlib.import_module("workflow_16s.api.qiime.execute")
    seqs_mod = importlib.import_module(
        "workflow_16s.api.sequence.qiime.execute_seqs_to_features"
    )
    phylo_mod = importlib.import_module(
        "workflow_16s.api.sequence.qiime.execute_phylogenetic_metaanalysis"
    )

    assert compat.seqs_to_features is seqs_mod.execute_per_dataset_qiime_workflow
    assert (
        compat.phylogenetic_metaanalysis
        is phylo_mod.execute_phylogenetic_meta_analysis
    )


def test_utils_logging_compat_exports_logger_helpers():
    compat = importlib.import_module("workflow_16s.utils.logging")
    canonical = importlib.import_module("workflow_16s.utils.logger")

    for name in ["get_logger", "initialize_logging", "setup_logging", "with_logger"]:
        assert hasattr(compat, name), f"compat logging missing {name}"
        assert hasattr(canonical, name), f"canonical logging missing {name}"
        assert getattr(compat, name) is getattr(canonical, name)
