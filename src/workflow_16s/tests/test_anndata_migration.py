import importlib
import inspect

import anndata as ad
import numpy as np
import pandas as pd


def test_anndata_module_imports():
    core = importlib.import_module("workflow_16s.core.io.anndata")
    utils = importlib.import_module("workflow_16s.utils.io.anndata")

    # Core should expose at least these helpers
    expected = ["clean_metadata", "get_cfg_value"]
    for name in expected:
        assert hasattr(core, name), f"core missing {name}"
        assert hasattr(utils, name), f"shim missing {name}"
        # The shim should re-export the same object reference
        assert getattr(utils, name) is getattr(core, name)


def _make_sample_adata():
    X = np.array([[10, 0], [0, 5]])
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "depth": ["1000", "2000"]})
    var = pd.DataFrame(index=["f1", "f2"]) 
    return ad.AnnData(X=X, obs=obs, var=var)


def test_clean_metadata_behavior():
    core = importlib.import_module("workflow_16s.core.io.anndata")
    utils = importlib.import_module("workflow_16s.utils.io.anndata")

    adata1 = _make_sample_adata()
    adata2 = _make_sample_adata()

    out1 = core.clean_metadata(adata1, config=None)
    out2 = utils.clean_metadata(adata2, config=None)

    # Both should return AnnData and preserve observation counts
    assert isinstance(out1, ad.AnnData)
    assert isinstance(out2, ad.AnnData)
    assert out1.n_obs == out2.n_obs == 2
