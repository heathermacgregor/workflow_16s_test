class Ordination:
    """
    Performs ordination analyses (PCA, PCoA, t-SNE, UMAP) and stores figures.
    """
    
    TEST_CONFIG = {
        "pca": {
            "key": "pca", 
            "func": calculate_pca, 
            "plot_func": plot_pca, 
            "name": "PCA"
        },
        "pcoa": {
            "key": "pcoa", 
            "func": calculate_pcoa, 
            "plot_func": plot_pcoa, 
            "name": "PCoA"
        },
        "tsne": {
            "key": "tsne",
            "func": calculate_tsne,
            "plot_func": plot_mds,
            "name": "t‑SNE",
            "plot_kwargs": {"mode": "TSNE"},
        },
        "umap": {
            "key": "umap",
            "func": calculate_umap,
            "plot_func": plot_mds,
            "name": "UMAP",
            "plot_kwargs": {"mode": "UMAP"},
        },
    }

    def __init__(
        self, 
        cfg: Dict, 
        output_dir: Union[str, Path], 
        verbose: bool = False
    ):
        self.cfg = cfg
        self.verbose = verbose
        self.figure_output_dir = Path(output_dir)
        self.color_columns = cfg["maps"].get(
            "color_columns",
            [
                DEFAULT_DATASET_COLUMN, DEFAULT_GROUP_COLUMN,
                "env_feature", "env_material", "country"
            ],
        )

    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        symbol_col: str,
        transformation: str,
        enabled_tests: List[str],
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        trans_cfg = self.cfg.get("ordination", {}).get(transformation, {})
        tests_to_run = [t for t in enabled_tests if t in self.TEST_CONFIG]
        if not tests_to_run:
            return {}, {}

        table, metadata = update_table_and_meta(table, metadata)
        return self._run_without_progress(
            table, metadata, symbol_col, transformation,
            tests_to_run, trans_cfg, kwargs,
        )

    def _run_without_progress(
        self, table, metadata, symbol_col, transformation, tests_to_run, 
        trans_cfg, kwargs,
    ):
        results, figures = {}, {}
        for tname in tests_to_run:
            cfg = self.TEST_CONFIG[tname]
            try:
                res, figs = self._run_ordination_method(
                    cfg, table, metadata, symbol_col, transformation, 
                    trans_cfg, kwargs
                )
                results[cfg["key"]] = res
                figures[cfg["key"]] = figs
            except Exception as e:
                logger.error(f"Failed {tname} for {transformation}: {e}")
                figures[cfg["key"]] = {}
        return results, figures

    def _run_ordination_method(
        self, cfg, table, metadata, symbol_col, transformation, trans_cfg,
        kwargs
    ):
        try:
            if debug_mode:
                time.sleep(3)
                return
            method_params = {}
            if cfg["key"] == "pcoa":
                method_params["metric"] = trans_cfg.get("pcoa_metric", "braycurtis")
                logger.info(method_params["metric"])
            
            # Handle UMAP/TSNE thread safety
            if cfg["key"] in ["tsne", "umap"]:
                method_params["n_jobs"] = 1
                # Use global lock to prevent NUMBA thread conflicts
                with umap_lock:
                    # Set NUMBA threads before importing/executing
                    os.environ['NUMBA_NUM_THREADS'] = '1'
                    import numba
                    numba.config.NUMBA_NUM_THREADS = 1
                    ord_res = cfg["func"](table=table, **method_params)
            else:
                ord_res = cfg["func"](table=table, **method_params)
        except Exception as e:
            logger.error(f"Params: {method_params}")
            logger.error(f"Failed {cfg['key']} for {transformation}: {e}")
            return None, {}

        try:
            figures = {}
            pkwargs = {**cfg.get("plot_kwargs", {}), **kwargs}
            
            for color_col in self.color_columns:
                if color_col not in metadata.columns:
                    logger.warning(f"Color column '{color_col}' not found in metadata")
                    continue
    
                if cfg["key"] == "pca":
                    pkwargs.update({
                        "components": ord_res["components"],
                        "proportion_explained": ord_res["exp_var_ratio"],
                    })
                elif cfg["key"] == "pcoa":
                    pkwargs.update({
                        "components": ord_res.samples,
                        "proportion_explained": ord_res.proportion_explained,
                    })
                else:
                    pkwargs["df"] = ord_res
    
                fig, _ = cfg["plot_func"](
                    metadata=metadata,
                    color_col=color_col,
                    symbol_col=symbol_col,
                    transformation=transformation,
                    output_dir=self.figure_output_dir,
                    **pkwargs,
                )
                if fig:  # Only add if figure was created
                    figures[color_col] = fig
            return ord_res, figures

        except Exception as e:
            logger.error(f"Failed {cfg['key']} plot for {transformation}: {e}")
            return ord_res, {}

