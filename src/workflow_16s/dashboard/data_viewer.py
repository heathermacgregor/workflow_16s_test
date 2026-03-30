# workflow_16s/dashboard/data_viewer.py

import time
import sqlite3
import json
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, List

class DataViewer:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.processed_dir = self.project_root / "03_processed_data"
        self._last_scan_time = 0
        
        # 🟢 FIXED: File-level caching to prevent the server from freezing
        self.parsed_files = {} 
        self._cached_stats = {"total_files": 0, "total_samples": 0, "total_features": 0}
        self._cached_map_data = []
        self._cached_composition = {"environments": {}, "layouts": {}}

    def refresh_data(self):
        if not self.processed_dir.exists(): return

        h5ad_files = list(self.processed_dir.glob("*.h5ad"))
        
        # Count the files IMMEDIATELY, regardless of read success
        total_samples = 0
        total_features = 0
        map_points = []
        env_counts = {}
        layout_counts = {}

        for f in h5ad_files:
            try:
                adata = ad.read_h5ad(f, backed='r')
                n_obs, n_vars = adata.shape
                total_samples += n_obs
                total_features += n_vars

                obs = adata.obs
                if 'env_broad_scale' in obs.columns:
                    for val in obs['env_broad_scale'].dropna().unique():
                        env_counts[str(val)] = env_counts.get(str(val), 0) + int((obs['env_broad_scale'] == val).sum())
                
                if 'library_layout' in obs.columns:
                    for val in obs['library_layout'].dropna().unique():
                        layout_counts[str(val)] = layout_counts.get(str(val), 0) + int((obs['library_layout'] == val).sum())

                if 'lat' in obs.columns and 'lon' in obs.columns:
                    mapped_obs = obs.dropna(subset=['lat', 'lon']).copy()
                    if not mapped_obs.empty:
                        cols_to_keep = ['lat', 'lon', 'study_accession']
                        for c in ['env_broad_scale', 'description']:
                            if c in mapped_obs.columns: cols_to_keep.append(c)
                        mapped_obs['dataset_id'] = f.stem
                        map_points.extend(mapped_obs[cols_to_keep + ['dataset_id']].head(100).to_dict('records'))

            except Exception as e:
                # Expose the error to the terminal so we know WHY it's failing
                print(f"⚠️ Dashboard Error: Could not read {f.name} - {e}")

        # Update cache with the actual file count, even if samples/features remain 0 due to read errors
        self._cached_stats = {
            "total_files": len(h5ad_files),
            "total_samples": total_samples,
            "total_features": total_features
        }
        self._cached_map_data = map_points
        self._cached_composition = {
            "environments": env_counts,
            "layouts": layout_counts
        }

    def get_stats(self) -> Dict:
        self.refresh_data()
        return self._cached_stats
    
    def get_provenance_stats(self) -> Dict:
        """Reads the provenance.db to generate X/Y/Z error and Node stats."""
        db_path = self.project_root / "provenance.db"
        if not db_path.exists():
            return {"errors": {}, "nodes": {}, "total_processed": 0}

        try:
            with sqlite3.connect(db_path) as conn:
                # Count failures and successes
                df_errors = pd.read_sql_query("SELECT error_code, COUNT(*) as count FROM provenance GROUP BY error_code", conn)
                errors = dict(zip(df_errors['error_code'], df_errors['count']))

                # Count which 'thar' nodes are doing the work
                df_nodes = pd.read_sql_query("SELECT node, COUNT(*) as count FROM provenance GROUP BY node", conn)
                nodes = dict(zip(df_nodes['node'], df_nodes['count']))

                # Total datasets that have finished (either pass or fail)
                total = pd.read_sql_query("SELECT COUNT(*) as count FROM provenance", conn).iloc[0]['count']

            return {"errors": errors, "nodes": nodes, "total_processed": int(total)}
        except Exception as e:
            print(f"⚠️ Provenance read error: {e}")
            return {"errors": {}, "nodes": {}, "total_processed": 0}

    def get_map_data(self) -> List[Dict]:
        self.refresh_data()
        return self._cached_map_data
        
    def get_composition(self) -> Dict:
        self.refresh_data()
        return self._cached_composition
    
    def get_nfc_hot_sites(self) -> List[Dict]:
        """Scans the env cache for facilities that returned > 0 samples."""
        cache_db = self.project_root / ".cache" / "env" / "cache.db"
        hot_sites = []
        
        if not cache_db.exists(): return []

        try:
            with sqlite3.connect(cache_db) as conn:
                query = "SELECT key, CAST(data AS TEXT) FROM cache WHERE key LIKE 'sweep_done_%' AND data NOT LIKE '%EMPTY%'"
                cursor = conn.execute(query)
                for key, data in cursor.fetchall():
                    try:
                        samples = json.loads(data)
                        if samples and len(samples) > 0:
                            facility = samples[0].get('target_facility', 'Unknown')
                            hot_sites.append({
                                "facility": facility,
                                "hit_count": len(samples),
                                "coords": f"{samples[0].get('query_lat')}, {samples[0].get('query_lon')}"
                            })
                    except: continue
        except Exception: pass
        
        return sorted(hot_sites, key=lambda x: x['hit_count'], reverse=True)[:10]