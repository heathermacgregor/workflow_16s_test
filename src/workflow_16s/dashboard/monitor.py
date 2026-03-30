# workflow_16s/dashboard/monitor.py

import os
import time
import sqlite3
import psutil
from pathlib import Path
from typing import List, Dict, Optional

class SystemMonitor:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.log_dir = self.project_root / "07_logs"
        self.raw_dir = self.project_root / "01_raw_data"
        self.qiime_dir = self.project_root / "02_qiime"
        self.cache_dir = self.project_root / ".cache"
        self.last_log_pos = 0
        self.current_log_file = None

    def get_process_tree(self) -> List[Dict]:
        tree = []
        # Scan everything, ignore strict parent/child hierarchy
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if not proc.info['cmdline']: continue
                cmd = " ".join(proc.info['cmdline']).lower()
                
                # Catch the Master Process
                if 'workflow_16s.upstream' in cmd and 'dashboard' not in cmd:
                    tree.append({
                        "pid": proc.info['pid'], "name": "AmpliScout Master", 
                        "cpu": proc.info['cpu_percent'], "ram": proc.info['memory_percent']
                    })
                # Catch the rogue QIIME / VSEARCH / Processor children
                elif ('qiime' in cmd or 'vsearch' in cmd or ('python' in cmd and 'dada2' in cmd)):
                    tree.append({
                        "pid": proc.info['pid'], "name": proc.info['name'], 
                        "cpu": proc.info['cpu_percent'], "ram": proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Sort master to top, children below
        return sorted(tree, key=lambda x: 0 if x['name'] == 'AmpliScout Master' else 1)

    def get_recent_files(self, limit: int = 8) -> List[Dict]:
        recent = []
        for d in [self.raw_dir, self.qiime_dir]:
            if d.exists():
                files = []
                for f in d.rglob('*'):
                    if f.is_file() and not f.name.startswith('.'):
                        try: files.append(f)
                        except: pass
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for f in files[:limit]:
                    rel_path = f.relative_to(self.project_root)
                    recent.append({"path": str(rel_path), "time": f.stat().st_mtime})
                    
        recent.sort(key=lambda x: x['time'], reverse=True)
        return [{"path": r["path"], "time": time.strftime('%H:%M:%S', time.localtime(r['time']))} for r in recent[:limit]]

    def get_latest_log(self) -> Optional[Path]:
        if not self.log_dir.exists(): return None
        logs = [f for f in self.log_dir.glob("*.log") if f.is_file()]
        if not logs: return None
        logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return logs[0]

    def yield_log_lines(self):
        while True:
            latest = self.get_latest_log()
            if not latest:
                yield "data: Waiting for logs...\n\n"
                time.sleep(2)
                continue

            if self.current_log_file != latest:
                self.current_log_file = latest
                self.last_log_pos = 0
                yield f"data: [SYSTEM] Now tracking newest log: {latest.name}\n\n"

            try:
                with open(self.current_log_file, "r") as f:
                    f.seek(self.last_log_pos)
                    lines = f.readlines()
                    if lines:
                        self.last_log_pos = f.tell()
                        for line in lines:
                            if line.strip():
                                yield f"data: {line.strip()}\n\n"
            except Exception as e:
                yield f"data: [SYSTEM ERROR] Log read failed: {e}\n\n"
            
            time.sleep(1)

    def get_cache_counts(self) -> Dict[str, int]:
        counts = {"ena_metadata": 0, "partitions": 0}
        ena_db = self.cache_dir / "ena_metadata" / "ena_cache.db"
        part_db = self.cache_dir / "partition_cache.db"
        
        try:
            if ena_db.exists():
                with sqlite3.connect(ena_db) as conn:
                    counts["ena_metadata"] = conn.execute("SELECT count(*) FROM cache").fetchone()[0]
            if part_db.exists():
                with sqlite3.connect(part_db) as conn:
                    counts["partitions"] = conn.execute("SELECT count(*) FROM cache").fetchone()[0]
        except Exception: pass
        return counts