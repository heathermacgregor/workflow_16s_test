with open("src/workflow_16s/upstream/metadata/partition.py", "r") as f:
    content = f.read()

old_block = """            if any(f.get("dataset") == dataset for f in self.failed):
                self.logger.info(f" ⏭️ [{progress_str}] Skipped {dataset} due to upstream filter.")
                return False"""

new_block = """            dataset_failures = [f for f in self.failed if f.get("dataset") == dataset]
            if dataset_failures:
                if any(f.get("stage") == "QIIME Task" for f in dataset_failures):
                    err_msg = next((f.get("error") for f in dataset_failures if f.get("stage") == "QIIME Task"), "Unknown Error")
                    self.logger.error(f" 💥 [{progress_str}] QIIME Execution Failed for {dataset}: {err_msg}")
                    self.cache.add_failed_dataset(dataset, f"QIIME Failure: {err_msg}")
                else:
                    self.logger.info(f" ⏭️ [{progress_str}] Skipped {dataset} due to upstream filter.")
                return False"""

content = content.replace(old_block, new_block)

with open("src/workflow_16s/upstream/metadata/partition.py", "w") as f:
    f.write(content)
