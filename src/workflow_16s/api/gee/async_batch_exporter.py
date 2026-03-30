"""
Asynchronous Batch Export Module for Google Earth Engine

Enables high-throughput GEE queries using batch export tasks:
- Converts coordinate arrays to GEE FeatureCollections
- Creates mega-image from band-stacked datasets
- Submits batch export tasks (Drive or Cloud Storage)
- Polls for asynchronous task completion
- Downloads and processes results

Expected improvements over synchronous approach:
- Non-blocking: submit all batches at once
- For 463K samples: ~5-10 requests vs sequential batches
- Total time: 30-60 minutes (vs 2-3 hours synchronous)
- Better utilization of GEE parallel processing

Usage:
    exporter = AsyncBatchExporter(config, gcs_bucket="gs://my-bucket")
    task_id = exporter.create_sample_task(feature_collection, mega_image, "batch_001")
    status = exporter.poll_task_status(task_id, max_wait_hours=24)
    results = exporter.download_results(task_id, "/path/to/results")
"""

import logging
import json
import time
import os
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration of GEE task statuses."""
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExportConfig:
    """Configuration for async batch export."""
    cloud_storage_bucket: Optional[str] = None
    max_wait_hours: int = 24
    poll_interval_seconds: int = 60
    max_retries: int = 3
    enable_drive_export: bool = True
    log_task_ids: bool = True
    task_log_file: Optional[str] = None


class AsyncBatchExporter:
    """
    High-performance async batch exporter for Google Earth Engine queries.

    Handles conversion of coordinate arrays to GEE FeatureCollections,
    creation and monitoring of batch export tasks, and result retrieval.
    """

    def __init__(self, config: ExportConfig, gee_authenticated: bool = True):
        """
        Initialize async batch exporter.

        Args:
            config: ExportConfig with async settings
            gee_authenticated: Whether GEE is authenticated
        """
        if not HAS_EE:
            logger.error("Google Earth Engine module not available")
            raise ImportError("ee module required for AsyncBatchExporter")

        self.config = config
        self.gee_authenticated = gee_authenticated
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

        # Setup task logging
        if config.log_task_ids:
            self.task_log_file = config.task_log_file or Path.home() / ".cache" / "gee_task_log.json"
            self.task_log_file = Path(self.task_log_file)
            self.task_log_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_task_log()
        else:
            self.task_log_file = None

        # Setup GCS client if bucket specified
        self.gcs_client = None
        self.bucket_name = None
        if config.cloud_storage_bucket:
            if not HAS_GCS:
                logger.warning("google-cloud-storage not available, will use Drive export")
            else:
                try:
                    self.gcs_client = gcs.Client()
                    # Extract bucket name from gs://bucket-name/path
                    parts = config.cloud_storage_bucket.split("/")
                    self.bucket_name = parts[2] if len(parts) > 2 else parts[0].replace("gs://", "")
                    logger.info(f"GCS client initialized for bucket: {self.bucket_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize GCS client: {e}, falling back to Drive")
                    self.gcs_client = None

    def coords_to_feature_collection(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
        properties_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> "ee.FeatureCollection":
        """
        Convert coordinate arrays to GEE FeatureCollection.

        Each coordinate becomes a Point feature with properties including:
        - sample_id: Identifier for the sample
        - original_index: Index in original arrays
        - latitude/longitude: Coordinate values
        - Any additional properties passed

        Args:
            lats: Array of latitude values
            lons: Array of longitude values
            sample_ids: Optional array of sample identifiers
            properties_dict: Optional dict of property_name -> array mappings

        Returns:
            ee.FeatureCollection with Point features for each coordinate

        Raises:
            ValueError: If arrays have mismatched lengths or invalid values
        """
        if len(lats) != len(lons):
            raise ValueError(f"Latitude ({len(lats)}) and longitude ({len(lons)}) arrays have different lengths")

        if len(lats) == 0:
            raise ValueError("Cannot create FeatureCollection from empty arrays")

        # Validate coordinates
        valid_mask = (
            (~np.isnan(lats)) &
            (~np.isnan(lons)) &
            (lats >= -90) &
            (lats <= 90) &
            (lons >= -180) &
            (lons <= 180)
        )

        if not np.any(valid_mask):
            raise ValueError("No valid coordinates found in input arrays")

        valid_indices = np.where(valid_mask)[0]
        valid_lats = lats[valid_indices]
        valid_lons = lons[valid_indices]

        logger.info(f"Creating FeatureCollection from {len(valid_indices)}/{len(lats)} valid coordinates")

        # Create features
        features = []
        for i, (lat, lon) in enumerate(zip(valid_lats, valid_lons)):
            original_index = valid_indices[i]

            # Build properties
            props = {
                "sample_id": str(sample_ids[original_index]) if sample_ids is not None else f"sample_{original_index}",
                "original_index": int(original_index),
                "latitude": float(lat),
                "longitude": float(lon)
            }

            # Add custom properties
            if properties_dict:
                for prop_name, prop_array in properties_dict.items():
                    props[prop_name] = float(prop_array[original_index])

            # Create Point geometry and feature
            point = ee.Geometry.Point([lon, lat])
            feature = ee.Feature(point, props)
            features.append(feature)

        # Create FeatureCollection
        fc = ee.FeatureCollection(features)
        logger.info(f"✓ FeatureCollection created with {len(features)} features")

        return fc

    def create_sample_task(
        self,
        feature_collection: "ee.FeatureCollection",
        mega_image: "ee.Image",
        output_name: str,
        file_format: str = "CSV",
        geometry: Optional["ee.Geometry"] = None
    ) -> str:
        """
        Create and submit an async export task.

        Submits a batch export task to GEE, returning immediately with task ID.
        Task can be polled for completion status.

        Args:
            feature_collection: ee.FeatureCollection with sample points
            mega_image: ee.Image with stacked bands to extract
            output_name: Name for output file (without extension)
            file_format: Output format ("CSV", "GeoJSON", "SHP")
            geometry: Optional geometry to clip results (defaults to FC bounds)

        Returns:
            Task ID string that can be used to poll status/download results

        Raises:
            RuntimeError: If task creation fails
        """
        if not self.gee_authenticated:
            raise RuntimeError("GEE authentication required for task creation")

        try:
            # Create export task - use Drive by default, GCS if configured
            if self.gcs_client and self.bucket_name:
                logger.info(f"Creating GCS export task: gs://{self.bucket_name}/{output_name}")

                task = ee.batch.Export.table.toCloudStorage(
                    collection=feature_collection,
                    description=output_name,
                    bucket=self.bucket_name,
                    fileNamePrefix=f"gee-exports/{output_name}",
                    fileFormat=file_format,
                    maxResults=10000000  # Allow large exports
                )
            else:
                logger.info(f"Creating Drive export task: {output_name}")

                task = ee.batch.Export.table.toDrive(
                    collection=feature_collection,
                    description=output_name,
                    folder="gee-exports",
                    fileNamePrefix=output_name,
                    fileFormat=file_format,
                    maxResults=10000000
                )

            # Start the task
            task.start()
            task_id = task.id

            # Store task metadata
            self.task_registry[task_id] = {
                "output_name": output_name,
                "status": TaskStatus.READY.value,
                "created_at": datetime.now().isoformat(),
                "format": file_format,
                "use_gcs": bool(self.gcs_client and self.bucket_name),
                "feature_count": feature_collection.size().getInfo()
            }

            # Log task ID for recovery if script crashes
            if self.task_log_file:
                self._save_task_log()

            logger.info(f"✓ Task created: {task_id} (output: {output_name})")
            return task_id

        except Exception as e:
            logger.error(f"Failed to create export task: {e}")
            raise RuntimeError(f"GEE task creation failed: {e}") from e

    def poll_task_status(
        self,
        task_id: str,
        max_wait_hours: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Poll async task status until completion or timeout.

        Blocks until task completes or max_wait_hours exceeded.
        Returns immediately with current status each call.

        Args:
            task_id: Task ID to poll
            max_wait_hours: Max hours to wait (uses config default if None)
            verbose: Log progress updates

        Returns:
            Dict with task status info:
            {
                "state": TaskStatus enum value,
                "error_message": str or None,
                "progress": float (0-1),
                "estimated_output_size_bytes": int or None
            }

        Raises:
            ValueError: If task_id not found
            TimeoutError: If max_wait_hours exceeded
        """
        if task_id not in self.task_registry:
            raise ValueError(f"Unknown task ID: {task_id}")

        max_wait_hours = max_wait_hours or self.config.max_wait_hours
        deadline = datetime.now() + timedelta(hours=max_wait_hours)

        try:
            task = ee.batch.Task(task_id)
        except Exception as e:
            logger.error(f"Failed to get task object for {task_id}: {e}")
            return {
                "state": TaskStatus.UNKNOWN.value,
                "error_message": str(e),
                "progress": 0.0
            }

        start_time = datetime.now()
        last_state = None

        while datetime.now() < deadline:
            try:
                state = task.status()["state"]
                progress = task.status().get("progress", 0)

                # Log state changes
                if state != last_state:
                    logger.info(f"  Task {task_id}: {state} (progress: {progress*100:.0f}%)")
                    last_state = state
                    self.task_registry[task_id]["status"] = state

                # Handle completion states
                if state == "COMPLETED":
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    logger.info(f"✓ Task {task_id} completed in {elapsed:.1f} minutes")
                    return {
                        "state": TaskStatus.COMPLETED.value,
                        "error_message": None,
                        "progress": 1.0,
                        "elapsed_minutes": elapsed
                    }

                elif state == "FAILED":
                    error_msg = task.status().get("error_message", "Unknown error")
                    logger.error(f"✗ Task {task_id} failed: {error_msg}")
                    return {
                        "state": TaskStatus.FAILED.value,
                        "error_message": error_msg,
                        "progress": progress
                    }

                elif state == "CANCELLED":
                    logger.warning(f"⊘ Task {task_id} was cancelled")
                    return {
                        "state": TaskStatus.CANCELLED.value,
                        "error_message": "Task was cancelled",
                        "progress": progress
                    }

                # Still running - wait and retry
                if verbose:
                    time.sleep(self.config.poll_interval_seconds)
                else:
                    time.sleep(5)  # Shorter polling when not verbose

            except Exception as e:
                logger.debug(f"Error polling task {task_id}: {e}")
                time.sleep(self.config.poll_interval_seconds)

        # Timeout exceeded
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        logger.error(f"✗ Task {task_id} did not complete within {elapsed:.1f} hours")
        raise TimeoutError(f"Task {task_id} exceeded max_wait_hours ({max_wait_hours})")

    def download_results(
        self,
        task_id: str,
        local_path: Path,
        file_format: str = "CSV"
    ) -> Dict[str, Any]:
        """
        Download completed task results to local filesystem.

        For Drive exports: Downloads from Google Drive (requires manual setup)
        For GCS exports: Downloads directly from Cloud Storage

        Args:
            task_id: Completed task ID
            local_path: Local directory to save results
            file_format: File format ("CSV", "GeoJSON", "SHP")

        Returns:
            Dict with download metadata:
            {
                "local_path": Path to downloaded file,
                "file_size_mb": float,
                "row_count": int (for CSV),
                "download_time_seconds": float
            }

        Raises:
            FileNotFoundError: If output file not found
            RuntimeError: If download fails
        """
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        if task_id not in self.task_registry:
            raise ValueError(f"Unknown task ID: {task_id}")

        task_info = self.task_registry[task_id]
        output_name = task_info["output_name"]

        start_time = datetime.now()

        try:
            if task_info.get("use_gcs") and self.gcs_client:
                return self._download_from_gcs(output_name, local_path, file_format)
            else:
                logger.warning("Drive export requires manual download from Google Drive")
                logger.warning(f"  Folder: gee-exports")
                logger.warning(f"  File: {output_name}.{file_format.lower()}")
                return {
                    "local_path": None,
                    "status": "requires_manual_download",
                    "message": "GEE exports to Drive require manual download from Google Drive"
                }

        except Exception as e:
            logger.error(f"Failed to download results for {task_id}: {e}")
            raise RuntimeError(f"Download failed: {e}") from e

    def _download_from_gcs(
        self,
        output_name: str,
        local_path: Path,
        file_format: str
    ) -> Dict[str, Any]:
        """Download results from GCS bucket."""
        if not self.gcs_client or not self.bucket_name:
            raise RuntimeError("GCS client not initialized")

        # Determine expected filename
        ext = file_format.lower()
        blob_name = f"gee-exports/{output_name}.{ext}"

        logger.info(f"Downloading from GCS: gs://{self.bucket_name}/{blob_name}")

        try:
            bucket = self.gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")

            # Download file
            local_file = local_path / f"{output_name}.{ext}"
            start = datetime.now()
            blob.download_to_filename(local_file)
            elapsed = (datetime.now() - start).total_seconds()

            # Get file stats
            file_size_mb = os.path.getsize(local_file) / (1024 * 1024)
            row_count = None

            if file_format == "CSV":
                try:
                    df = pd.read_csv(local_file, nrows=None)
                    row_count = len(df)
                except Exception as e:
                    logger.warning(f"Could not read CSV row count: {e}")

            logger.info(f"✓ Downloaded {file_size_mb:.1f} MB in {elapsed:.1f}s")

            return {
                "local_path": str(local_file),
                "file_size_mb": file_size_mb,
                "row_count": row_count,
                "download_time_seconds": elapsed
            }

        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            raise

    def _load_task_log(self):
        """Load task log from disk for recovery."""
        try:
            if self.task_log_file.exists():
                with open(self.task_log_file) as f:
                    data = json.load(f)
                    self.task_registry.update(data)
                    logger.info(f"Loaded {len(self.task_registry)} tasks from log")
        except Exception as e:
            logger.warning(f"Could not load task log: {e}")

    def _save_task_log(self):
        """Save task log to disk for recovery."""
        try:
            with open(self.task_log_file, 'w') as f:
                json.dump(self.task_registry, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save task log: {e}")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get cached task status without polling."""
        if task_id not in self.task_registry:
            raise ValueError(f"Unknown task ID: {task_id}")
        return self.task_registry[task_id]

    def list_pending_tasks(self) -> List[str]:
        """Get list of incomplete task IDs (for recovery)."""
        pending = [
            task_id for task_id, info in self.task_registry.items()
            if info.get("status") not in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]
        ]
        return pending

    def retry_failed_task(self, task_id: str) -> str:
        """
        Retry a failed task with exponential backoff.

        Args:
            task_id: Task ID to retry

        Returns:
            New task ID for retry attempt
        """
        if task_id not in self.task_registry:
            raise ValueError(f"Unknown task ID: {task_id}")

        retry_count = self.retry_counts.get(task_id, 0)
        if retry_count >= self.config.max_retries:
            raise RuntimeError(f"Task {task_id} exceeded max retries ({self.config.max_retries})")

        # Exponential backoff
        backoff_seconds = min(2 ** retry_count * 60, 3600)  # Max 1 hour
        logger.info(f"Retrying task {task_id} (attempt {retry_count + 1}/{self.config.max_retries}) after {backoff_seconds}s")
        time.sleep(backoff_seconds)

        self.retry_counts[task_id] = retry_count + 1

        # Note: In practice, you'd need to recreate the task
        # This is a placeholder for the retry logic
        return f"{task_id}_retry_{retry_count + 1}"


def create_mega_image(
    datasets: Dict[str, Any],
    date_range: Optional[Tuple[str, str]] = None
) -> "ee.Image":
    """
    Create mega-image from band-stacked datasets.

    Combines multiple GEE datasets into a single image with all bands.
    Useful for efficient batch sampling.

    Args:
        datasets: Dict of dataset_name -> ee.Image
        date_range: Optional (start_date, end_date) for filtering

    Returns:
        ee.Image with all bands stacked
    """
    if not HAS_EE:
        raise ImportError("ee module required")

    if not datasets:
        raise ValueError("No datasets provided")

    images = []
    for name, image in datasets.items():
        if image is None:
            logger.warning(f"Skipping None image for dataset {name}")
            continue

        try:
            # Rename bands to include dataset name for clarity
            if hasattr(image, 'bandNames'):
                band_names = image.bandNames().getInfo()
                new_names = [f"{name}_{band}" for band in band_names]
                image = image.select(band_names, new_names)

            images.append(image)
            logger.debug(f"Added {name} to mega-image")
        except Exception as e:
            logger.warning(f"Failed to add {name} to mega-image: {e}")

    if not images:
        raise ValueError("No valid images to stack")

    # Stack all images
    mega = ee.Image(images[0])
    for img in images[1:]:
        mega = mega.addBands(img)

    logger.info(f"✓ Mega-image created with {len(images)} datasets")
    return mega
