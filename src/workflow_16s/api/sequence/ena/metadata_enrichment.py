"""
Location and Date Metadata Enrichment for ENA/SRA Data.

Provides extraction and standardization of:
- Geographic coordinates (lat/lon) from various formats
- Collection dates in multiple formats
- Location confidence and precision metrics
"""

import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from workflow_16s.utils.logger import get_logger


logger = get_logger("workflow_16s")


class LocationParser:
    """
    Parses and extracts geographic coordinates from various formats.

    Supports:
    - Explicit lat/lon fields (DMS, decimal degrees)
    - Location strings with embedded coordinates
    - Country/region information
    - Location confidence metrics
    """

    # Regex patterns for coordinate extraction
    DMS_PATTERN = re.compile(
        r'(\d+)\s*°\s*(\d+)\s*[\'′]\s*([\d.]+)\s*["″]\s*([NSEW])',
        re.IGNORECASE
    )

    DECIMAL_PATTERN = re.compile(
        r'(-?\d+\.?\d*)\s*[,;]\s*(-?\d+\.?\d*)',
    )

    # Bounding boxes for quick validation
    LAT_BOUNDS = (-90, 90)
    LON_BOUNDS = (-180, 180)

    @staticmethod
    def parse_decimal_degrees(
        lat_str: str,
        lon_str: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Parse latitude and longitude as decimal degrees.

        Args:
            lat_str: Latitude string
            lon_str: Longitude string

        Returns:
            Tuple of (lat, lon) or None if invalid
        """
        try:
            lat = float(lat_str)
            lon = float(lon_str)

            if LocationParser.LAT_BOUNDS[0] <= lat <= LocationParser.LAT_BOUNDS[1]:
                if LocationParser.LON_BOUNDS[0] <= lon <= LocationParser.LON_BOUNDS[1]:
                    return (lat, lon)

        except (ValueError, TypeError):
            pass

        return None

    @staticmethod
    def parse_dms(dms_str: str) -> Optional[float]:
        """
        Parse Degrees-Minutes-Seconds format.

        Format: dd° mm' ss" [N/S/E/W]

        Args:
            dms_str: DMS string

        Returns:
            Decimal degrees or None if invalid
        """
        match = LocationParser.DMS_PATTERN.search(dms_str)
        if not match:
            return None

        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        direction = match.group(4).upper()

        decimal = degrees + minutes / 60 + seconds / 3600

        # Apply sign based on direction
        if direction in ['S', 'W']:
            decimal = -decimal

        return decimal

    @staticmethod
    def extract_from_location_string(location_str: str) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from a location string.

        Handles formats like:
        - "37.7749, -122.4194"
        - "37.7749; -122.4194"
        - "latitude=37.7749,longitude=-122.4194"
        - "lat 37.7749 lon -122.4194"

        Args:
            location_str: Location string

        Returns:
            Tuple of (lat, lon) or None
        """
        if not location_str:
            return None

        # Normalize the string
        normalized = str(location_str).replace('=', ':').lower()

        # Try decimal pattern first
        match = LocationParser.DECIMAL_PATTERN.search(normalized)
        if match:
            try:
                val1 = float(match.group(1))
                val2 = float(match.group(2))
                # Assume first value is lat, second is lon
                return LocationParser.parse_decimal_degrees(str(val1), str(val2))
            except (ValueError, TypeError):
                pass

        # Try named coordinates
        lat_match = re.search(r'lat(?:itude)?[:=\s]+(-?\d+\.?\d*)', normalized)
        lon_match = re.search(r'lon(?:gitude)?[:=\s]+(-?\d+\.?\d*)', normalized)

        if lat_match and lon_match:
            return LocationParser.parse_decimal_degrees(lat_match.group(1), lon_match.group(1))

        return None

    @staticmethod
    def extract_coordinates(
        data: Dict[str, Any],
        lat_fields: Optional[List[str]] = None,
        lon_fields: Optional[List[str]] = None,
        location_fields: Optional[List[str]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from a metadata dictionary.

        Tries multiple fields in order of preference.

        Args:
            data: Metadata dictionary
            lat_fields: Field names to check for latitude
            lon_fields: Field names to check for longitude
            location_fields: Field names to check for location strings

        Returns:
            Tuple of (lat, lon) or None
        """
        if lat_fields is None:
            lat_fields = ["lat", "latitude", "sample_lat", "sample_latitude"]
        if lon_fields is None:
            lon_fields = ["lon", "longitude", "sample_lon", "sample_longitude"]
        if location_fields is None:
            location_fields = ["location", "environment", "isolation_source", "sample_title"]

        # Try explicit lat/lon fields
        for lat_field in lat_fields:
            for lon_field in lon_fields:
                if lat_field in data and lon_field in data:
                    coords = LocationParser.parse_decimal_degrees(
                        str(data[lat_field]),
                        str(data[lon_field])
                    )
                    if coords:
                        logger.debug(f"Extracted coordinates from {lat_field}/{lon_field}")
                        return coords

        # Try location strings
        for loc_field in location_fields:
            if loc_field in data:
                coords = LocationParser.extract_from_location_string(str(data[loc_field]))
                if coords:
                    logger.debug(f"Extracted coordinates from location string: {loc_field}")
                    return coords

        return None


class DateParser:
    """
    Parses and standardizes collection dates from various formats.

    Supports:
    - ISO 8601 format (YYYY-MM-DD)
    - Year-month format (YYYY-MM)
    - Year only (YYYY)
    - Other common formats
    - Partial dates (unknown day/month)
    """

    # Common date patterns
    PATTERNS = [
        (r'^(\d{4})-(\d{2})-(\d{2})$', 'YMD', '%Y-%m-%d'),  # YYYY-MM-DD
        (r'^(\d{4})/(\d{2})/(\d{2})$', 'YMD', '%Y/%m/%d'),  # YYYY/MM/DD
        (r'^(\d{2})/(\d{2})/(\d{4})$', 'DMY', '%m/%d/%Y'),  # MM/DD/YYYY
        (r'^(\d{2})-(\d{2})-(\d{4})$', 'DMY', '%m-%d-%Y'),  # MM-DD-YYYY
        (r'^(\d{4})-(\d{2})$', 'YM', '%Y-%m'),  # YYYY-MM
        (r'^(\d{4})/(\d{2})$', 'YM', '%Y/%m'),  # YYYY/MM
        (r'^(\d{4})$', 'Y', '%Y'),  # YYYY
    ]

    @staticmethod
    def parse_date(
        date_str: Union[str, datetime, date, None],
        return_precision: bool = False,
    ) -> Union[Optional[str], Tuple[Optional[str], str]]:
        """
        Parse a date string and return ISO 8601 format.

        Args:
            date_str: Date string to parse
            return_precision: If True, return (date, precision) tuple

        Returns:
            ISO 8601 formatted date string (YYYY-MM-DD), or (date, precision) if requested
            Returns None if date cannot be parsed

        Precision levels:
        - 'day' : YYYY-MM-DD
        - 'month' : YYYY-MM
        - 'year' : YYYY
        - 'unknown' : Unparseable
        """
        if date_str is None:
            return (None, 'unknown') if return_precision else None

        # Already a datetime object
        if isinstance(date_str, datetime):
            result = date_str.strftime('%Y-%m-%d')
            return (result, 'day') if return_precision else result

        if isinstance(date_str, date):
            result = date_str.isoformat()
            return (result, 'day') if return_precision else result

        date_str = str(date_str).strip()
        if not date_str:
            return (None, 'unknown') if return_precision else None

        # Try each pattern
        for pattern, pattern_type, fmt in DateParser.PATTERNS:
            if re.match(pattern, date_str):
                try:
                    parsed = datetime.strptime(date_str, fmt)

                    if pattern_type == 'Y':
                        precision = 'year'
                        result = parsed.strftime('%Y')
                    elif pattern_type == 'YM':
                        precision = 'month'
                        result = parsed.strftime('%Y-%m')
                    else:  # YMD
                        precision = 'day'
                        result = parsed.strftime('%Y-%m-%d')

                    return (result, precision) if return_precision else result

                except (ValueError, TypeError):
                    continue

        # If no pattern matched, return None
        return (None, 'unknown') if return_precision else None

    @staticmethod
    def standardize_date_field(
        df: pd.DataFrame,
        source_col: str,
        target_col: Optional[str] = None,
        precision_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Standardize dates in a DataFrame column.

        Args:
            df: Input DataFrame
            source_col: Column name with dates to parse
            target_col: Output column name (default: source_col + '_standardized')
            precision_col: Column to store date precision (optional)

        Returns:
            DataFrame with standardized dates
        """
        if source_col not in df.columns:
            logger.warning(f"Column {source_col} not found in DataFrame")
            return df

        if target_col is None:
            target_col = f"{source_col}_standardized"

        df_copy = df.copy()

        if precision_col:
            dates, precisions = zip(*[
                DateParser.parse_date(val, return_precision=True)
                for val in df_copy[source_col]
            ])
            df_copy[target_col] = dates
            df_copy[precision_col] = precisions
        else:
            df_copy[target_col] = [
                DateParser.parse_date(val)
                for val in df_copy[source_col]
            ]

        return df_copy


def enrich_metadata_with_location(
    df: pd.DataFrame,
    email: Optional[str] = None,
    lat_fields: Optional[List[str]] = None,
    lon_fields: Optional[List[str]] = None,
    location_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract and standardize geographic coordinates in metadata DataFrame.

    Adds columns:
    - 'latitude': Extracted/standardized latitude
    - 'longitude': Extracted/standardized longitude
    - 'location_confidence': 'explicit' or 'extracted'

    Args:
        df: Input DataFrame with metadata
        email: Email address (optional, for logging)
        lat_fields: Field names to check for latitude
        lon_fields: Field names to check for longitude
        location_fields: Field names to check for location strings

    Returns:
        DataFrame with extracted coordinates
    """
    df_copy = df.copy()

    # FIX #4: Replace iterrows() with apply() for better performance (10-100x faster)
    def extract_location_row(row: pd.Series) -> pd.Series:
        """Extract location from a single row."""
        row_dict = row.to_dict()

        # Try to extract coordinates
        coords = LocationParser.extract_coordinates(
            row_dict,
            lat_fields=lat_fields,
            lon_fields=lon_fields,
            location_fields=location_fields,
        )

        if coords:
            lat, lon = coords
            # Determine confidence
            if any(row.get(f) for f in (lat_fields or ["lat", "latitude"])):
                confidence = "explicit"
            else:
                confidence = "extracted"
            return pd.Series({
                'latitude': lat,
                'longitude': lon,
                'location_confidence': confidence
            })
        else:
            return pd.Series({
                'latitude': None,
                'longitude': None,
                'location_confidence': None
            })

    # Apply the function to all rows
    location_data = df_copy.apply(extract_location_row, axis=1)

    df_copy['latitude'] = location_data['latitude']
    df_copy['longitude'] = location_data['longitude']
    df_copy['location_confidence'] = location_data['location_confidence']

    # Log summary
    valid_count = sum(1 for c in location_data['location_confidence'] if c is not None)
    logger.info(f"Extracted/standardized coordinates for {valid_count}/{len(df_copy)} samples")

    return df_copy


def enrich_metadata_with_dates(
    df: pd.DataFrame,
    collection_date_col: str = "collection_date",
    target_col: Optional[str] = None,
    precision_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize collection dates in metadata DataFrame.

    Adds columns:
    - target_col: Standardized date in ISO 8601 format
    - precision_col: Date precision ('day', 'month', 'year', 'unknown')

    Args:
        df: Input DataFrame with dates
        collection_date_col: Column name containing collection dates
        target_col: Output column name
        precision_col: Column to store precision

    Returns:
        DataFrame with standardized dates
    """
    if collection_date_col not in df.columns:
        logger.warning(f"Column {collection_date_col} not found in DataFrame")
        return df

    if target_col is None:
        target_col = f"{collection_date_col}_standardized"

    df_copy = df.copy()

    # Parse dates with precision
    dates, precisions = zip(*[
        DateParser.parse_date(val, return_precision=True)
        for val in df_copy[collection_date_col]
    ])

    df_copy[target_col] = dates

    if precision_col:
        df_copy[precision_col] = precisions

    # Log summary
    valid_count = sum(1 for d in dates if d is not None)
    logger.info(f"Standardized dates for {valid_count}/{len(df_copy)} samples")

    return df_copy


def create_metadata_enrichment_pipeline(
    df: pd.DataFrame,
    extract_location: bool = True,
    standardize_dates: bool = True,
    collection_date_col: str = "collection_date",
) -> pd.DataFrame:
    """
    Create a complete metadata enrichment pipeline.

    Applies location extraction and date standardization in sequence.

    Args:
        df: Input metadata DataFrame
        extract_location: Whether to extract coordinates
        standardize_dates: Whether to standardize collection dates
        collection_date_col: Name of collection date column

    Returns:
        Enriched DataFrame with extracted/standardized metadata
    """
    df_result = df.copy()

    if extract_location:
        logger.info("Extracting geographic coordinates...")
        df_result = enrich_metadata_with_location(df_result)

    if standardize_dates:
        logger.info("Standardizing collection dates...")
        df_result = enrich_metadata_with_dates(
            df_result,
            collection_date_col=collection_date_col,
            precision_col=f"{collection_date_col}_precision",
        )

    logger.info("Metadata enrichment pipeline complete")

    return df_result
