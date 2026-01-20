# ==================================================================================== #

# Standard Imports
import json
from pathlib import Path
from typing import Dict, List, Union

# ==================================================================================== #

def is_bbox_global(bbox, tolerance=10):
    """Checks if a bounding box is global.

    Args:
        bbox:      The bounding box, e.g., [-180, -90, 180, 90].
        tolerance: The allowed deviation from the full range for longitude and latitude.

    Returns:
        bool: True if the bounding box is global, False otherwise.
    """
    if not bbox or len(bbox) != 4:
        return False

    lon_min, lat_min, lon_max, lat_max = bbox
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Check if the longitude range is close to 360 and latitude range is close to 180
    if (360 - tolerance) <= lon_range <= (360 + tolerance) and \
       (180 - tolerance) <= lat_range <= (180 + tolerance):
        return True
    return False


def find_global_datasets(catalog_file: Union[str, Path] = '/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google_earth_engine/resources/catalog.json') -> List[Dict]:
    """Finds global datasets from a JSON catalog file.

    Args:
        catalog_file: The path to the catalog JSON file.

    Returns:
        list: A list of dictionaries, each containing the title and ID of a global dataset.
    """
    try:
        with open(catalog_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{catalog_file}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{catalog_file}'.")
        return []

    global_datasets = []
    global_keywords = ['global', 'world', 'worldwide']

    for asset in data:
        is_global = False
        title = asset.get('title', 'No Title')
        description = asset.get('description', '')
        asset_id = asset.get('id', 'No ID')

        # 1. Check for keywords in title and description
        if any(keyword in title.lower() for keyword in global_keywords) or \
           any(keyword in description.lower() for keyword in global_keywords):
            is_global = True

        # 2. Check the spatial extent (bounding box)
        try:
            # The bbox can be nested inside a list
            bbox = asset['extent']['spatial']['bbox'][0]
            if is_bbox_global(bbox): is_global = True
        except (KeyError, IndexError, TypeError):
            # Asset does not have a valid bounding box, so we skip this check
            pass

        # If identified as global and not already added, add it to our list
        if is_global:
            dataset_info = {'title': title, 'id': asset_id}
            if dataset_info not in global_datasets:
                global_datasets.append(dataset_info)

    return global_datasets

# ==================================================================================== #

if __name__ == "__main__":
    CATALOG_FILE = '/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/api/environmental_data/google_earth_engine/resources/catalog.json'
    found_datasets = find_global_datasets(CATALOG_FILE)

    if found_datasets:
        print("✅ Found the following global environmental datasets:")
        print("-" * 50)
        for i, dataset in enumerate(found_datasets, 1):
            print(f"{i}. {dataset['title']}")
            print(f"   (ID: {dataset['id']})\n")
    else:
        print("❌ No global datasets were found based on the criteria.")