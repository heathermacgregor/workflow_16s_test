# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from typing import Any, Union
import requests

# Local Imports
from workflow_16s import constants

# ==================================== FUNCTIONS ===================================== #

def get_citation(
    doi_url: str, 
    style: str = 'apa', 
    email: str = constants.DEFAULT_EMAIL
) -> Union[str, None]:
    """Fetches a formatted citation for a given DOI using the Crossref API.

    Args:
        doi_url: A full DOI URL (e.g., "https://doi.org/10.1000/xyz123").
        style:   Citation style to use (e.g., "apa", "mla", "chicago"). Defaults to "apa".
        email:   Contact email for API requests (used in User-Agent header). Defaults 
                 to the value in constants.

    Returns:
        The formatted citation as a string if successful, None if an HTTP error occurred,
        or an error message string if another request error occurred.
    """
    # Extract the DOI from the URL
    doi = doi_url.split('doi.org/')[-1].strip('/')
    
    # Prepare the request URL and headers
    url = f'https://doi.org/{doi}'
    headers = {
        'Accept': f'text/x-bibliography; style={style}',
        'User-Agent': f'AcademicScript/1.0 (mailto:{email})'  
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.text.strip()
    except requests.exceptions.HTTPError as e:
        #return f"Error: {e}"
        return None
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
        return None
        

def print_structure(
    obj: Any, 
    indent: int = 0, 
    _key: str = "root"
) -> None:
    """Recursively prints the structure of a nested Python object (dicts and lists) 
    in a tree format.

    Args:
        obj:    The object to inspect (typically a nested dict or list).
        indent: The current indentation level (used internally for recursion).
        _key:   The key or index being printed (used internally for recursion).
    """
    spacer = " " * indent
    tname = type(obj).__name__
    print(f"{spacer}{'|-- ' if indent else ''}{_key} ({tname})")
    if isinstance(obj, dict):
        for k, v in obj.items():
            print_structure(v, indent + 4, k)
    elif isinstance(obj, list) and obj:
        print_structure(obj[0], indent + 4, "[0]")
