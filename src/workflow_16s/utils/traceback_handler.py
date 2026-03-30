import traceback
import functools
import logging

logger = logging.getLogger(__name__)

def catch_and_trace(func):
    """
    Forensic decorator that captures and logs full stack traces 
    for downstream analysis modules.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Capture the full forensic record
            tb = traceback.format_exc()
            logger.error(f"❌ Critical failure in module '{func.__name__}': {str(e)}")
            logger.error(f"--- START TRACEBACK ---\n{tb}--- END TRACEBACK ---")
            
            # We raise the error so the workflow manager handles the 
            # cleanup/halt, but now we have the 'why' in the logs.
            raise e 
    return wrapper
