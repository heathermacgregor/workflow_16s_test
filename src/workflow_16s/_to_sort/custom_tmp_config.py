import os

# Step 1: Set environment variables
os.environ['TMPDIR'] = '/opt/tmp/macgregor'
os.environ['TMP'] = '/opt/tmp/macgregor'
os.environ['TEMP'] = '/opt/tmp/macgregor'

# Step 3: Ensure directory exists
os.makedirs('/opt/tmp/macgregor', exist_ok=True)
os.chmod('/opt/tmp/macgregor', 0o1777)  # Adjust permissions for security

# Step 2: Configure tempfile module
import tempfile
tempfile.tempdir = '/opt/tmp/macgregor'
