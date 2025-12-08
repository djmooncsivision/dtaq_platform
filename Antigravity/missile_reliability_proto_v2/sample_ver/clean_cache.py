
import shutil
import os
import sys

# This script is designed to be run on Windows.
# It clears the PyTensor cache directory.

try:
    # Construct the path to the PyTensor cache directory
    # Usually %LOCALAPPDATA%\\PyTensor
    local_app_data = os.getenv('LOCALAPPDATA')
    if not local_app_data:
        # Fallback for older systems or unusual configurations
        home_dir = os.path.expanduser('~')
        local_app_data = os.path.join(home_dir, 'AppData', 'Local')

    if not os.path.isdir(local_app_data):
        print(f"Error: Local AppData directory not found at {local_app_data}")
        sys.exit(1)

    pytensor_dir = os.path.join(local_app_data, 'PyTensor')

    if os.path.exists(pytensor_dir):
        print(f"Found PyTensor cache directory at: {pytensor_dir}")
        print("Attempting to delete...")
        shutil.rmtree(pytensor_dir)
        print("PyTensor cache directory deleted successfully.")
    else:
        print(f"PyTensor cache directory not found at: {pytensor_dir}")
        print("No action taken.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

sys.exit(0)
