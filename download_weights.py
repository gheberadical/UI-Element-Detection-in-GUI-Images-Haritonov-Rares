"""Download results_v2/ (model weights + evaluation results) from Google Drive.

Usage:
    python download_weights.py

Requires: pip install gdown
"""

import subprocess
import sys


FOLDER_ID = "1LoIImexr0NsWZ9jEeUuft6FpXs5V475h"


def main():
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print("Downloading results_v2/ from Google Drive...")
    gdown.download_folder(
        id=FOLDER_ID,
        output="results_v2",
        quiet=False,
        resume=True,
    )
    print("\nDone. Model weights are in results_v2/*/weights/best.pt")


if __name__ == "__main__":
    main()
