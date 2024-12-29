import os
import subprocess
from typing import List

import pytest


@pytest.fixture
def download_weights(output_directory: str = "artifacts") -> None:
    version_base_urls: dict = {
        "2": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/",
        "2.1": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/",
    }

    file_suffixes: List[str] = [
        "hiera_tiny.pt",
        "hiera_small.pt",
        "hiera_base_plus.pt",
        "hiera_large.pt",
    ]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for version, base_url in version_base_urls.items():
        for suffix in file_suffixes:
            file_name = f"sam{version}_{suffix}"
            file_path = os.path.join(output_directory, file_name)
            if not os.path.exists(file_path):
                url = f"{base_url}{file_name}"
                command = ["wget", url, "-P", output_directory]
                try:
                    result = subprocess.run(
                        command,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    print(f"Download of {file_name} completed successfully.")
                    print(result.stdout.decode())
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred during the download of {file_name}.")
                    print(e.stderr.decode())
            else:
                print(f"{file_name} already exists. Skipping download.")
