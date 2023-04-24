import requests
import zipfile
import os
import glob
from pathlib import Path

import tqdm
import scipy.io
import numpy as np


def download_data(data_dir, url, unpack=True, block_size=10 * 1024):
    filename = os.path.join(data_dir, os.path.basename(url))
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(filename):
        print("{} already exists. Skipping download".format(filename))
        return

    print("Downloading {} to {}".format(url, filename))
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    progress_bar = tqdm.tqdm(total=total, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total != 0 and progress_bar.n != total:
        raise RuntimeError("Error downloading {}".format(url))

    if unpack and filename[-3:] == "zip":
        with open(filename, "rb") as f:
            with zipfile.ZipFile(f) as zip_ref:
                zip_ref.extractall(data_dir)
        print("Unzipped {} to {}".format(filename, data_dir))


def load_matlab_data(key, data_dir, *folders):
    folders = [data_dir] + list(folders) + ["*/*.mat"]
    examples, labels = [], []
    for filename in glob.glob(os.path.join(*folders)):
        examples.append(scipy.io.loadmat(filename)[key])
        labels.append(int(Path(filename).parts[-2]))
    return np.stack(examples), np.array(labels) - 1
