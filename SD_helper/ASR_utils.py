import torch
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import shutil
import tarfile
from zipfile import ZipFile

import IPython.display as ipd

from torchaudio._internal import download_url_to_file

MUSAN_URL = 'https://www.openslr.org/resources/17/musan.tar.gz'
RIRS_URL = 'https://www.openslr.org/resources/28/rirs_noises.zip'

def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

def _download_gz(data_url: str, 
              dest_directory: str) -> None:
    """
    Download data files from given data url.

    Args:
    data_url: str
    dest_directory: str
        where your files will be at.
    """
    filename = os.path.split(data_url)[-1]

    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filepath = os.path.join(dest_directory, filename)

    download_url_to_file(data_url, filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def _download_zip(data_url: str, 
              dest_directory: str) -> None:
    """
    Download data files from given data url.

    Args:
    data_url: str
    dest_directory: str
        where your files will be at.
    """
    filename = os.path.split(data_url)[-1]

    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filepath = os.path.join(dest_directory, filename)

    download_url_to_file(data_url, filepath)
    with ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_directory)

def _init_weight(m):
    nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)