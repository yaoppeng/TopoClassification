import matplotlib.pyplot as plt
"""Dataset.

    Customize your dataset here.
"""

import os

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from batchgenerators.utilities.file_and_folder_operations import *
from torchvision import transforms


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def process_pd(pd, dims=None, samples=None, case=None):
    # print(samples)
    if dims is None:
        dims = [0, 1, 2]

    for dim in dims:
        if samples is not None:
            # print(pd[dim][(pd[dim][:, 1] - pd[dim][:, 0]).argsort()[::-1][:samples]])

            pd[dim] = pd[dim][(pd[dim][:, 1]-pd[dim][:, 0]).argsort()[::-1][:samples], :]

        # add position encoding
        # pd[dim] = pc_normalize(np.concatenate([pd[dim], np.ones((pd[dim].shape[0], 1), dtype=pd[dim].dtype) * dim], axis=1))
        pd[dim] = np.concatenate([pd[dim], np.ones((pd[dim].shape[0], 1), dtype=pd[dim].dtype) * dim], axis=1)

    pd = np.vstack(pd)
    # pd = pd[(pd[:, 1]-pd[:, 0]).argsort()[::-1][:samples], :]

    # pd = pd[(pd[:, 1] - pd[:, 0]).argsort()[::-1][:], :]
    pd = pc_normalize(pd)

    # print(case, pd.shape)
    return pd.transpose((1, 0))  # np.vstack(pd)
