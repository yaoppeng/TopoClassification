import os.path

from dataset.pd_utils import *
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
from einops import repeat
import cv2


class MammographyDataset(Dataset):
    def __init__(self, img_path, label_file, transform, fold, train, use_inverse_topo=False):
        super().__init__()
        split_file = join(os.getenv('HOME'), f"data/raw_data/CBIS_DDSM/calc_splits_3_fold.json")
        splits = load_json(split_file)
        self.train = train

        train_df = pd.read_csv(join(os.getenv('HOME'), "data/raw_data/CBIS_DDSM/split_data_2_fold_train.csv"))
        val_df = pd.read_csv(join(os.getenv('HOME'), "data/raw_data/CBIS_DDSM/split_data_2_fold_test.csv"))

        if fold is None:
            self.cases = np.concatenate([splits[0]['train'], splits[0]['val']])
            print(f"all cases: {len(self.cases)}")
        else:
            if train:
                self.cases = train_df.image_path.tolist()
            else:
                self.cases = val_df.image_path.tolist()
            # self.cases = splits[fold]['train'] if train else splits[fold]['val']
            if train:
                print(f"train cases: {len(self.cases)}")
            else:
                print(f"val cases: {len(self.cases)}")

        self.img_arr = [io.imread(join(img_path, case)) for case in self.cases]

        # self.df = pd.read_csv(label_file, header=0, dtype={"img": str, "label": int})
        if train:
            self.df = train_df
        else:
            self.df = val_df

        self.img_path = img_path
        self.transform = transform
        self.class_name = {'BENIGN': 0, 'MALIGNANT': 1}

        pd_base_dir = join(os.getenv('HOME'), "data/raw_data/CBIS_DDSM")
        if use_inverse_topo:
            self.pd_dir = join(pd_base_dir, "persistent_diagram")
            self.pl_dir = join(pd_base_dir, "persistent_landscape")
        else:
            self.pd_dir = join(pd_base_dir, "persistent_diagram_old")
            self.pl_dir = join(pd_base_dir, "persistent_landscape")

    def __getitem__(self, i):
        case = self.cases[i]
        # img = io.imread(join(self.img_path, case))
        img = self.img_arr[i]
        img = img / 65535  # uint16 -> float
        img = img.astype(np.float32)
        # img = (img - img.min()) / (img.max() - img.min())

        # img = (img * 255).astype(np.uint8)

        # ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        # img = img * thr
        if len(img.shape) == 2:
            img = repeat(img, 'h w -> h w 3')

        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape[-1] == 242)
        # label = self.df[self.df["image_path"] == case]["pathology_id"].tolist()[0]
        label = self.class_name[self.df[self.df["image_path"] == case]["pathology"].tolist()[0]]

        # load persistent diagram
        pd_case_dir = os.path.basename(case).split(".")[0]
        pd = [np.load(join(self.pd_dir, pd_case_dir, f'dim_{x}.npy')) for x in range(0, 2)]
        # # pd = process_pd(pd, dims=[0, 1], samples=73, case=case)
        pd = process_pd(pd, dims=[0, 1], samples=150, case=case)
        #
        pl = [np.load(join(self.pl_dir, pd_case_dir, f'gray_{x}.npy')) for x in range(0, 2)]
        pl = process_pd(pl, dims=[0, 1], samples=150, case=case)
        # print(img.max(), img.min())

        return img, label, pd, pl

    def __len__(self):
        return len(self.cases)

