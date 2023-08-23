import os.path

from dataset.pd_utils import *
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from einops import repeat
import cv2


class ISICDataset(Dataset):
    def __init__(self, img_path, transform, train, fold, label_file=None):
        """SKin Lesion"""
        self.img_path = img_path
        self.transform = transform
        self.train = train
        self.pd_dir = join(self.img_path, "persistent_diagram_1_old")
        self.pl_dir = join(self.img_path, "persistent_landscape_old")
        self.data, self.targets = self.get_data(fold)
        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))
        self.target_img_dict = {}
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def __getitem__(self, i):
        """
                Args:
                    index (int): Index
                Returns:
                    tuple: (sample, target) where target is class_index of the
                           target class.
                """
        path = self.data[i]

        case = os.path.basename(path).split(".")[0]
        target = self.targets[i]
        img = io.imread(path)
        # img = pil_loader(path)

        pd = []
        pl = []
        # for mode in ['r', 'g', 'b', 'gray', 'r_inverse', 'g_inverse', 'b_inverse', 'gray_inverse']:
        #     for d in [0, 1]:
        #         pd.append(np.load(join(self.pd_dir, case, f'{mode}_{d}.npy')))
        #         pl.append(np.load(join(self.pl_dir, case, f'{mode}_{d}.npy')))

        for d in [0, 1]:
            pd.append(np.load(join(self.pd_dir, case, f'dim_{d}.npy')))
            pl.append(np.load(join(self.pl_dir, case, f'dim_{d}.npy')))

        # pd = [np.load(join(self.pd_dir, case, f'dim_{x}.npy')) for x in range(0, 2)]
        # pd = process_pd(pd, dims=[0, 1], samples=73, case=case)

        pd = process_pd(pd, dims=[0, 1], samples=85, case=case)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, pd, np.vstack(pl)

        # return img, target, pd, np.vstack(pl), \
        #     cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), path

    def __len__(self):
        return len(self.data)

    def get_data(self, fold):

        if self.train:
            csv = f'split_data/split_data_{fold}_fold_train.csv'
        else:
            csv = f'split_data/split_data_{fold}_fold_test.csv'

        fn = os.path.join(self.img_path, csv)
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            data.append(os.path.join(self.img_path,
                                     "ISIC2018_Task3_Training_Input", path))
            targets.append(label)

        return data, targets


if __name__ == "__main__":
    data = ISICDataset()
    dataloader = DataLoader(data, batch_size=2)
    for item in dataloader:
        print(item)
