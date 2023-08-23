import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io
import pandas as pd
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    def __init__(self, img_path, label_file, transform, fold, train):
        super().__init__()
        split_file = join(os.getenv('HOME'), "data/raw_data/Glaucoma/splits_3_fold.json")
        splits = load_json(split_file)
        self.train = train

        if fold is None:
            self.cases = np.concatenate([splits[0]['train'], splits[0]['val']])
        else:
            self.cases = splits[fold]['train'] if train else splits[fold]['val']

        self.df = pd.read_csv(label_file, header=0, dtype={"img": str, "label": int})
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, i):
        case = self.cases[i]
        img = io.imread(join(self.img_path, f"{case}.jpg"))

        if self.transform is not None:
            img = self.transform(img)
        if img.shape[-1] == 242:
            print(case)
        # print(img.shape[-1] == 242)
        label = self.df[self.df["img"] == case]["label"].tolist()[0]

        return img, label

    def __len__(self):
        return len(self.cases)
