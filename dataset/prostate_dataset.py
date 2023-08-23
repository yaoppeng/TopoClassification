from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io
import pandas as pd
from torch.utils.data import Dataset


class ProstateDataset(Dataset):
    def __init__(self, img_path, label_file, transform, fold, train):
        super().__init__()
        split_file = join(os.getenv('HOME'), "data/raw_data/Prostate/splits_5_fold.pkl")
        splits = load_pickle(split_file)
        self.train = train

        self.cases = splits[fold]['train'] if train else splits[fold]['val']

        self.df = pd.read_csv(label_file, header=0, dtype={"roi_lable": str, "grade": int})
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, i):
        roi_label = self.cases[i]
        img = io.imread(join(self.img_path, f"{roi_label}.tiff"))

        if self.transform is not None:
            img = self.transform(img)

        grade = self.df[self.df["roi_label"]==roi_label]["grade"].tolist()[0]
        # roi_label, grade = self.df.loc[i, "roi_label"], self.df.loc[i, "grade"]

        grade = grade - 3  # class 3, 4, 5 -> 0, 1, 2

        return img, grade

    def __len__(self):
        return len(self.cases)

