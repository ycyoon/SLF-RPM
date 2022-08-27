import os

import numpy as np

from torch.utils.data.dataset import Dataset

from .augmentation import Transformer, RandomStride


class MAHNOBHCIDataset(Dataset):
    """Dataset for MAHNOB-HCI
    """
    
    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
        Args:
            data_path (str): Path to the dataset.
            train (bool): `True` to use train split and `False` to use test split.
            transforms (Transformer, optional): Data transformations to apply. Defaults to None.
            vid_frame (int, optional): Number of video frames. Defaults to 150.
            vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        #self.test_fold = [str(x) for x in [3, 4, 9, 11, 17, 27]]
        self.test_fold = ['Sessions_8', 'Sessions_6', 'Sessions_10', 'Sessions_24', 'Sessions_2', 'Sessions_30', 'Sessions_36', 'Sessions_18', 'Sessions_4', 'Sessions_2996', 'Sessions_810', 'Sessions_2346']
        self.train_fold = [subject for subject in os.listdir(data_path) if subject not in self.test_fold]
        
        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])	

            print("{} of videos in MAHNOB-HCI train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])		

            print("{} of videos in MAHNOB-HCI test split".format(len(self.files)))

        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"]

        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)

        return sample

class VIPLHRDataset(Dataset):
    """Dataset for VIPL-HR-V2.
    """
    
    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):
        """
        Args:
            data_path (str): Path to the dataset.
            train (bool): `True` to use train split and `False` to use test split.
            transforms (Transformer, optional): Data transformations to apply. Defaults to None.
            vid_frame (int, optional): Number of video frames. Defaults to 150.
            vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        self.test_fold = [250, 299, 105, 233, 50, 220, 368, 208, 432, 354, 435, 271, 425, 
                            405, 121, 332, 236, 185, 467, 273, 314, 86, 41, 304, 439, 219, 
                            239, 137, 209, 34, 36, 230, 265, 418, 414, 325, 387, 18, 161, 
                            55, 255, 315, 171, 40, 295, 125, 59, 444, 300, 9, 322, 89, 372, 
                            244, 98, 309, 485, 33, 346, 443, 441, 25, 136, 382, 114, 336, 30,
                             477, 498, 402, 202, 144, 56, 500, 491, 451, 78, 287, 222, 181, 37, 
                             187, 296, 487, 394, 475, 259, 142, 214, 328, 302, 134, 149, 482, 
                             410, 496, 247, 127, 190, 446]
        self.train_fold = [i for i in range(1, 501) if i not in self.test_fold]
        assert len([x for x in self.test_fold if x in self.train_fold]) == 0

        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = [f for f in os.listdir(data_path) if subject == int(f.split('_')[0])]
                self.files.extend([os.path.join(data_path, f) for f in file_name])	

            print("{} of videos in VIPL-HR-V2 train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = [f for f in os.listdir(data_path) if subject == int(f.split('_')[0])]
                self.files.extend([os.path.join(data_path, f) for f in file_name])		

            print("{} of videos in VIPL-HR-V2 test split".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"].astype(np.float32)

        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)
        return sample

class UBFCDataset(Dataset):
    """Dataset for UBFC-rPPG.
    """

    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (bool): `True` to use train split and `False` to use test split.
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        self.test_fold = ['subject15', 'subject17', 'subject3', 'subject34', 'subject42', 'subject48', 'subject49', 'subject5']
        self.train_fold = [ f for f in os.listdir(data_path) if f not in self.test_fold ]
        self.train_fold.sort()
        self.test_fold.sort()
        assert len([x for x in self.test_fold if x in self.train_fold]) == 0

        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])	

            print("{} of videos in UBFC-rPPG train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])			

            print("Use subject {} as test set.".format(self.test_fold))
            print("{} of videos in UBFC-rPPG test".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"].astype(np.float32)
        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)
        return sample

class PUREDataset(Dataset):
    """Dataset for UBFC-rPPG.
    """

    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (bool): `True` to use train split and `False` to use test split.
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        self.test_fold = ['10-01', '09-02', '08-03', '08-04', '09-05', '10-06']
        self.train_fold = [ f for f in os.listdir(data_path) if f not in self.test_fold ]
        self.train_fold.sort()
        self.test_fold.sort()
        assert len([x for x in self.test_fold if x in self.train_fold]) == 0

        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])	

            print("{} of videos in UBFC-rPPG train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])			

            print("Use subject {} as test set.".format(self.test_fold))
            print("{} of videos in UBFC-rPPG test".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"].astype(np.float32)
        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)
        return sample

class CohfaceDataset(Dataset):
    """Dataset for Cohface
    """

    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (bool): `True` to use train split and `False` to use test split.
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        self.test_fold = ["17_0", "17_1", "17_2", "17_3", "9_0", "9_1", "9_2", "9_3", "18_0", "18_1", "18_2", "18_3", "37_0", "37_1", "37_2", "37_3", "2_0", "2_1", "2_2", "2_3", "7_0", "7_1", "7_2", "7_3", "39_0", "39_1", "39_2", "39_3", "24_0", "24_1", "24_2", "24_3", "19_0", "19_1", "19_2", "19_3", "6_0", "6_1", "6_2", "6_3", "36_0", "36_1", "36_2", "36_3", "5_0", "5_1", "5_2", "5_3"]
        self.train_fold = [ f for f in os.listdir(data_path) if f not in self.test_fold ]
        self.train_fold.sort()
        self.test_fold.sort()
        assert len([x for x in self.test_fold if x in self.train_fold]) == 0

        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])	

            print("{} of videos in Cohface train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])			

            print("Use subject {} as test set.".format(self.test_fold))
            print("{} of videos in Cohface test".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"].astype(np.float32)
        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            assert len(data) != 0, self.files[idx]
            sample = (self.transforms(data), label)
        return sample

class MergedDataset(Dataset):
    """Dataset for UBFC-rPPG.
    """

    def __init__(self, data_path: str, train: bool, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (bool): `True` to use train split and `False` to use test split.
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride

        #self.test_fold = ['subject15', 'subject17', 'subject3', 'subject34', 'subject42', 'subject48', 'subject49', 'subject5'] + [str(x) for x in [3, 4, 9, 11, 17, 27]] + ['10-01', '09-02', '08-03', '08-04', '09-05', '10-06']
        self.test_fold = ['subject15', 'subject17', 'subject3', 'subject34', 'subject42', 'subject48', 'subject49', 'subject5'] + \
             ['10-01', '09-02', '08-03', '08-04', '09-05', '10-06']
        self.train_fold = [ f for f in os.listdir(data_path) if f not in self.test_fold ]
        self.train_fold.sort()
        self.test_fold.sort()
        assert len([x for x in self.test_fold if x in self.train_fold]) == 0

        self.files = []
        if self.train:
            for subject in self.train_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])	

            print("{} of videos in merged train split".format(len(self.files)))	

        else:
            for subject in self.test_fold:
                file_name = os.listdir(os.path.join(data_path, subject))
                self.files.extend([os.path.join(data_path, subject, f) for f in file_name])			

            print("Use subject {} as test set.".format(self.test_fold))
            print("{} of videos in merged test".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["hr"].astype(np.float32)
        if isinstance(self.transforms, RandomStride):
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)
        return sample