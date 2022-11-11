import os
import sys
import numpy as np
from torch.utils.data.dataset import Dataset

from .augmentation import Transformer, RandomStride

class rPPGDataset(Dataset):
    """Dataset for rPPG.
    """

    def __init__(self, dataset_name:str, data_path: str, train: int, transforms: Transformer = None, vid_frame: int = 150, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (int): train: 1, valid: 0, test:-1
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride
        self.files = []
        self.dataset_init()   

    def dataset_init(self):
        print('Dataset: ', self.dataset_name)
        if self.dataset_name=='ubfc2':
            self.valid_fold = ['subject15', 'subject17', 'subject3', 'subject34', 'subject42', 'subject48', 'subject49', 'subject5']
            self.train_fold = [ f for f in os.listdir(self.data_path) if f not in self.valid_fold ]
        elif self.dataset_name=='ubfc1':
            self.valid_fold = ['after-exercise']
            self.train_fold = [ f for f in os.listdir(self.data_path) if f not in self.valid_fold ]
        elif self.dataset_name=='pure':
            self.valid_fold = ['09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '10-01', '10-02', '10-03', '10-04', '10-05', '10-06']
            self.train_fold = [ f for f in os.listdir(self.data_path) if f not in self.valid_fold]
        elif self.dataset_name=='cohface':
            self.valid_fold = ["3/0", "3/1", "8/0", "8/1", "10/0", "10/1", "11/0", "11/1", "12/0", "12/1", "13/0", "13/1", "14/0", "14/1", "15/0", "15/1", "20/0", "20/1", "22/0", "22/1", "23/0", "23/1", "26/0", "26/1", "30/0", "30/1", "32/0", "32/1", "34/0", "34/1", "40/0", "40/1"]
            self.train_fold = ["25/0", "25/1", "29/0", "29/1", "31/0", "31/1", "35/0", "35/1", "27/0", "27/1", "33/0", "33/1", "1/0", "1/1", "16/0", "16/1", "38/0", "38/1", "21/0", "21/1", "28/0", "28/1", "4/0", "4/1", "17/0", "17/1", "9/0", "9/1", "18/0", "18/1", "37/0", "37/1", "2/0", "2/1", "7/0", "7/1", "39/0", "39/1", "24/0", "24/1", "19/0", "19/1", "6/0", "6/1", "36/0", "36/1", "5/0", "5/1"]
        elif self.dataset_name=='ubfc-phys':
            self.valid_fold = ["s1", "s2", "s3", "s4", "s5"]
            self.train_fold = [ f for f in os.listdir(self.data_path) if f not in self.valid_fold]
        elif self.dataset_name=='vicar':
            self.valid_fold = ["09-base.mp4", "09-mov.mp4", "09-run.mp4", "10-base.mp4", "10-hrv.mp4", "10-mov.mp4", "10-run.mp4"]
            self.train_fold = [ f for f in os.listdir(self.data_path) if f not in self.valid_fold]
        else:
            print('unsupported dataset', self.dataset_name)
            sys.exit(1)
        assert len([x for x in self.valid_fold if x in self.train_fold]) == 0
        self.train_fold.sort()
        self.valid_fold.sort()
        
        if self.train == 1:
            for subject in self.train_fold:
                file_name = [ f for f in os.listdir(os.path.join(self.data_path, subject)) if f.endswith('.npz') ]
                self.files.extend([os.path.join(self.data_path, subject, f) for f in file_name])	

            print("{} of videos in {} train split".format(len(self.files), self.dataset_name))	

        elif self.train == 0:
            for subject in self.valid_fold:
                file_name = [ f for f in os.listdir(os.path.join(self.data_path, subject)) if f.endswith('.npz') ]
                self.files.extend([os.path.join(self.data_path, subject, f) for f in file_name])			

            print("Use subject {} as valid set.".format(self.valid_fold))
            print("{} of videos in {} valid".format(len(self.files), self.dataset_name))
            
        elif self.train == -1:
            for subject in self.train_fold + self.valid_fold:
                file_name = [ f for f in os.listdir(os.path.join(self.data_path, subject)) if f.endswith('.npz') ]
                self.files.extend([os.path.join(self.data_path, subject, f) for f in file_name])			

            print("Use subject {} as test set.".format(self.dataset_name))
            print("{} of videos in {} test".format(len(self.files), self.dataset_name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as f:
            data = f["frames"][:self.vid_frame:self.vid_frame_stride]
            label = f["wave"].astype(np.float32)
        if isinstance(self.transforms, RandomStride):            
            data_, label_spatial, label_temporal = self.transforms(data)
            sample = (data_, label, label_spatial, label_temporal)
        else:
            sample = (self.transforms(data), label)  
        return sample