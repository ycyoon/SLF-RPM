from email.mime import image
import enum
import os
import numpy as np
import cv2
import random
from itertools import permutations
from natsort import natsorted # pip install natsort

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

        self.test_fold = [str(x) for x in [3, 4, 9, 11, 17, 27]]
        #self.test_fold = ['Sessions_8', 'Sessions_6', 'Sessions_10', 'Sessions_24', 'Sessions_2', 'Sessions_30', 'Sessions_36', 'Sessions_18', 'Sessions_4', 'Sessions_2996', 'Sessions_810', 'Sessions_2346']
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

class UBFC2Dataset(Dataset):
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

class UBFC1Dataset(Dataset):
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

        self.test_fold = ['DATASET_1_after-exercise']
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

        self.test_fold = ['09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '10-01', '10-02', '10-03', '10-04', '10-05', '10-06']
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

        self.test_fold = ["3_0", "3_1", "8_0", "8_1", "10_0", "10_1", "11_0", "11_1", "12_0", "12_1", "13_0", "13_1", "14_0", "14_1", "15_0", "15_1", "20_0", "20_1", "22_0", "22_1", "23_0", "23_1", "26_0", "26_1", "30_0", "30_1", "32_0", "32_1", "34_0", "34_1", "40_0", "40_1"]
        self.train_fold = ["25_0", "25_1", "29_0", "29_1", "31_0", "31_1", "35_0", "35_1", "27_0", "27_1", "33_0", "33_1", "1_0", "1_1", "16_0", "16_1", "38_0", "38_1", "21_0", "21_1", "28_0", "28_1", "4_0", "4_1", "17_0", "17_1", "9_0", "9_1", "18_0", "18_1", "37_0", "37_1", "2_0", "2_1", "7_0", "7_1", "39_0", "39_1", "24_0", "24_1", "19_0", "19_1", "6_0", "6_1", "36_0", "36_1", "5_0", "5_1"]
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

        pure_dataset = ['09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '10-01', '10-02', '10-03', '10-04', '10-05', '10-06']
        mahnob = [str(x) for x in [3, 4, 9, 11, 17, 27]]
        ubfc2 = ['subject15', 'subject17', 'subject3', 'subject34', 'subject42', 'subject48', 'subject49', 'subject5']
        cohface = ["3_0", "3_1", "8_0", "8_1", "10_0", "10_1", "11_0", "11_1", "12_0", "12_1", "13_0", "13_1", "14_0", "14_1", "15_0", "15_1", "20_0", "20_1", "22_0", "22_1", "23_0", "23_1", "26_0", "26_1", "30_0", "30_1", "32_0", "32_1", "34_0", "34_1", "40_0", "40_1"]
        ubfc1 = ['DATASET_1_after-exercise']
        self.test_fold = pure_dataset + ubfc2 +  ubfc1
        self.train_fold = [ f for f in os.listdir(data_path) if f not in self.test_fold and '_' not in f]
        #cohface_train = ["25_0", "25_1", "29_0", "29_1", "31_0", "31_1", "35_0", "35_1", "27_0", "27_1", "33_0", "33_1", "1_0", "1_1", "16_0", "16_1", "38_0", "38_1", "21_0", "21_1", "28_0", "28_1", "4_0", "4_1", "17_0", "17_1", "9_0", "9_1", "18_0", "18_1", "37_0", "37_1", "2_0", "2_1", "7_0", "7_1", "39_0", "39_1", "24_0", "24_1", "19_0", "19_1", "6_0", "6_1", "36_0", "36_1", "5_0", "5_1"]
        #self.train_fold += cohface_train
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
            assert len(data) == self.vid_frame/self.vid_frame_stride, self.files[idx]
            sample = (self.transforms(data), label)
        return sample


class FERAugDataset(Dataset):
    """Dataset for FER dataset based Self-training learning
    """

    def __init__(self, data_path: str, masknum: int = 5, transforms: Transformer = None, vid_frame: int = 30, vid_frame_stride: int = 1):	
        """
            Args:
                data_path (str): Path to the dataset.
                train (bool): `True` to use train split and `False` to use test split.
                transforms (Transformer, optional): Data transformations to apply. Defaults to None.
                vid_frame (int, optional): Number of video frames. Defaults to 150.
                vid_frame_stride (int, optional): Number of video stride. Defaults to 1.
        """
        self.data_path = data_path
        self.transforms = transforms
        self.vid_frame = vid_frame
        self.vid_frame_stride = vid_frame_stride
        self.masknum = masknum
        self.task = 0 # task 0: shuffle, task 1: histogram prediction
        labels = list(permutations(list(range(masknum)),masknum))
        self.label_dict = {}
        for i, l in enumerate(labels):
            self.label_dict['_'.join(map(str,l))] = i

        self.fold = [ f for f in os.listdir(data_path)]
        self.videos = []
        #print('total fold : ', len(self.fold))
        for subject in self.fold:
            file_names = os.listdir(os.path.join(data_path, subject))
            images = []
            file_names = natsorted(file_names)
            for i in range(0, len(file_names), vid_frame_stride):
                images.append(cv2.resize(cv2.imread(os.path.join(data_path, subject, file_names[i])), (112,112)))
            if len(images) == 0:
                continue
            elif len(images) <= self.vid_frame/self.vid_frame_stride:
                #extend images
                #배율
                extend_num = int(self.vid_frame/self.vid_frame_stride/len(images)) + 1
                extend_images = []
                for i in range(len(images)):
                    for j in range(extend_num):
                        extend_images.append(images[i])
                images = extend_images
                #print('too short', os.path.join(data_path, subject), len(images), '<', self.vid_frame/self.vid_frame_stride)
            self.videos.append(np.asarray(images))

        print("{} of videos in the split".format(len(self.videos)))	

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx: int):
        # 랜덤하게 150 frame 시작 위치 설정
        try:
            randidx = random.randint(0, len(self.videos[idx])-self.vid_frame/self.vid_frame_stride)
        except:
            print(len(self.videos[idx]), self.vid_frame/self.vid_frame_stride)
            raise
        video = self.videos[idx][randidx:int(randidx+self.vid_frame/self.vid_frame_stride)]
        #나중에 gan으로 생성하는걸로 바꾸기?
        if self.task == 0:
            #suffle video
            if random.random() > 0.5:
                target = 0
                np.random.shuffle(video)
                return (self.transforms(video), target)
            else:
                target = 1
                return (self.transforms(video), target)
        else:
            histcat = np.zeros(256*3*5)
            for l in range(-5,0):
                img = np.copy(video[l])
                colors = ('b','g','r')
                hists = np.zeros(256*3)
                for i,color in enumerate(colors):
                    hist = cv2.calcHist([img],[i],None,[256],[0,256])
                    #cv2.normalize(hist, hist)
                    hists[i*256:(i+1)*256] = hist.flatten()
                histcat[(l+5)*256*3:(l+6)*256*3] = hists
                return (self.transforms(video), histcat)


        # b. vid_frame/vid_frame_stride * i 위치의 이미지 프레임들 뽑아서 랜덤하게 섞기
        # flag_images = []
        # for i in range(0, len(video)-5, int((self.vid_frame/self.vid_frame_stride)/self.masknum)):
        #     flag_images.append(np.copy(video[i]))        

        # flag_idx = [i for i in range(len(flag_images))]
        # random.shuffle(flag_idx)
        # for i, idx in enumerate(flag_idx):
        #     video[int((self.vid_frame/self.vid_frame_stride)/self.masknum)*i] = np.copy(flag_images[flag_idx[i]])
        # # c. 바뀐 위치를 클래스 레이블로 반환
        # label = str(flag_idx[0])
        # for i in range(1, len(flag_idx)):
        #     label += '_' + str(flag_idx[i])
        
        # histcat = np.zeros(256*3*5)
        # for l in range(-5,0):
        #     img = np.copy(video[l])
        #     colors = ('b','g','r')
        #     hists = np.zeros(256*3)
        #     for i,color in enumerate(colors):
        #         hist = cv2.calcHist([img],[i],None,[256],[0,256])
        #         #cv2.normalize(hist, hist)
        #         hists[i*256:(i+1)*256] = hist.flatten()
        #     histcat[(l+5)*256*3:(l+6)*256*3] = hists

        # # flatten the histogram
        # target = self.label_dict[label]
        # sample = (self.transforms(video), target, histcat)
        # return sample
