# Modified based on https://pytorch.org/vision/stable/transforms.html#
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

class Transformer:
    """Data transformer for SLF-RPM
    """
    def __init__(self, transform: list, mean: float, std: float) -> None:
        """
        Args:
            transform (list): List of augmentations to apply.
            mean (float): Mean value for data normalisation.
            std (float): Std value for data normalisation.
        """
        super().__init__()

        if mean is None or std is None:
            self.transform = transforms.Compose([*transform, ToTensor()])

        elif transform is not None:
            self.transform = transforms.Compose(
                [*transform, Normalise(mean=mean, std=std)]
            )
        else:
            self.transform = transforms.Compose(
                [ToTensor(), Normalise(mean=mean, std=std)]
            )

    def __call__(self, vid_seq):
        return self.transform(vid_seq)


########################
# Spatial augmentations
########################
class RandomROI:
    """Random selected ROIs
    """

    def __init__(self, roi_list):
        self.roi_list = roi_list
        assert (
            min(self.roi_list) >= 0 and max(self.roi_list) <= 6
        ), "Invalid ROI list range!"

    def __call__(self, vid_seq):
        self.roi_idx = torch.randint(0, len(self.roi_list), (1,)).item()
        idx = self.roi_list[self.roi_idx]

        return vid_seq[:, idx, :]


########################
# Temporal augmentations
########################
class RandomStride:
    def __init__(
        self, stride_list: list, n_frame: int, base_transform: transforms.Compose
    ) -> None:
        self.stride_list = stride_list
        self.n_frame = n_frame
        self.base_transform = base_transform

    def __call__(self, vid_seq):
        _, _, h, w, c = vid_seq.shape
        vid_aug = [
            torch.empty((self.n_frame, h, w, c)),
            torch.empty((self.n_frame, h, w, c)),
        ]
        fn_idx = torch.randperm(2)
        strides = []
        rois = []

        for fn_id in fn_idx:
            idx = torch.randint(0, len(self.stride_list), (1,)).item()
            stride = self.stride_list[idx]
            #assert (vid_seq.shape[0] // stride) >= self.n_frame, "%d %d %d" % (vid_seq.shape[0], stride, self.n_frame)
            vid_aug[fn_id] = self.base_transform(vid_seq[::stride][: self.n_frame])
            
            if vid_aug[fn_id].shape[0] < self.n_frame:
                #reverse order of vid_aug[fn_id]
                vid_aug_reverse = torch.flip(vid_aug[fn_id], [0])
                #copy vid_aug[fn_id]
                vid_aug_right = vid_aug[fn_id].clone()
                
                #repeat vid_aug[fn_id] to make it have the same length as self.n_frame
                for i in range(self.n_frame // vid_aug[fn_id].shape[0] + 1):
                    if i % 2 == 1:
                        vid_aug[fn_id] = torch.cat((vid_aug[fn_id], vid_aug_right), 0)
                    else:
                        vid_aug[fn_id] = torch.cat((vid_aug[fn_id], vid_aug_reverse), 0)

                vid_aug[fn_id] = vid_aug[fn_id][:self.n_frame]
                
                #vid_aug\
                #    [fn_id] = torch.cat([vid_aug[fn_id], vid_aug[fn_id][-1].unsqueeze(0).repeat(self.n_frame - vid_aug[fn_id].shape[0], 1, 1, 1)], dim=0)

                #vid_aug[fn_id] = torch.cat([vid_aug[fn_id], torch.zeros(self.n_frame - vid_aug[fn_id].shape[0], c, h, w)], dim=0)
            
            roi_idx = [
                t.roi_idx
                for t in self.base_transform.transform.transforms
                if isinstance(t, RandomROI)
            ][0]
            strides.append(idx)
            rois.append(roi_idx)

        return vid_aug, rois, strides


########################
# General augmentations
########################
class Normalise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid_seq):
        return torch.stack(
            [F.normalize(img, mean=self.mean, std=self.std) for img in vid_seq]
        )

#ycyoon rppg_toolbox의 standard, normalized 두개 적용
class StdAndNormalise:
    @staticmethod
    def diff_normalize_data(data):
        """Difference frames and normalization data"""
        n, h, w, c = data.shape
        #normalized_len = n - 1
        normalized_len = n
        normalized_data = torch.zeros((normalized_len, h, w, c))
        for j in range(normalized_len - 1):
            normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        normalized_data = normalized_data / torch.std(normalized_data)
        normalized_data[torch.isnan(normalized_data)] = 0
        return normalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Difference frames and normalization labels"""
        diff_label = torch.diff(label, axis=0)
        normalized_label = diff_label / torch.std(diff_label)
        normalized_label[torch.isnan(normalized_label)] = 0
        return normalized_label

    @staticmethod
    def standardized_data(data):
        """Difference frames and normalization data"""
        data = data - torch.mean(data)
        data = data / torch.std(data)
        data[torch.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        label = label - torch.mean(label)
        label = label / torch.std(label)
        label[torch.isnan(label)] = 0
        return label

    def __call__(self, vid_seq):
        # data_type
        data = list()
        for data_type in ['Normalized', 'Standardized']:
            f_c = vid_seq.clone()
            if data_type == "Raw":
                data.append(f_c[:-1, :, :, :])
            elif data_type == "Normalized":
                data.append(self.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                #data.append(self.standardized_data(f_c)[:-1, :, :, :])
                data.append(self.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = torch.concat(data, axis=1)
        return data

class ToTensor:
    def __call__(self, vid_seq):
        try:
            a = [F.to_tensor(img) for img in vid_seq]
            torch.stack(a)
        except Exception as e:
            raise
        return torch.stack([F.to_tensor(img) for img in vid_seq])

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, vid_seq):
        try:
            a = [F.to_tensor(F.resize(F.to_pil_image(img),self.size)) for img in vid_seq]
            a = torch.stack(a)
        except Exception as e:
            raise
        return a