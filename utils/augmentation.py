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
                [*transform, ToTensor(), Normalise(mean=mean, std=std)]
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
            #zero padding
            if vid_aug[fn_id].shape[0] < self.n_frame:
                #extend last frame
                vid_aug[fn_id] = torch.cat([vid_aug[fn_id], vid_aug[fn_id][-1].unsqueeze(0).repeat(self.n_frame - vid_aug[fn_id].shape[0], 1, 1, 1)], dim=0)

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


class ToTensor:
    def __call__(self, vid_seq):
        try:
            a = [F.to_tensor(img) for img in vid_seq]
            torch.stack(a)
        except Exception as e:
            raise
        return torch.stack([F.to_tensor(img) for img in vid_seq])
