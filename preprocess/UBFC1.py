import csv
import json
import os
from preprocess.BaseLoader import BaseLoader
import cv2
import numpy as np
import imageio as iio


class UBFC1Preprocess(BaseLoader):
    
    def __init__(self, data_path, des_path_root, face_size=128, frame_length=180, debug=False, detect_face_every_time=False):
        """Initializes the data loader.
        """    
        super(UBFC1Preprocess, self).__init__(data_path, des_path_root, face_size, frame_length, debug, detect_face_every_time)
        
    def extract_session(self, session, device):
        """Extracts frames from the given session."""
        print("Processing session: ", session)
        # Sessions
        
        subject_path = os.path.join(self.data_path, session)

        video_file_path = None
        mata_data_file_path = None

        # File paths
        for file in os.listdir(subject_path):
            if file.endswith(".avi"):
                video_file_path = os.path.join(subject_path, file)
            elif file.endswith(".xmp"):
                mata_data_file_path = os.path.join(subject_path, file)

        if video_file_path is None or mata_data_file_path is None:
            raise OSError("Files are incomplete in {}".format(subject_path))

        raw_frames = self.extract_video_frame(video_file_path)
        wave_gt = self.read_wave(mata_data_file_path)
        wave_gt = self.sample(wave_gt, len(raw_frames))
        # Align face
        # get directory name and file name
        face_path = os.path.join(self.des_path_root, session, 'faces') 
        # get filelists from face_path
        if not os.path.exists(face_path):
            os.makedirs(face_path)
        
        align_frames = self.align_face(raw_frames, device, self.detect_face_every_time)
        #save aligned frames
        for i in range(len(align_frames)):
            face_file = os.path.join(face_path, '%05d.jpg' % i)
            iio.imwrite(face_file, align_frames[i])

        n_vid = len(wave_gt) // self.frame_length
        for i in range(n_vid):
            start = i * self.frame_length
            end = (i + 1) * self.frame_length

            # Save file
            des_path = os.path.join(self.des_path_root, session)
            if not os.path.exists(des_path):
                os.makedirs(des_path)
            file_path = os.path.join(des_path, str(i))
            np.savez(file_path, frames=align_frames[start:end], hr=wave_gt[start:end], wave=wave_gt[start:end])
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames[start:end].shape, len(wave_gt[start:end])
                )
            )
        print('Done session: ', session)

    def read_wave(self, mata_data_file_path):
        # Extract ground truth HR
        wave_gt = []

        with open(mata_data_file_path, 'r') as csvfile:
            xmp = csv.reader(csvfile)
            for row in xmp:
                wave_gt.append(float(row[3]))

        wave_gt = self.diff_normalize_label(wave_gt)
        return wave_gt
    
    