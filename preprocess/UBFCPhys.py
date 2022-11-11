import json
import os
from preprocess.BaseLoader import BaseLoader
import cv2
import numpy as np

class UBFCPhysPreprocess(BaseLoader):
    
    def __init__(self, data_path, des_path_root, face_size=128, frame_length=180, debug=False, detect_face_every_time=False, level=1):
        """Initializes the data loader.
        """    
        super(UBFCPhysPreprocess, self).__init__(data_path, des_path_root, face_size, frame_length, False, debug, detect_face_every_time)
        self.level = level

    def read_wave(self, session, length):
        """Reads wave file."""
        bvp_file = os.path.join(self.data_path, session, 'bvp_%s_T%d.csv' % (session, self.level))
        with open(bvp_file) as file:
            wave_gt = [float(line.strip()) for line in file]
        wave_gt = np.array(wave_gt)
        print('sample from %d to %d' % ( len(wave_gt), length))
        wave_gt = self.sample(wave_gt, length)
        wave_gt = self.diff_normalize_label(wave_gt)
        return wave_gt
    
    def read_raw_frames(self, session):
        """Reads raw frames."""
        video_file = os.path.join(self.data_path, session, 'vid_%s_T%d.avi' % (session, self.level))
        return self.extract_video_frame(video_file)
    
    def extract_session(self, session, device):
        """Extracts frames from the given session."""
        print("Processing session: ", session)
        # Sessions

        raw_frames = self.read_raw_frames(session)
        wave_gt = self.read_wave(session, len(raw_frames))

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
            if self.cvtbgr:
                align_frames[i] = cv2.cvtColor(align_frames[i], cv2.COLOR_BGR2RGB)            
            iio.imwrite(face_file, align_frames[i])

        n_vid = len(wave_gt) // self.frame_length
        print("Number of videos: ", n_vid)
        for i in range(n_vid):
            start = i * self.frame_length
            end = (i + 1) * self.frame_length

            # Save file
            des_path = os.path.join(self.des_path_root, session)
            if not os.path.exists(des_path):
                os.makedirs(des_path)
            file_path = os.path.join(des_path, str(i)+'_level_'+str(self.level))
            np.savez(file_path, frames=align_frames[start:end], wave=wave_gt[start:end])
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames[start:end].shape, len(wave_gt[start:end])
                )
            )
        print('Done session: ', session)

