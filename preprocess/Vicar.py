import csv
import json
import os
from preprocess.BaseLoader import BaseLoader
import cv2
import numpy as np
from torch.multiprocessing import Pool, set_start_method
from Retinaface import Retinaface
import time, datetime
import imageio as iio

class VicarPreprocess(BaseLoader):
    
    def __init__(self, data_path, des_path_root, face_size=128, frame_length=180, debug=False, detect_face_every_time=False):
        """Initializes the data loader.
        """    
        super(VicarPreprocess, self).__init__(data_path, des_path_root, face_size, frame_length, False, debug, detect_face_every_time)

    def read_wave(self, session, length):
        """Reads wave file."""
        bvp_path = os.path.join(self.data_path, 'GroundTruth/PPG/Raw')
        #01-base.mp4 -> 01-base PPG.csv
        bvp_file = session.replace('.mp4', ' PPG.csv')
        bvp_file = os.path.join(bvp_path, bvp_file)
        #open csv file
        wave_gt = []
        with open(bvp_file) as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                wave_gt.append(float(row[1]))
        wave_gt = np.array(wave_gt)
        print('sample from %d to %d' % ( len(wave_gt), length))
        wave_gt = self.sample(wave_gt, length)
        wave_gt = self.diff_normalize_label(wave_gt)
        return wave_gt

    def read_raw_frames(self, session):
        """Reads raw frames."""
        video_file = os.path.join(self.data_path, 'Videos', session)
        return self.extract_video_frame(video_file)

    def do_preprocess_multi(self):
        """Preprocesses the data.

        This function is used to preprocess the data, including reading files, resizing each frame,
        chunking, and video-signal synchronization.

        Args:
            dmodel: face detection model
            detect_every_time: if True, detect face every frame
        """
        try:
            set_start_method('spawn', force=True)
            print('Using spawn')
        except RuntimeError:
            pass


        if not os.path.exists(self.des_path_root):
            os.makedirs(self.des_path_root)

        start = time.time()
        sessions = os.listdir(os.path.join(self.data_path, 'Videos'))
        sessions.sort()
        

        # multi process 
        dmodels = []
        pool = Pool(processes=8)
        for i in range(8):
            device = 'cuda:{}'.format(i % 8)
            dmodel= Retinaface.Retinaface(device=device)  
            dmodels.append(dmodel)
        for i, session in enumerate(sessions):
            device = i % 8
            res = pool.apply_async(self.extract_session,
                args=(session, dmodels[device]),
            )
            if self.print_debug:
               print(res.get())

        pool.close()
        pool.join()

        duration = str(datetime.timedelta(seconds=time.time() - start))
        print("It takes {} time for extracting dataset".format(duration))
