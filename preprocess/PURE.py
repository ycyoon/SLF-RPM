import json
import os
from preprocess.BaseLoader import BaseLoader
import cv2
import numpy as np

class PUREPreprocess(BaseLoader):
    
    def __init__(self, data_path, des_path_root, face_size=128, frame_length=180, debug=False, detect_face_every_time=False):
        """Initializes the data loader.
        """    
        super(PUREPreprocess, self).__init__(data_path, des_path_root, face_size, frame_length, debug, detect_face_every_time)
    

    def read_wave(self, session, length):
        """Reads wave file."""
        jsonpath = os.path.join(self.data_path, session + '.json')
        with open(jsonpath, 'r') as f:
            jsonread = json.load(f)
            hr_inf = jsonread["/FullPackage"]
        #hr_gt = []
        wave_gt = []
        
        for i in range(len(hr_inf)):
            #hr_gt.append(hr_inf[i]["Value"]["pulseRate"])
            wave_gt.append(hr_inf[i]["Value"]["waveform"])
            
        # PURE는 hr 데이터와 이미지 데이터 길이가 달라서 선형으로 맞춰줘야함
        #hr_gt = np.array(hr_gt)
        #hr_gt = self.sample(hr_gt, length)
        wave_gt = np.array(wave_gt)
        wave_gt = self.sample(wave_gt, length)
        #hr_gt = self.diff_normalize_label(hr_gt)
        wave_gt = self.diff_normalize_label(wave_gt)
        return wave_gt
    
    def read_raw_frames(self, session):
        """Reads raw frames."""
        raw_frames = []
        # File paths
        jsonpath = os.path.join(self.data_path, session + '.json')
        with open(jsonpath, 'r') as f:
            jsonread = json.load(f)
            image_inf = jsonread["/Image"]

        session_path = os.path.join(self.data_path, session)        
        if not os.path.isdir(session_path):
            return
        
        for i, ii in enumerate(image_inf):
            file = 'Image' + str(ii["Timestamp"]) + '.png'
            raw_frames.append(cv2.imread(os.path.join(session_path, file)))
            
        return raw_frames
    