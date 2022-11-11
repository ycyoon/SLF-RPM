"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported:UBFC,PURE and COHFACE
"""
import datetime
import os
import time
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import imageio as iio
import cv2
import numpy as np
from Retinaface import Retinaface

class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))     
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape)) 
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_

class BaseLoader():
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    def __init__(self, data_path, des_path_root, face_size=128, frame_length=180, cvtbgr=True,print_debug=False, detect_face_every_time=False):
        """Initializes the data loader.
        """        
        self.data_path = data_path
        self.face_size = face_size
        self.des_path_root = des_path_root
        self.frame_length = frame_length
        self.detect_face_every_time = detect_face_every_time
        self.print_debug = print_debug
        self.cvtbgr = cvtbgr
        print('Detect every face', detect_face_every_time)
        
    def sample(self, a, len):
        """Samples a sequence into specific length."""
        return np.interp(
            np.linspace(
                1, a.shape[0], len), np.linspace(
                1, a.shape[0], a.shape[0]), a)

    def diff_normalize_label(self, label):
        """Difference frames and normalization labels"""
        diff_label = np.diff(label, axis=0)
        normalized_label = diff_label / np.std(diff_label)
        normalized_label[np.isnan(normalized_label)] = 0
        return normalized_label

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
            file_path = os.path.join(des_path, str(i))
            np.savez(file_path, frames=align_frames[start:end], wave=wave_gt[start:end])
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames[start:end].shape, len(wave_gt[start:end])
                )
            )
        print('Done session: ', session)

    def read_wave(self, session, length):
        """Reads wave file."""
        pass
    
    def read_raw_frames(self, session):
        """Reads raw frames from the given session."""
        pass

    def extract_video_frame(self, video_path, frame_path=None):
        """Extract frames from the given video

        Extract each frame from the given video file and store them into '.jpg' format. It
        extracts every frame of the video. If the given frame path exsits, it overwrites
        the contents if users choose that.

        Args:
                video_path (str): Required. The path of video file.

                frame_path (str): Required. The path to store extracted frames. If the path exists, it tries to
                                        remove it by asking the user.

        """

        frames = []
        count = 0
        fps = iio.v3.immeta(video_path, plugin="pyav")["fps"]
        
        video = iio.get_reader(video_path, "ffmpeg", fps=29.97)
        # get video width and height
        width = int(video.get_meta_data()["size"][0])
        height = int(video.get_meta_data()["size"][1])
        print("fps(org): ", fps, "width: ", width, "height: ", height)
        for idx, frame in enumerate(video):
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            if frame_path is None:                
                frames.append(frame)
            else:
                fname = "frame_{:0>4d}.png".format(idx)
                ofname = os.path.join(frame_path, fname)
                #ret = cv2.imwrite(ofname, frame)
                iio.imwrite(ofname, frame)

            count += 1

        return frames

    def align_face(self, frames, dmodel, detect_every_time=False):     
        new_bbox = None
        align_frames = np.empty((len(frames), self.face_size, self.face_size, 3), dtype=np.uint8)
            
        for i, img in enumerate(tqdm(frames, leave=False)):
            height,width,_=img.shape
            faces = dmodel(img) 

            if (new_bbox is None or detect_every_time) and len(faces) > 0:
                # select the largest face
                face = faces[0] # 첫번째 얼굴만 사용
                
                #if face[4]<0.9: # remove low confidence detection
                #    continue
                x1=face[0]
                y1=face[1]
                x2=face[2]
                y2=face[3]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                size = int(min([w, h])*1.2)
                cx = x1 + w//2
                cy = y1 + h//2
                x1 = cx - size//2
                x2 = x1 + size
                y1 = cy - size//2
                y2 = y1 + size

                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)
                new_bbox = list(map(int, [x1, x2, y1, y2]))
                new_bbox = BBox(new_bbox)

            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            cropped_face = cv2.resize(cropped, (self.face_size, self.face_size))

            align_frames[i] = cropped_face

        return align_frames

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
        sessions = os.listdir(self.data_path)
        sessions = [s for s in sessions if os.path.isdir(os.path.join(self.data_path, s))]
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
        

