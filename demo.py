# Modified from https://github.com/facebookresearch/detectron2

import os
import sys
import tempfile
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import face_alignment
from scipy.spatial import ConvexHull
import torch
from models import classifier
from utils.utils import poly2mask
from utils.augmentation import ToTensor
from utils.augmentation import Transformer, ToTensor, Normalise
from time import time
import face_detection


WINDOW_NAME = "SLF-RPM Demo"

class Predictor():
    def __init__(self, args) -> None:
        self.device = args.device
        self.transform = ToTensor()
        self._build_model(args)

    def _build_model(self, args):
        # Create SLF-RPM model
        print("\n=> Creating SLF-RPM Classifier Model: 3D ResNet-50")
        self.model = classifier.LinearClsResNet3D(model_depth=50, n_class=1)

        # Load from pretrained model
        if args.pretrained:
            print("=> Loaded pre-trained model '{}'".format(args.pretrained))
                        
            if os.path.isfile(args.pretrained):
                print(f"=> Loading model weights '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                state_dict = checkpoint["state_dict"]
                self.model.load_state_dict(state_dict)
                print(f"=> Loaded model weights '{args.pretrained}")
            else:
                print(f"=> Error: No checkpoint found at '{args.pretrained}'")
                print("Please check your inputs agian!")
                sys.exit()
            

        else:
            print(
                "=> Error: Pretrained model does not specify, demo program cannot run!"
            )
            sys.exit()

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(args.device)
        self.model.eval()
        # print(self.model)

    def __call__(self, clip):
        assert len(clip.shape) == 4, clip.shape
        clip_tensor = torch.Tensor(clip).unsqueeze(0).to(self.device)
        with torch.no_grad():    
            pred = self.model(clip_tensor)[0].item()
        return pred

class DemoProcessor():
    def __init__(self, args) -> None:
        self.clip_len = args.clip_len
        self.predictor = Predictor(args)
        # Detect face landmarks model
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            device=args.device,
            flip_input=False,
            face_detector="blazeface",
        )
        #self.detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)
        self.detector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
        self.ROI_forehead = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.ROI_cheek_left = [0, 1, 2, 3, 4, 5, 31, 41, 48]
        self.ROI_cheek_right = [16, 15, 14, 13, 12, 11, 35, 46, 54]
        self.ROI_mouth = [5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 48]

    def video_reader(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def landmarker(self, frame):
        landmark = self.fa.get_landmarks(frame)
        forehead_idx = np.zeros((1,1,2), dtype=np.uint8)
        left_cheek_idx = np.zeros((1,1,2), dtype=np.uint8)
        right_cheek_idx = np.zeros((1,1,2), dtype=np.uint8)
        mouth_idx = np.zeros((1,1,2), dtype=np.uint8)

        # Crop frame based on landmarks
        if landmark is None:
            # If landmarks cannot be detected, reture a black frame
            frame = np.zeros((64, 64), dtype=np.uint8)

        else:
            landmark = landmark[0]
            ROI_face = ConvexHull(landmark).vertices
            frame = poly2mask(
                landmark[ROI_face, 1], landmark[ROI_face, 0], frame, (64, 64)
            )
            forehead_idx = np.flip(landmark[self.ROI_forehead], -1).reshape((-1, 1, 2))
            left_cheek_idx = np.flip(landmark[self.ROI_cheek_left], -1).reshape((-1, 1, 2))
            right_cheek_idx = np.flip(landmark[self.ROI_cheek_right], -1).reshape((-1, 1, 2))
            mouth_idx = np.flip(landmark[self.ROI_mouth], -1).reshape((-1, 1, 2))

        return frame, forehead_idx, left_cheek_idx, right_cheek_idx, mouth_idx


    def run(self, video):
        augmentation = Transformer(
            [], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_loader = self.video_reader(video)  
        pred = -1
        landmark_time = 0
        predict_time = 0
        clip = np.empty((self.clip_len, 64, 64, 3), dtype=np.uint8)
        idx = 0
        tt = time()
        for i, frame in enumerate(frame_loader):
            if pred != -1:
                vis_frame = cv2.putText(frame, f"HR: {pred:.2f}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            else:
                vis_frame = cv2.putText(frame, "HR: 측정중", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i % 2 == 0: #stride 2                
                lt = time()
                landmark = self.fa.get_landmarks(frame)[0]
                
                # Crop frame based on landmarks
                if landmark is None:
                    # If landmarks cannot be detected, reture a black frame
                    face = np.zeros((64, 64, 3), dtype=np.uint8)
                else:
                    # TODO: Smooth landmarks
                    ROI_face = ConvexHull(landmark).vertices
                    face = poly2mask(landmark[ROI_face, 1], landmark[ROI_face, 0], frame, (64, 64))
                    #rois = util.cal_rois(frame, landmark[0], (64, 64))  # (n_roi, h, w, c)
                # frame = cv2.resize(frame, (64,64))
                landmark_time += time() - lt
                clip[idx] = face
                idx += 1

                if idx == self.clip_len:
                    frames = augmentation(clip)
                    pt = time()
                    pred = self.predictor(frames)
                    predict_time += time() - pt
                    print(f"HR: {pred:.2f}")   
                    idx = 0
            
            yield vis_frame 
        print('video fps', fps)
        print('Total Time', time()-tt)
        print('Landmark prediction time', landmark_time)
        print('HR Prediction time', predict_time)


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLF-RPM demo for builtin configs")

    parser.add_argument("--device", default="cuda:0", type=str, help="Device to run the program")
    parser.add_argument("-i", "--input", help="Path to video file.")

    parser.add_argument(
        "-o", "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("-l", "--clip-len", default=75, type=int) # stride 2 감안해서 150 /2
    parser.add_argument("-p", "--pretrained", default="logs/merged/test/best_test_model.pth.tar", type=str)

    args = parser.parse_args()

    print(f"=> Use device {args.device} for demo")
    processor = DemoProcessor(args)

    if args.input:
        video = cv2.VideoCapture(args.input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Total frames', num_frames)
        basename = os.path.basename(args.input)

        codec, file_ext = ("mp4v", ".mp4")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.input)
          
        for vis_frame in tqdm(processor.run(video), total=num_frames, position=0, leave=True):
            if args.output:
                output_file.write(vis_frame)
            else:
                continue
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
