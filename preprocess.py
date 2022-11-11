import argparse
import sys
from preprocess.PURE import PUREPreprocess
from preprocess.UBFC2 import UBFC2Preprocess
from preprocess.UBFC1 import UBFC1Preprocess
from preprocess.Cohface import CohfacePreprocess
from preprocess.UBFCPhys import UBFCPhysPreprocess
from preprocess.Vicar import VicarPreprocess


parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='PFLD', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')
parser.add_argument('-d', '--datatype', default='mahnob', type=str,)
parser.add_argument('--debug', action='store_true', help='debug multithreading')
parser.add_argument('--level', default=1, type=int, help='ubfc level')
args = parser.parse_args()

if __name__=="__main__":
    if args.datatype == 'pure':
        p = PUREPreprocess('/home/yoon/data/PPG/PURE', '/home/yoon/data/PPG/preprocess/pure', 128, 180, args.debug, False)
    elif args.datatype == 'ubfc2':
        p = UBFC2Preprocess('/home/yoon/data/PPG/UBFC/UBFC_DATASET/DATASET_2/', '/home/yoon/data/PPG/preprocess/ubfc2', 128, 180, args.debug,  False)
    elif args.datatype == 'ubfc1':
        p = UBFC1Preprocess('/home/yoon/data/PPG/UBFC/UBFC_DATASET/DATASET_1/', '/home/yoon/data/PPG/preprocess/ubfc1', 128, 180,  args.debug, False)
    elif args.datatype == 'cohface':
        p = CohfacePreprocess('/home/yoon/data/PPG/cohface/', '/home/yoon/data/PPG/preprocess/cohface', 128, 180,  args.debug, False)
    elif args.datatype == 'ubfc-phys':
        p = UBFCPhysPreprocess('/home/yoon/data/PPG/UBFC-Phys/', '/home/yoon/data/PPG/preprocess/ubfc-phys', 128, 180,  args.debug, True, 1)
    elif args.datatype == 'vicar':
        p = VicarPreprocess('/home/yoon/data/PPG/Vicar/', '/home/yoon/data/PPG/preprocess/vicar', 128, 180,  args.debug, True)
    
    else:
        print('no such dataset')
        sys.exit(1)
    p.do_preprocess_multi()
