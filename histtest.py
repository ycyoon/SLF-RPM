import cv2
import numpy as np
from matplotlib import pyplot as plt
  
# reading the input image
img = cv2.imread('/home/yoon/data/face/VGG-Face2/face_extract/train/n000014/0001_01.jpg')
img2 = cv2.imread('/home/yoon/data/face/VGG-Face2/face_extract/train/n000014/0001_02.jpg')
# define colors to plot the histograms
colors = ('b','g','r')
  
# compute and plot the image histograms
histcat = np.zeros(256*3)
for i,color in enumerate(colors):
    hist = cv2.calcHist([img, img2],[i],None,[256],[0,256])
    print(hist.flatten(), hist.flatten().shape)
    cv2.normalize(hist, hist)

    histcat[i*256:(i+1)*256] = hist.flatten()

print(histcat, histcat.shape)