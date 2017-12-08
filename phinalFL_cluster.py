import numpy as np
import cv2
from collections import deque
import toolz
import copy
import math
from itertools import repeat
import scipy.ndimage
from scipy.stats import mode
from matplotlib import pyplot as pl
from phinalIR_cluster import Variables
import pickle
import os

class Fluorescence_Analyzer: 

  def __init__(self):
    self.gutarea = []
    self.gutintensity = []
    self.gut_values = []
    self.gutmax_list = []

  def exporter(self,directory):
    with open(directory +'fluordata.pkl','wb') as file:
      pickle.dump(self,file)
   
  def multiply_gut_values(self):
    self.gut_values = [a*b for a,b in zip(self.gutintensity,self.gutarea)]

  def get_fluor_data(self, data_directory):
    top_fl = cv2.VideoCapture(data_directory + 'top_fl.AVI')
    frametimes = np.load(data_directory + 'frametimes_fl.npy')
    irdata = pickle.load(open(data_directory + 'fishdata.pkl','rb'))
    fishcenter_x = irdata.low_res_x
    fishcenter_y = [1888-y for y in irdata.low_res_y]  #see if this needs to be set back to just y.
    frame_types = np.load(data_directory + 'frametypes.npy')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fc = cv2.CAP_PROP_FRAME_COUNT
    curr_frame = cv2.CAP_PROP_POS_FRAMES
    framecount = int(top_fl.get(fc))
    gut_video = cv2.VideoWriter(data_directory + 'gut_video.AVI',fourcc, 5,(1888,1888),True)

    fr_counter = 0
    for frame in frame_types: #frame_types has all frames. have to
 
        if frame == 0:
            fr_counter += 1

        if frame == 1:
            print(fr_counter)
            ret,im = top_fl.read()
            im_color = np.copy(im)
            if not ret:
                break
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mask = np.zeros(im.shape, np.uint8)
            center_x = fishcenter_x[fr_counter]
            center_y = fishcenter_y[fr_counter]

            if math.isnan(center_x):
                self.gutarea.append(float('NaN'))
                self.gutintensity.append(float('NaN'))
                gut_video.write(im_color)
                continue
            circle = cv2.circle(mask, (int(center_x), int(center_y)),200, (1,1,1), -1) # TEST RADIUS
            gut = cv2.multiply(im, mask).astype(np.uint8)
            gutcopy = np.copy(gut)
            ret,thresh = cv2.threshold(gutcopy, 38, 255, cv2.THRESH_BINARY) 
            rim, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# FILTER HERE AND MAKE A GUT CONTOUR
            filt = filter(lambda x: cv2.contourArea(x) > 5,contours)
            byarea = sorted(filt, key = cv2.contourArea, reverse = True)
            if not byarea:
                self.gutarea.append(float('nan'))
                self.gutintensity.append(float('nan'))
                gut_video.write(im_color)
                continue
            gut_contour = byarea[0]
            self.gutarea.append(cv2.contourArea(gut_contour))
            gutmask = np.zeros(im.shape, np.uint8)
            cv2.drawContours(gutmask, [gut_contour], 0,255,-1)
            gutmean,a,b,c = cv2.mean(im, mask = gutmask)
            self.gutintensity.append(gutmean)
            gutmax = np.sum(im == 252) #252 is top of dynamic range of camera
            self.gutmax_list.append(gutmax)
            cv2.drawContours(im_color,[gut_contour],0,(0,0,255),1)
            gut_video.write(im_color)
   
    gut_video.release()
    top_fl.release()

def wrap_fluor(dr):
  fl_analyze = Fluorescence_Analyzer()
  fl_analyze.get_fluor_data(dr)
  fl_analyze.multiply_gut_values()
  fl_analyze.exporter(dr)

 
if __name__ == '__main__':
   wrap_fluor(os.get_cwd())
  
