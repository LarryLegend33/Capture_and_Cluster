import numpy as np
import cv2
import math
import pickle
from matplotlib import pyplot as pl
import os
from phinalIR_cluster_wik import Variables


class Fluorescence_Analyzer:

    def __init__(self):
        self.gutarea_xy = []
        self.gutarea_xz = []
        self.gutintensity_xy = []
        self.gutintensity_xz = []
        self.lowres_gut_xy = []
        self.lowres_gut_xz = []
        self.gut_values = []
        self.xyz = []
        
    def exporter(self, directory):
        with open(directory + 'fluordata.pkl', 'wb') as file:
            pickle.dump(self, file)
   
    def multiply_gut_values(self):
        gval_xy = [a*b for a, b in zip(
            self.gutintensity_xy, self.gutarea_xy)]
        gval_xz = [a*b for a, b in zip(
            self.gutintensity_xz, self.gutarea_xz)]
        self.gut_values = [np.nanmax(
            [xy, xz]) for xy, xz in zip(gval_xy, gval_xz)]

    def get_fluor_data(self, data_directory):
        top_fl = cv2.VideoCapture(data_directory + 'top_fl.AVI')
        side_fl = cv2.VideoCapture(data_directory + 'side_fl.AVI')
        irdata = pickle.load(open(data_directory + 'fishdata.pkl', 'rb'))
        fishcenter_x = irdata.low_res_x
         #see if this needs to be set back to just y or if z needs inverting too
        fishcenter_y = [1888-y for y in irdata.low_res_y]
        fishcenter_z = [1888-z for z in irdata.low_res_z]
        frame_types = np.load(data_directory + 'frametypes.npy')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        gut_video_top = cv2.VideoWriter(
          data_directory + 'gut_video_top.AVI',
          fourcc, 5, (1888, 1888), True)
        gut_video_side = cv2.VideoWriter(
          data_directory + 'gut_video_side.AVI',
          fourcc, 5, (1888, 1888), True)
        maskvid_top = cv2.VideoWriter(
          data_directory + 'maskvid_top.AVI',
          fourcc, 5, (1888, 1888), False)
        maskvid_side = cv2.VideoWriter(
          data_directory + 'maskvid_side.AVI',
          fourcc, 5, (1888, 1888), False)
        fr_counter = 0
        for frame in frame_types:
            if frame == 0:
                fr_counter += 1
            if frame == 1:
                print(fr_counter)
                ret, top_im = top_fl.read()
                ret2, side_im = side_fl.read()
                top_im_color = np.copy(top_im)
                side_im_color = np.copy(side_im)
                if not ret:
                    break
                top_im_bw = cv2.cvtColor(top_im, cv2.COLOR_BGR2GRAY)
                side_im_bw = cv2.cvtColor(side_im, cv2.COLOR_BGR2GRAY)
                mask_xy = np.zeros(top_im_bw.shape, np.uint8)
                mask_xz = np.zeros(side_im_bw.shape, np.uint8)
                center_x = fishcenter_x[fr_counter]
                center_y = fishcenter_y[fr_counter]
                center_z = fishcenter_z[fr_counter]
                self.xyz.append([center_x, center_y, center_z])
                if math.isnan(center_x):
                    self.lowres_gut_xy.append(np.nan)
                    self.lowres_gut_xz.append(np.nan)
                    self.gutarea_xy.append(float('NaN'))
                    self.gutintensity_xy.append(float('NaN'))
                    self.gutarea_xz.append(float('NaN'))
                    self.gutintensity_xz.append(float('NaN'))
                    gut_video_top.write(top_im_color)
                    gut_video_side.write(side_im_color)
                    continue
                mask_radius = 120
                cv2.circle(mask_xy,
                           (int(center_x), int(center_y)),
                           mask_radius, (1, 1, 1), -1)
                cv2.circle(mask_xz,
                           (int(center_x), int(center_z)),
                           mask_radius, (1, 1, 1), -1)
                gut_xy = cv2.multiply(top_im_bw, mask_xy).astype(np.uint8)
                gut_xz = cv2.multiply(side_im_bw, mask_xz).astype(np.uint8)
                maskvid_top.write(gut_xy)
                maskvid_side.write(gut_xz)
#                self.lowres_gut_xy.append(np.mean(gut_xy))
#                self.lowres_gut_xz.append(np.mean(gut_xz))
                self.lowres_gut_xy.append(np.mean(
                    np.clip(gut_xy,
                            np.percentile(gut_xy, 95), 255)))
                self.lowres_gut_xz.append(np.mean(
                    np.clip(gut_xz,
                            np.percentile(gut_xz, 95), 255)))
                gutcopy_xy = np.copy(gut_xy)
                gutcopy_xz = np.copy(gut_xz)
                ret, thresh_xy = cv2.threshold(
                  gutcopy_xy, 40, 255, cv2.THRESH_BINARY)
                rim, contours_xy, hierarchy = cv2.findContours(
                  thresh_xy,
                  cv2.RETR_EXTERNAL,
                  cv2.CHAIN_APPROX_NONE)
                ret, thresh_xz = cv2.threshold(
                  gutcopy_xz, 40, 255, cv2.THRESH_BINARY)
                rim, contours_xz, hierarchy = cv2.findContours(
                  thresh_xz,
                  cv2.RETR_EXTERNAL,
                  cv2.CHAIN_APPROX_NONE)

    # FILTER HERE AND MAKE A GUT CONTOUR
                filt_conts_xy = filter(lambda x: cv2.contourArea(x) > 0,
                                       contours_xy)
                filt_conts_xz = filter(lambda x: cv2.contourArea(x) > 0,
                                       contours_xz)
                byarea_xy = sorted(filt_conts_xy,
                                   key=cv2.contourArea, reverse=True)
                byarea_xz = sorted(filt_conts_xz,
                                   key=cv2.contourArea, reverse=True)

                if (not byarea_xy) or (not byarea_xz):
                    self.gutarea_xy.append(float('nan'))
                    self.gutarea_xz.append(float('nan'))
                    self.gutintensity_xy.append(float('nan'))
                    self.gutintensity_xz.append(float('nan'))
                    gut_video_top.write(top_im_color)
                    gut_video_side.write(side_im_color)
                    continue
                  
                gut_contour_xy = byarea_xy[0]
                gut_contour_xz = byarea_xz[0]
                self.gutarea_xy.append(cv2.contourArea(gut_contour_xy))
                self.gutarea_xz.append(cv2.contourArea(gut_contour_xz))
                gutmask_xy = np.zeros(top_im.shape, np.uint8)
                gutmask_xz = np.zeros(side_im.shape, np.uint8)
                cv2.drawContours(gutmask_xy, [gut_contour_xy], 0, 255, -1)
                cv2.drawContours(gutmask_xz, [gut_contour_xz], 0, 255, -1)
                gutmask_xy = cv2.cvtColor(gutmask_xy, cv2.COLOR_BGR2GRAY)
                gutmask_xz = cv2.cvtColor(gutmask_xz, cv2.COLOR_BGR2GRAY)
                gutmean_xy, a, b, c = cv2.mean(top_im_bw, mask=gutmask_xy)
                gutmean_xz, a, b, c = cv2.mean(side_im_bw, mask=gutmask_xz)
                self.gutintensity_xy.append(gutmean_xy)
                self.gutintensity_xz.append(gutmean_xz)
                cv2.drawContours(top_im_color,
                                 [gut_contour_xy], 0, (0, 0, 255), 1)
                cv2.drawContours(side_im_color,
                                 [gut_contour_xz], 0, (0, 0, 255), 1)
                gut_video_top.write(top_im_color)
                gut_video_side.write(side_im_color)
   
        gut_video_top.release()
        gut_video_side.release()
        maskvid_top.release()
        maskvid_side.release()
        top_fl.release()
        side_fl.release()

        
def wrap_fluor(dr):
    fl_analyze = Fluorescence_Analyzer()
    fl_analyze.get_fluor_data(dr)
    fl_analyze.multiply_gut_values()
    fl_analyze.exporter(dr)
    return fl_analyze

 
if __name__ == '__main__':
#    wrap_fluor(os.get_cwd())
    fl_obj = wrap_fluor('/Users/nightcrawler2/PreycapMaster/020419_1/')
