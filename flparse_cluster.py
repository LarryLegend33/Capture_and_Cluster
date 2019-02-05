import pickle
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
from phinalIR_cluster_wik import wrap_ir
from phinalFL_cluster import wrap_fluor
import os

## THIS PROGRAM TAKES THE RAW CAM0 AND CAM1 MOVIES AND PARSES THEM INTO FL and HIGH CONTRAST VIDEO STREAMS. IT FIRST CREATES ARRAYS OF BACKGROUND IMAGES TO BE USED FOR GENERATING HIGH CONTRAST IMAGES AND EXPORTS THESE BACKGROUND ARRAYS FOR USE IN PHINAL_IR. ##


#this function just fixes the line that is out on the camera. 

def fix_blackline(im):
    cols_to_replace = [833, 1056, 1135, 1762, 1489]
    for c in cols_to_replace:
        col_replacement = [int(np.mean([a, b])) for a, b in zip(
           im[:, c-1], im[:, c+1])]
        im[:, c] = col_replacement
    return im


def validate_timing(times):
    delta_gen = toolz.itertoolz.sliding_window(2, times)
    deltas = [b-a for a, b in delta_gen]
    fig = pl.figure()
    ax = fig.add_subplot(121)
    ax.hist(deltas)
    ax2 = fig.add_subplot(122)
    ax2.plot(deltas)
    pl.show()
    

def get_frametypes(dict_file):

# provide fl_interval in seconds (ie one frame ever 10 sec),
# ir_freq for acquisition in hz. duration of epoch in minutes.
# for each, the total number of frames will be duration*60*ir_freq.

    def assign_frames(fluor, duration, ir_freq, fl_interval):
        temp_framelist = []
        #fl_count is directly related to ir_freq. 
        fl_count = int(ir_freq*fl_interval)
        numframes = int(duration*60*ir_freq)
        if fluor:
            count = 0
            for framecount in range(numframes):
                if count % fl_count == 0:
                    temp_framelist.append(1)
                else:
                    temp_framelist.append(0)
                count += 1
        else:
            temp_framelist = np.zeros(numframes).tolist()
        return temp_framelist, numframes, ir_freq
#0 for ir, 1 for fl

    frametypes = []
    frames_in_epoch = []
    ir_freq_in_epoch = []
    exp_dict = eval(dict_file.readline())
    for epoch, entry in enumerate(exp_dict):
        types, total_frames, ir_freq = assign_frames(
           *exp_dict['epoch' + str(epoch+1)])
        frametypes += types
        frames_in_epoch.append(total_frames)
        ir_freq_in_epoch.append(ir_freq)
    return frametypes, frames_in_epoch, ir_freq_in_epoch


def br_boundaries(counts, frequencies):

#take one frame every two seconds for the mode. modes should be calculated every 30 seconds. 
    br_interval = 2
    mode_interval = 30
    cumulative_counts = np.cumsum(counts)
    frames_for_br = []
    mode_frames = []
    for epoch_id, framecount in enumerate(cumulative_counts):
        print framecount
        if epoch_id == 0:
            temp_frames_br = range(0, framecount,
                                   int(frequencies[epoch_id])*br_interval)
     #every 10 for temp_frames, every 150 for mode if 5 hz. every 1860 , 124 for high freq.  
            mode_indices =  range(0, framecount,
                                  int(frequencies[epoch_id])*mode_interval)
        else:
            temp_frames_br = range(cumulative_counts[epoch_id-1], framecount,
                                   int(frequencies[epoch_id])*br_interval)
            mode_indices = range(cumulative_counts[epoch_id-1], framecount,
                                 int(frequencies[epoch_id])*mode_interval)

        frames_for_br += temp_frames_br
        mode_frames += mode_indices
# modeframes shouldn't have 0 as an index.         
    mode_frames = mode_frames[1:]
    return frames_for_br, mode_frames

#This function takes the first 4 pixels of the top row of each image and extracts the timestamp in ms. 

def frametime(img):
    # timestamp info is in first row, first 4 columns.
    bytes = img[0][0:4].tolist()
    # binary pattern for each integer in bytes occurs after prefix '0b'
    bits_sep = [bin(x)[2:].zfill(8) for x in bytes]
    # this joins all the binary bits into a 32 bit word.
    bits = ''.join(bits_sep)
    # int function with ,2 converts a bit string back into a base 10 integer.
    seconds = int(bits[0:7], 2)
    cycle = int(bits[7:20], 2)
    # last 4 bits are supposedly inaccurate according to docs. replace w zeros
    offset = int(bits[20:28]+'0000', 2)
    # returns time in ms
    t = seconds * 1000.0 + cycle * .125 + ((offset / 3072.0)*.125)
    return t


#this function takes a deque of numpy arrays and returns the mode of each pixel in a new numpy array. it uses the scipy.stats mode function to take the mode of each pixel projected through the entire deque. 

def calc_mode(deq, nump_arr):
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j, :] = mode(np.array([x[j, :] for x in deq]))[0]
    return nump_arr

 
def calc_median_orig(deq, nump_arr):
    for j in range(nump_arr.shape[0]):
        for k in range(nump_arr.shape[1]):
            nump_arr[j, k] = np.median([x[j, k] for x in deq])
    return nump_arr

 
def calc_median(deq, nump_arr):
    # so k are the values, j are the indicies.
    for j, k in enumerate(nump_arr[:, 0]):
        nump_arr[j] = np.median([x[j] for x in deq], axis=0)
    return nump_arr
    
# this function takes a series of delta times and returns the true time from the beginning of the experiment. frame times are modulo 128000 ms, so diffs are corrected for this in delta calculation. 


def timegenerator(times):
    delta_gen = toolz.itertoolz.sliding_window(2, times)
    delta = [b-a if b > a else 128000-a+b for a, b in delta_gen]
    ts = np.cumsum(delta)
    return ts


#This function makes all paramecia and fish 255 white. Tested 3 different ways to improve contrast of paramecia. gaussian blur with 15,15 parameter was the most effective and fastest, but kept commented boxfiltering and scipy gauss for history and in case of future failure of GaussianBlur.

def imcont(image, rangemax, hardthresh):
    #  gf_im = cv2.GaussianBlur(image,(3,3), 0)
    # this now maxes out the image at max without stretching dynamic range (i.e. 26 in image is 26 in th if max > 26. everything over max is set to 255). 
    r, th = cv2.threshold(image,
                          rangemax,
                          255, cv2.THRESH_TRUNC)
    adj = ((255 / float(rangemax)) * th).astype(np.uint8)
    # now stretch dynamic range so that your max is 255, and distance from max is linearly represented. this stretches the range so your choice of threshold is less important.  
    #  bf = cv2.boxFilter(adj, -1, (5,5)) #filter twice to get rid of noise that could be picked as a para. 
    #bf = scipy.ndimage.filters.gaussian_filter(adj, 5)
    bf = cv2.GaussianBlur(adj, (5, 5), 0)
    r, th2 = cv2.threshold(bf, hardthresh, 255, cv2.THRESH_BINARY)
    # now set a hard threshold. your choice here is made easier by your spread of the dynamic range
    th2_c = cv2.cvtColor(th2.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return th2_c


#this function just puts a black line over the edges of the tank. don't need this if background is good.
 
def edgefix(image, x, y, isY):
    wndw = 15
    # has to be bounds of colindex
    rowreplace = np.zeros(x[1]-x[0])
    # has to be bounds of rowindex
    colreplace = np.zeros(y[1]-y[0])
    rowrange = range(y[0]-wndw, y[0]+wndw) + range(y[1]-wndw, y[1]+wndw)
    colrange = range(x[0]-wndw, x[0]+wndw) + range(x[1]-wndw, x[1]+wndw)
    if isY:
        for i in rowrange:
            image[i, x[0]:x[1]] = rowreplace 
        for j in colrange:
            image[y[0]:y[1], j] = colreplace
    return image

 
def run_flparse(data_directory):
    # MAINLINE. FIRST CREATES BACKGROUND ARRAYS, THEN HIGH CONTRAST VIDEOS.
    cam0id = [file_id for file_id in os.listdir(
       data_directory) if file_id[-8:] == 'cam0.AVI'][0]
    cam1id = [file_id for file_id in os.listdir(
       data_directory) if file_id[-8:] == 'cam1.AVI'][0]
    top = cv2.VideoCapture(data_directory + cam0id)
    side = cv2.VideoCapture(data_directory + cam1id)
    dict_file = open(data_directory + 'experiment.txt')
    frame_types, frame_counts, frequencies = get_frametypes(dict_file)
    print(frame_counts)
    ir_br_frames_to_take, mode_index = br_boundaries(frame_counts, frequencies)
    # play with these params if contrast is not good. but works for now. 
    contrast_params_top = [30, 60]
    contrast_params_side = [30, 50]
    # fourcc2 = cv2.cv.CV_FOURCC('M','J','P','G')
    # fourcc2 = 0
    fourcc2 = cv2.VideoWriter_fourcc(*'MJPG')
    top_contrasted = cv2.VideoWriter(data_directory + 'top_contrasted.AVI',
                                     fourcc2, 62, (1888, 1888), True)
    side_contrasted = cv2.VideoWriter(data_directory + 'side_contrasted.AVI',
                                      fourcc2, 62, (1888, 1888), True)
    top_fl = cv2.VideoWriter(data_directory + 'top_fl.AVI',
                             0, 5, (1888, 1888), True)
    side_fl = cv2.VideoWriter(data_directory + 'side_fl.AVI',
                              0, 5, (1888, 1888), True)
    frametimes_all = []
    curr_frame = cv2.CAP_PROP_POS_FRAMES
    fc = cv2.CAP_PROP_FRAME_COUNT
    framecount = top.get(fc)
    top_ir_temp = []
    side_ir_temp = []
    top_ir_backgrounds = []
    side_ir_backgrounds = []
    startframe = 0
    endframe = int(framecount)
#        endframe = 3000
    createbackground = True
    getfishdata = True

    # FIRST MAKE BACKGROUND ARRAYS. WILL NEED THEM FOR PHINAL_IR AND HIGH CONTRAST STREAM GENERATION. 

    if createbackground:
        # make sure mode is always a multiple of frames to take.
        for i in ir_br_frames_to_take:
            if i > endframe:
                break
            top.set(curr_frame, i)
            side.set(curr_frame, i)
            ret, im = top.read()
            ret, im2 = side.read()
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            img = fix_blackline(img)
            img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            if frame_types[i] == 0:
                print('add_to_ir_temp')
                print(i)
                top_ir_temp.append(img)
                side_ir_temp.append(img2)

          # take a mode if i is in the mode_index list (i.e. times where you should take a mode)

            if i in mode_index:
                top_ir_backgrounds.append(calc_median(top_ir_temp,
                                                      np.zeros([1888, 1888])))
                side_ir_backgrounds.append(calc_median(side_ir_temp,
                                                       np.zeros([1888, 1888])))
                print('after modes')
                print(i)
                top_ir_temp = []
                side_ir_temp = []

        #accounts for leftovers at the end (i.e. if there are still arrays in temp that haven't been emptied by the modulo emptying at modefreq*ir_br_freq  

        if top_ir_temp:
            top_ir_backgrounds.append(calc_median(top_ir_temp,
                                                  np.zeros([1888, 1888])))
            side_ir_backgrounds.append(calc_median(side_ir_temp,
                                                   np.zeros([1888, 1888])))
            top_ir_temp = []
            side_ir_temp = []

        np.save(data_directory + 'backgrounds_top.npy', top_ir_backgrounds)
        np.save(data_directory + 'backgrounds_side.npy', side_ir_backgrounds)

    else:
        top_ir_backgrounds = np.load(data_directory + 'backgrounds_top.npy')
        side_ir_backgrounds = np.load(data_directory + 'backgrounds_side.npy')


    #now go through each frame and subtract the proper background to get high contrast fish and para images.

    brcounter = 0 
    for j in range(startframe, endframe, 1):
        if (j in mode_index) or j == 0:
            ir_t_br = top_ir_backgrounds[brcounter].astype(np.uint8)
            ir_s_br = side_ir_backgrounds[brcounter].astype(np.uint8)
            brmean_top = np.mean(ir_t_br)
            brmean_side = np.mean(ir_s_br)
            brcounter += 1

        top.set(curr_frame, j)
        side.set(curr_frame, j)
        ret, im = top.read()
        ret, im2 = side.read()
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        if frame_types[j] == 0:
            if j % 20 == 0:
              print j
            img = fix_blackline(img)
            top_avg = np.mean(img)
            side_avg = np.mean(img2)
            # normalize means of br and image
            img_adj = (img * (brmean_top / top_avg)).astype(np.uint8)
            img2_adj = (img2 * (brmean_side / side_avg)).astype(np.uint8)
            frametimes_all.append(frametime(img))
            top_brsub = cv2.absdiff(img_adj, ir_t_br)
            side_brsub = cv2.absdiff(img2_adj, ir_s_br)
            img_contrasted = imcont(top_brsub,
                                    contrast_params_top[0],
                                    contrast_params_top[1])
            img2_contrasted = imcont(side_brsub,
                                     contrast_params_side[0],
                                     contrast_params_side[1])
            top_contrasted.write(img_contrasted)
            side_contrasted.write(img2_contrasted)

        else:
            frametimes_all.append(frametime(img))
            print('FLUORESCENT FRAME')
            top_fl.write(im)
            side_fl.write(im2)



    #frametimes are real time stamps of each image. timegenerator tells you how much time is between the current frame and the one in front of it. 
    #first time in frametimes_fl is t0 for the experiment. arrange times_ir and times_fl accordingly. (i.e. times_fl has to have a zero first, times_ir must have the time between frametimes_ir[0] and frametimes_fl[0] added to all indicies. 
    times_all = timegenerator(frametimes_all)
    times_all = np.insert(times_all, 0, 0)
    times_ir = [fr_time for fr_time,
                fr_type in zip(times_all, frame_types) if fr_type == 0]
    times_fl = [fr_time for fr_time,
                fr_type in zip(times_all, frame_types) if fr_type == 1]
    np.save(data_directory + 'frametimes_ir.npy', times_ir)
    np.save(data_directory + 'frametimes_fl.npy', times_fl)
    np.save(data_directory + 'frametimes_all.npy', times_all)
    np.save(data_directory + 'mode_indices.npy', mode_index)
    np.save(data_directory + 'frametypes.npy', frame_types)
    np.save(data_directory + 'framecounts.npy', frame_counts)
    np.save(data_directory + 'ir_freqs.npy', frequencies)
    top.release()
    side.release()
    top_contrasted.release()
    side_contrasted.release()
    top_fl.release()
    side_fl.release()
    if getfishdata:
        wrap_ir(data_directory)
        wrap_fluor(data_directory)
#            validate_timing(times_all)



        # make a linspace for each epoch. it'll be a 0:numframesinepoch by ir frequency. it'll take an ir image at every vale in the linspace. it'll take a mode every time there have been freq*30 entries. every time it hits a freq/2 entry, put that frame value in a list so that the background subtraction during contour finding knows at which frame to switch backgrounds to the next background. 

if __name__ == "__main__":
    run_flparse(os.getcwd() + '/')
