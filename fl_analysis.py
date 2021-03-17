import numpy as np
import pickle
import cv2
import math
from matplotlib import pyplot as pl
import os
from phinalIR_cluster_wik import Variables
from phinalFL_cluster import Fluorescence_Analyzer
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans, convolve
import seaborn as sb
import pandas as pd
import itertools


# TODO
# for the top video in fluor, have to threshold the mask video. it's catching the mask as a contour.

# also still seeing fluorescence in the IR record. shouldn't see a white frame.
# get the exact phinalIR and FL files on the cluster working on your computer.
# make sure you get the exact results you need, even wrt eye convergence. 

def tsplot(data, ax, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)

    
def ts_plot(list_of_lists, ax):
    fig, ax = pl.subplots(1, 1)
    index_list = list(
        itertools.chain.from_iterable(
            [range(len(arr)) for arr in list_of_lists]))
    id_list = [(ind*np.ones(len(arr))).tolist() for ind, arr in enumerate(list_of_lists)]
    ids_concatenated = list(itertools.chain.from_iterable(id_list))
    #this works if passed np.arrays instead of lists
    value_list = list(itertools.chain.from_iterable(list_of_lists))
    df_dict = {'x': index_list, 
               'y': value_list}
    df = pd.DataFrame(df_dict)
    sb.lineplot(data=df, x='x', y='y', ax=ax)
    pl.show()
    return df
    

def fluor_wrapper(drct_lists_by_condition):

    def fill_condition_list(drct_list):
        fl_gutvals = []
        fl_gutintensity = []
        fl_gutarea = []
        fl_lowres = []
        gkern = Gaussian1DKernel(1)
        fluor_directory = '/Users/nightcrawler2/FluorData/'
        for drct in drct_list:
            fish_fl = pickle.load(open(fluor_directory + drct +
                                       '/fluordata.pkl', 'rb'))
            fl_gutvals.append(fish_fl.gut_values)
            g_int = [np.mean([g1, g2]) for g1, g2 in zip(
                fish_fl.gutintensity_xy,
                fish_fl.gutintensity_xz)]
            fl_gutintensity.append(g_int)
            g_area = [np.mean([g1, g2]) for g1, g2 in zip(
                fish_fl.gutarea_xy,
                fish_fl.gutarea_xz)]
            fl_gutarea.append(g_area)
            g_lowres = [np.mean([g1, g2]) for g1, g2 in zip(
                fish_fl.lowres_gut_xy,
                fish_fl.lowres_gut_xz)]
            fl_lowres.append(g_lowres)
        filt_gutvals = [convolve(flgv, gkern,
                                 preserve_nan=False) for flgv in fl_gutvals]
        filt_gutintensity = [convolve(fl_int, gkern,
                                      preserve_nan=False)
                             for fl_int in fl_gutintensity]
        filt_gutarea = [convolve(fla, gkern,
                                 preserve_nan=False) for fla in fl_gutarea]
        filt_lowres = [convolve(lr, gkern,
                                preserve_nan=False) for lr in fl_lowres]
        return [filt_gutvals, filt_gutintensity,
                filt_gutarea, filt_lowres]

    
    fl_by_condition_dictlist = []
    # fl_condition_list will contain lists with 4 elements each
    # each of the four elements is a fl readout (gutval, avg, lowres, size)
    # each element contains x lists of that value, where x is the
    # number of fish (i.e. directories) for that condition
    # you want an axis for each of these for a tsplot
    for drct_list in drct_lists_by_condition:
        gutvals, gutintensity, gutarea, lowres_gut = fill_condition_list(drct_list)
        fl_by_condition_dictlist.append({"Gut Values": gutvals,
                                         "Gut Intensity": gutintensity,
                                         "Gut Area": gutarea,
                                         "Lowres GutVals": lowres_gut})

        
    fig, axes = pl.subplots(2, 2, figsize=(10, 6))

    # here want to do a new ts plot on each axes for each
    # condition. tsplot wants a list of arrays of the same length of the same condition. 
    
    for fl_instance in fl_by_condition_dictlist:
        ts_plot(np.array(fl_instance["Gut Values"]), axes[0, 0])
        ts_plot(np.array(fl_instance["Gut Intensity"]), axes[1, 0])
        ts_plot(np.array(fl_instance["Gut Area"]), axes[0, 1])
        ts_plot(np.array(fl_instance["Lowres GutVals"]), axes[1, 1])
    pl.show()
    return fl_by_condition_dictlist

fl_conditions = fluor_wrapper([['1_2', '2_2'], ['1_2', '1_2']])
