import pickle
from matplotlib import pyplot as pl
from phinalIR import Variables
from phinalFL import Fluorescence_Analyzer
#from pvidFINAL import Para,ParaMaster
#from master import Experiment
from scipy import ndimage
import numpy as np
import seaborn as sb
import toolz

directory = raw_input("Enter Directory of Data: ")
fishdata = pickle.load(open(directory + '/fishdata.pkl','rb'))
for i in range(7):
    ta = [t[i] + i*100 for t in fishdata.tailangle]
    pl.plot(ta)
pl.show()


#z_list = []
#lt_inds = range(1,9,1)
#lt_directory = 'D:/Movies/LightTest'
#directory = raw_input("Enter Directory of Data: ")
#paramaster = pickle.load(open('paradata.pkl','rb'))
#for ind in lt_inds:
#    fishdata = pickle.load(open(lt_directory + str(ind) + '/fishdata.pkl','rb'))
#    z_list.append(fishdata.low_res_z)
#sb.tsplot(z_list)
#pl.xlabel('Time (s)')
#pl.ylabel('Height (pix: 0-1888)')
#pl.title('0-600: 255, 600-900: 245, 900-1200: 235, 1200-1500: overhead, 1500-1800: darkness')
#pl.show()

#myexp = pickle.load(open('master.pkl','rb'))
#fig = pl.figure(figsize = (10,10))
#counter = 0
#for j in range(8):
#    ax = fig.add_subplot(2,4,j+1)
#    pl.title('LightTest' + str(j+1))
#    ax.plot(ndimage.gaussian_filter(z_list[j],10)) 
#pl.show()
