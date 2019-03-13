from ipyparallel import Client
import numpy as np
from capclasses import camwrap
import time
#import pyboard

num_frames = 5000
c = Client()
core1 = c[0]
core2 = c[1]
cores = c[0:2]
res = core1.apply_async(camwrap, 0, num_frames)
res2 = core2.apply_async(camwrap, 1, num_frames)
res.get()
res2.get()


#IF EVER PORTED TO PYTHON3, CAN USE PYBOARD MODULE TO TALK TO BOARD INSTEAD OF LAUNCHING ON YOUR OWN
#time.sleep(5)

#myboard = pyboard.Pyboard('COM3')
#myboard.enter_raw_repl()
#myboard.exec('from rig_load import flash')
#myboard.exec('flash()')
#USE KEYBOARD TO GO HERE
#myboard.exit_raw_repl()
#myboard.close()
 



 



  
  


   

   

   
     
   

   
   
