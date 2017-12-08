import cv2
import flycapture2 as fc2
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from pyfirmata import Arduino, util
import numpy as np
import time
from skimage.measure import block_reduce
from collections import deque
from tempfile import TemporaryFile
from random import randint
from numpy import random

app = QtGui.QApplication([])

                          
class Widget:

 def __init__(self):

  self.sidepic = pg.ImageItem()
  self.sidepic.rotate(270)
  self.toppic = pg.ImageItem()
  self.toppic.rotate(270)
  self.w = QtGui.QWidget()
  self.layout = QtGui.QGridLayout()
  self.w.setLayout(self.layout)
  self.interface = pg.GraphicsLayoutWidget()
  self.interface.setWindowTitle('Fish Viewer')
  self.topview = self.interface.addViewBox()
  self.sideview = self.interface.addViewBox()
  self.stop_btn = QtGui.QPushButton('Stop')
  self.stop_btn.clicked.connect(self.stop)
  self.topview.addItem(self.toppic)
  self.sideview.addItem(self.sidepic)
  self.topview.setAspectLocked(True)
  self.sideview.setAspectLocked(True)   
  self.layout.addWidget(self.stop_btn, 1,1)
  self.layout.addWidget(self.interface, 2,1)
  self.w.show()
  self.typeframe = []
  self.stopper = False

 def widget_update(self, i_side, i_top):
  
  self.sidepic.setImage(i_side)
  self.toppic.setImage(i_top) 

 def stop(self):

  self.stopper = True
  
   
class Camera:

 def __init__(self, camnum):

  fourcc = 0
  fpsec = 70
  resolution = (1504,1500)
  self.cam = fc2.Context()
  self.cam.connect(*self.cam.get_camera_from_index(camnum))
  self.serial_num = self.cam.get_camera_info()
  self.im = fc2.Image()
  self.video = cv2.VideoWriter('cam'+str(camnum)+'.AVI', fourcc, fpsec, resolution, False)
  #self.video = cv2.VideoWriter('cam1.AVI', fourcc, fpsec, resolution, False)
  self.cam.start_capture()
  self.cycles = []
  self.camID = camnum
  

 def takepic(self):

  self.cam.retrieve_buffer(self.im)  
  frame = np.array(self.im) #if you want frametimes put in during acquisition, use these lines
  #bytes = frame[0][0:4]
  #bits_sep = [bin(x)[2:].zfill(8) for x in bytes]
  #bits = ''.join(bits_sep)
  #cycle = int(bits[7:20],2)
  #self.cycles.append(cycle)
  self.video.write(frame)


 def takeonepic(self):

  self.cam.retrieve_buffer(self.im)  
  frame = np.array(self.im)
  cv2.imwrite('cam'+str(self.camID)+'.tif',frame)
  
  
 def disconnect(self):
                        
  self.video.release()
  self.cam.stop_capture()
  self.cam.disconnect()
  cv2.destroyAllWindows() 


def camwrap(cam,numframes):
  
  import cv2
  import flycapture2 as fc
  import numpy as np
  from capclasses import Camera
  camera = Camera(cam)

  for i in range(numframes):
      
   camera.takepic()

  camera.disconnect()

  
  



if __name__ == '__main__':
     
  main()
 
