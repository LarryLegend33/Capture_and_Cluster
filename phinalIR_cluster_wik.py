import cv2
import numpy as np
import pickle
import toolz
import math
import scipy.ndimage
from sklearn.metrics.pairwise import pairwise_distances
import scipy.signal
from sklearn.neighbors.kde import KernelDensity
from matplotlib import pyplot as pl
import os


class Variables:
    def __init__(self):
        self.pitch = []
        self.phileft = []
        self.phiright = []
        self.leftxy = []
        self.rightxy = []
        self.tailangle = []
        self.headingangle = []
        self.swimbcoords_top = []
        self.swimbcoords_side = []
        self.x = []
        self.y = []
        self.z = []
        self.x2 = []
        self.vectV = []
        self.frametimes = []
        self.bouttimes = []
        self.hunt_inds = []
        self.abort_inds = []
        self.low_res_x = []
        self.low_res_y = []
        self.low_res_z = []

    # Simple function that replaces NaNs in data with previous non-nan value. 

    def infer(self, array):
        # this just makes a run of all zeros if the data is missing at the beginning. 
        if math.isnan(array[0]):
            array[0] = 0
        for k in range(len(array)):
            if not math.isnan(array[k]):
                pass
            elif math.isnan(array[k]):
                array[k] = array[k-1]
        return array

   # NaN Replacement then Filtering of Data using a specified Gaussian window. 

    def data_filter(self, array, window):
        inferred_array = self.infer(array)
        filt_array = scipy.ndimage.filters.gaussian_filter(
         inferred_array, window)
        return filt_array

   # This funciton is for plotting the Z-trajectories of fish after the initiation of a bout. 

    def plotinterbout_z(self):
        bout_windows = toolz.itertoolz.sliding_window(2, self.bouttimes)
        z_endposition = []
        for start, end in bout_windows:
            zplot = np.array(self.z[start:end]) - self.z[start]
            z_endposition.append(zplot[-1])
            pl.plot(zplot)
        pl.show()
        return z_endposition

   #This function plots the Z-trajectory of the fish between the initiation of a hunt and the abort of the hunt. 

    def hunt_z(self):
        abort_inds = self.abort_inds if self.abort_inds[
         0] < self.hunt_inds[0] else self.abort_inds[1:]
        hunt_windows = zip(self.hunt_inds, self.abort_inds)
        z_endposition_hunt = []
        z_endposition_abort = []
        pathduration = []
        for start, end in hunt_windows:
            zplot = np.array(self.z[start:end]) - self.z[start]
            z_endposition_hunt.append(zplot[-1])
            pathduration.append(len(zplot))
            pl.plot(zplot)
        pl.show()
        avgpathduration = int(np.median(pathduration))
        for abort_start in abort_inds:
            if abort_start+avgpathduration < len(self.z):
                zplot_abort = np.array(
                 self.z[
                  abort_start:
                  abort_start+avgpathduration]) - self.z[abort_start]
                z_endposition_abort.append(zplot_abort[-1])
        return z_endposition_hunt, z_endposition_abort

   #Important bout detection function. This function first sums up tail angles from each segement. It next calculates the running variance over a length 5 window of the tail angle, then postfilters that variance. Calculates local maxima of the tailvariance and 3D velocity, filtered by comparing values of local maxima to the average minimum variance and velocity. If tailvariance maxima co-localize with vector velocity maxima, bouts are called. 

    def findbouts(self):
        startbouts = []
        tailanglesum = [np.sum(tailangles) for tailangles in self.tailangle]
        tailanglesum = self.infer(tailanglesum)
        vectV = self.data_filter(self.vectV, 5)
        tailangle_var = [np.var(win) for win
                         in toolz.itertoolz.sliding_window(
                          5, tailanglesum)]
        tailangle_var = self.data_filter(tailangle_var, 5)
        # this calculates the average minimum value of the tailangle trace over time.
        tailangle_rms = np.median(
         [tailangle_var[ind] for ind in scipy.signal.argrelmin(
          np.array(tailangle_var), order=10)])
        print('rms tail')
        print(tailangle_rms)
        # local max calculation where previous 10 values have to be smaller,
        # next 10 have to be smaller.
        # data type is a tuple so you have to index it.
        tailangle_localmax, = scipy.signal.argrelmax(
         np.array(tailangle_var), order=10)
        tailangle_localmax_rms_thresh = [ind for ind in
                                         tailangle_localmax
                                         if tailangle_var[ind]
                                         > 5*tailangle_rms]
        velocity_localmax, = scipy.signal.argrelmax(
         np.array(vectV), order=10)
        velocity_localmin, = scipy.signal.argrelmin(
         np.array(vectV), order=10)
        velocity_rms = np.median([vectV[low] for low in velocity_localmin])
        # this calculates the average minimum value of the velocity trace over time. 
        velocity_localmax_rms_thresh = [vel_ind for vel_ind in
                                        velocity_localmax if vectV[vel_ind] > 2*velocity_rms]
        # Assigns index as a bout if the peak velocity occurs
        # inside a 15 frame window of the peak tailangle (5 before, 10 after).
        for tailmax in tailangle_localmax_rms_thresh:
            comprange = range(tailmax-5, tailmax+10)
            for velmax in velocity_localmax_rms_thresh:
                if velmax in comprange:
                    startbouts.append(tailmax)
                    break #stops from having two local velocity max in window. 
        fig = pl.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax.plot(startbouts, np.zeros(len(startbouts)),
                marker='.', ms=20, color='r')
        ax.plot(vectV)
        ax2.plot(startbouts, np.zeros(len(startbouts)),
                 marker='.', ms=20, color='r')
        ax2.plot(tailanglesum)
        pl.show()
        self.bouttimes = startbouts

    # calculates x,y and vectV after all data from video has been extracated.
    # call this yourself at any time. x and y is calculated by the average of the 2 eyes.
    # z has already been calculated as the location of the fish's head.

    def fillin_and_fix(self):
        for lxy,rxy in zip(self.leftxy, self.rightxy):
            self.x.append(np.mean([lxy[0], rxy[0]]))
            self.y.append(np.mean([lxy[1], rxy[1]]))
        deltax = [b-a for a, b in toolz.itertoolz.sliding_window(2, self.x)]
        deltay = [b-a for a, b in toolz.itertoolz.sliding_window(2, self.y)]
        deltaz = [b-a for a, b in toolz.itertoolz.sliding_window(2, self.z)]
        deltat = [b-a for a, b in
                  toolz.itertoolz.sliding_window(2, self.frametimes)]
        for x, y, z, t in zip(deltax, deltay, deltaz, deltat):
            if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isnan(t):
                self.vectV.append(float('nan'))
            else:
                vect = np.array([x, y, z])
                self.vectV.append(np.sqrt(np.dot(vect, vect))/t)
        self.y = [1888 - ycoord for ycoord in self.y]
        self.z = [1888 - zcoord for zcoord in self.z]
        self.leftxy = [[xc, 1888-yc] for (xc, yc) in self.leftxy]
        self.rightxy = [[xc, 1888-yc] for (xc, yc) in self.rightxy]

    # Nanify adds nan to member lists of Varbs objects when the algoritm doesn't
    # find a value. Specific length lists of
    # nans are added depending on the member list. 
    def nanify(self, arglist):
        for arg in arglist:
            if arg == 'leftxy' or arg == 'rightxy' or arg == 'swimbcoords_top' or arg == 'swimbcoords_side':
                getattr(self, arg).append([float('NaN'), float('NaN')])
            elif arg == 'tailangle':
                getattr(self, arg).append([float('NaN'), float('NaN'),
                                          float('NaN'), float('NaN'),
                                          float('NaN'), float('NaN'), float('NaN')])
            else:
                getattr(self, arg).append(float('NaN'))

   #This method just pickles Varbs objects and saves them as a file.       
    def exporter(self, dr):
        with open(dr + 'fishdata.pkl', 'wb') as file:
            pickle.dump(self, file)



#FUNCTIONS FOR MAIN TO USE. 

#Backgrounds are provided from flparse2 as an array. Brmean subtraction makes it so the average intensity of a given image is equated to the background. 

def brsub((ret, im), background, brmean):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    br_sub = cv2.absdiff((brmean/np.mean(img)*img).astype(np.uint8),background)
    return br_sub


#Contourfinder finds the three largest contours in the top plane (eye1, eye2, swimb) by dynamically thresholding each background subtracted image. The dynamic thresholding is accomplished by recursively passing lower values of the threshval to contourfinder if 3 contours are not found. On the side, recursively decreases threshval until one contour is found that is approximately the size of an eye. Current byarea parameters are good for the 1888x1888 setup.  
 
def contourfinder(im, threshval, toporside, fishcenter):
    if toporside == 'top':
        r, th = cv2.threshold(im, threshval, 255, cv2.THRESH_BINARY) 
        rim, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
        byarea = [cv2.convexHull(cnt) for cnt in sorted(
            contours,
            key=cv2.contourArea, reverse=True)]
        contcomp = [x for x in byarea if 300 < cv2.contourArea(x) < 800]
        if len(contcomp) < 2:
            return contourfinder(im, threshval-1, 'top', fishcenter)
        if len(contcomp) >= 2:
            cont_coords = np.array([cv2.minEnclosingCircle(
                c)[0] for c in contcomp])
            pwd = pairwise_distances(cont_coords)
            eye_inds = np.where((pwd > 20) & (pwd < 45))
            if eye_inds[0].shape[0] == 2:
                eye1 = [contcomp[eye_inds[0][0]], cont_coords[
                    eye_inds[0][0]][0], cont_coords[eye_inds[0][0]][1]]
                eye2 = [contcomp[eye_inds[0][1]], cont_coords[
                    eye_inds[0][1]][0], cont_coords[eye_inds[0][1]][1]]
                dist_eye1_to_center = pairwise_distances(np.array(
                    [fishcenter, cont_coords[eye_inds[0][0]]]))[0][1]
                dist_eye2_to_center = pairwise_distances(np.array(
                    [fishcenter, cont_coords[eye_inds[0][1]]]))[0][1]
                if (80 < dist_eye1_to_center < 150) and (
                        80 < dist_eye2_to_center < 150):
                    return eye1, eye2
                else:
                    if len(contcomp) >= 5:
                        return [], []
                    return contourfinder(im, threshval-1, 'top', fishcenter)
            elif len(contcomp) < 5:
                return contourfinder(im, threshval-1, 'top', fishcenter)
            else:
                return [], []

    elif toporside == 'side':
        r, th = cv2.threshold(im, threshval, 255, cv2.THRESH_BINARY)
        rim, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
        byarea = [cv2.convexHull(cnt) for cnt in sorted(contours,
                                                        key=cv2.contourArea,
                                                        reverse=True)]
        contcomp = [x for x in byarea if 80 < cv2.contourArea(x) < 800]
        if contcomp:
            x, z, h, w = cv2.boundingRect(contcomp[0])
            return contcomp[0], z, x
        # just means you missed it.
        if threshval < 40: 
            return np.array([]), float('NaN'), float('NaN')
        else:
            return contourfinder(im, threshval-1, 'side', fishcenter)



# This function assigns the eye as either the left or the right eye and then converts the angles so that the angle reflects degree of convergence (phi) from the vertical axis. 

def eyeanglefinder(eye1, eye2, ha, center):
    ha_contour = np.mod(ha-90, 360)
    eye1rot = rotate_contour(eye1[0], ha_contour, center)
    eye2rot = rotate_contour(eye2[0], ha_contour, center)
    (eye1rotx, eye1roty), (MA1r, ma1r), eye1angle_rot = cv2.fitEllipse(eye1rot)
    (eye2rotx, eye2roty), (MA2r, ma2r), eye2angle_rot = cv2.fitEllipse(eye2rot)

    if eye1rotx < eye2rotx:
        lefteye_angle = eye1angle_rot
        righteye_angle = eye2angle_rot
        leftxy = [eye1[1], eye1[2]]
        rightxy = [eye2[1], eye2[2]]
        lrbool = True

    elif eye2rotx < eye1rotx:
        lefteye_angle = eye2angle_rot
        righteye_angle = eye1angle_rot
        leftxy = [eye2[1], eye2[2]]
        rightxy = [eye1[1], eye1[2]]
        lrbool = False
# these calculations account for rotatedrect calling angles two ways
    if 180 < lefteye_angle < 270:
        lefteye_angle = lefteye_angle - 180
    elif 90 < lefteye_angle < 180:
        lefteye_angle = -1*(180-lefteye_angle)
    elif lefteye_angle > 270:
        lefteye_angle = -1*(360-lefteye_angle)
    if 90 < righteye_angle < 180:
        righteye_angle = 180 - righteye_angle
    elif righteye_angle < 90: 
        righteye_angle = -1*righteye_angle
    elif 180 < righteye_angle < 270:
        righteye_angle = -1*(righteye_angle - 180)     
    elif righteye_angle > 270:
        righteye_angle = 360-righteye_angle
    return leftxy, lefteye_angle, rightxy, righteye_angle, lrbool


# This function just sets a lower bound on small ROIs that prevents the ROI
# from crossing into pixel space unoccupied by the image (
# i.e. no negative X values).

def lower_bound(index, win, max):
    if index - (win/2) < 0:
        lb = 0
    elif index + (win/2) >= max:
        lb = max - win
    else: 
        lb = index-(win/2)
    return int(lb)

# Rotates an image by a specified angle using a 2D rotation matrix.


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
    result = cv2.warpAffine(image, rot_mat,
                            image.shape, flags=cv2.INTER_LINEAR)
    return result, rot_mat, image_center

# Rotates a contour by a specified angle using a 2D rotation matrix.


def rotate_contour(contour, angle, center):
    angle = np.radians(angle)
    rotated_contour = []  
    for i in range(contour.shape[0]):
        xcoord = contour[i][0][0] - center[0]
        ycoord = contour[i][0][1] - center[1]
        xcoord_rotated = (xcoord*np.cos(angle) - ycoord*np.sin(
            angle)) + center[0]
        ycoord_rotated = (xcoord*np.sin(angle) + ycoord*np.cos(
            angle)) + center[1]
        rotated_contour.append([int(xcoord_rotated), int(ycoord_rotated)])
    rotated_cont = np.array(rotated_contour)
    return rotated_cont


# This function takes the full tail image, rotates it, makes an 500 by 500 ROI
# centered on the mask. finds tail by fitting a threshold sized contour,
# then going out along the edge of the contour by a specified number of points.
# at each equally spaced point, fit a point to the middle of the tail and take
# its angle with respect to the body axis.  writes to a movie.
# There are important catches for assuring that the tail is
# close enough to the swimb.

def tailfinder(eyemidpoint, swimbcoords, tailimage, heading_angle):
    win = 500
    ha_adj = np.mod(90-heading_angle, 360)
    ha_contour = np.mod(heading_angle-90, 360)
    foundtail = False
    # calculates point halfway between swimb and eye midpoint
    mask_center = np.array([(eyemidpoint[0] + swimbcoords[0])/2,
                            (eyemidpoint[1] + swimbcoords[1])/2])
    circle_mask = np.ones(tailimage.shape, np.uint8)
    cv2.circle(circle_mask, (int(mask_center[0]), int(mask_center[1])),
               60, (0, 0, 0), -1)
    # nohead will be image with a mask over the head.
    nohead = cv2.multiply(tailimage, circle_mask).astype(np.uint8)
    nohead = cv2.cvtColor(nohead, cv2.COLOR_BGR2GRAY)
    lb_x, lb_y = [lower_bound(mask_center[0],
                              win,
                              tailimage.shape[1]),
                  lower_bound(mask_center[1],
                              win, tailimage.shape[0])]
    ROItail = nohead[lb_y:lb_y+win, lb_x:lb_x+win]
    imgrot, m, c = rotate_image(ROItail, ha_adj)
    rot_color = cv2.cvtColor(imgrot, cv2.COLOR_GRAY2RGB)
    center = [ROItail.shape[1]/2, ROItail.shape[0]/2]
    rim, tail_conts, hier = cv2.findContours(ROItail, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_NONE)
    # KEY TO KEEP THIS AS CHAIN_APPROX_NONE. THIS MEANS THAT ALL CONTOUR POINTS ARE KEPT AND STRAIGHT LINES ARENT INTERPOLATED. THIS IS KEY FOR LOCALIZING SEGMENTS ALONG THE TAIL
    tail_byarea = sorted(tail_conts, key=cv2.contourArea, reverse=True)
    tail_list = [j for j in tail_byarea if cv2.contourArea(
        j) > 3000 and cv2.contourArea(j) < 12000]
    if not tail_list:
        return False, [], rot_color
    # this function filters out large contours that aren't
    # w/in 20 pixels of the swimb
    for t in tail_list:
        dist_to_sb = [np.array(pts[0]) - swimbcoords for pts in t]
        dist_mags = np.array([np.sqrt(
            np.square(x)+np.square(y)) for x, y in dist_to_sb])
        closest_pt = np.argmin(dist_mags)
        # hunt through until you find the contour within 20 pixels of the swimb
        # (see dist to swimb graph).
        if np.min(dist_mags) < 60:
            tail = t
            foundtail = True
            break
    if not foundtail:
        return False, [], rot_color
    tail_rot = rotate_contour(tail, ha_contour, center)
    if closest_pt != 0:
        # if it is equal to zero, closest point is first element to begin with
        tail_rot = np.concatenate(
            [tail_rot[closest_pt:], tail_rot[0:closest_pt]])
    # this just puts the closest point to the swimb as the first point in the list
    tail_rot = tail_rot[0::3].astype(np.int)
    # decimates the array by taking 1 point in every 3. 
    tail_perimeter = cv2.arcLength(tail_rot, True)
    tail_segment1 = []
    tail_segment2 = []
    segment = 1.0
    numsegs = 18.0
    # now go out along tail until you reach 1/18
    # (i.e. 1/numsegs) of the total perimeter recursively.
    for i in range(len(tail_rot)):
        if cv2.arcLength(tail_rot[0:i+1], False) > tail_perimeter*(
                segment/numsegs):
            # False arg in arclength specifies that the contour isn't closed.
            if segment < (numsegs/2):
                tail_segment1.append(tail_rot[i])
            elif segment > (numsegs/2):
                tail_segment2.append(tail_rot[i])
            elif segment == (numsegs/2):
                endpoint = tail_rot[i].tolist()
            segment += 1
    avgtailpoints = [[np.mean([a[0], b[0]]),
                      np.mean([a[1], b[1]])] for a, b in zip(
                          tail_segment1,tail_segment2[::-1])]
    for tp in avgtailpoints:
        cv2.ellipse(rot_color, (
            int(tp[0]), int(tp[1])), (6, 6), 0, 0, 360, (255, 0, 255), -1)
    tailgen = toolz.itertoolz.sliding_window(2,avgtailpoints)
    taildiffs = [(0, 1)] + [(b[0]-a[0], b[1]-a[1]) for a, b in tailgen]
    angles = []
    for vec1, vec2 in toolz.itertoolz.sliding_window(2, taildiffs):
        dp = np.dot(vec1, vec2)
        mag1 = np.sqrt(np.dot(vec1, vec1))
        mag2 = np.sqrt(np.dot(vec2, vec2))
        ang = np.arccos(dp / (mag1*mag2))
        if np.cross(vec1, vec2) > 0:
            ang *= -1
        angles.append(np.degrees(ang))

    return True, angles, rot_color
    # don't count first one b/c positioning of closest point is
    # unreliable and not dependent on halving the tail width. 

#Pitchfinder takes a high contrast image generated in flparse2 and applies a rectangle contour to the largest contour in the image, provided by the main line ("fish" variable). Using heading angle information from the top frame and the angle of the rectangle contour in the side frame, a cone is fit to the fish which determines the true pitch angle. zcoord and x2coord are the coordinates of the contour fit to the fish's head. 


def pitchfinder_rect(img, zcoord, x2coord, phi, fishlen, fish):
    im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    rect = cv2.minAreaRect(fish)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # gives center of bounding rectangle.
    x2center, zcenter = np.mean(box, axis=0)
    cv2.drawContours(im, [box], 0, (0, 0, 255), 2)
    cv2.ellipse(im, (int(x2center), int(zcenter)),
                (2, 2), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(im, (int(x2coord), int(zcoord)),
                (2, 2), 0, 0, 360, (0, 255, 0), -1)
    point1 = box[0]
    distmat_wrt1 = [np.sqrt(np.dot(a-point1,
                                   a-point1)) for a in box[1:]]

    # this list shows that distances from pt 1 are always mid,largest,shortest.
    # if it is a square, pitch is zero. have to catch this b/c can't have two identical distances for rest of calc. 
    if len(np.unique(distmat_wrt1)) != 3:
        return im, 0, float('NaN')
    (midarg,), = np.argwhere(distmat_wrt1 == np.median(distmat_wrt1))
    point2 = box[1:][midarg]
    mag_fish = distmat_wrt1[midarg]
    short_axis = np.min(distmat_wrt1)
    # angle of the long axis of the bounding rectangle
    vert_angle = np.mod(np.degrees(
        np.arctan2((point2[1]-point1[1]), (point2[0]-point1[0]))), 360)

    # Sometimes you get the top point and sometimes you get the bottom,
    # which should be distinguished by correction below. 
    vert_angle_rot = np.mod(180+vert_angle, 360)
    # this is for validating the angle call. 
    validate = [x2coord-x2center, zcoord-zcenter]
    # this is CORRECT. provides the
    # correct angle for all frames between blue and red dot.
    val_angle = np.mod(np.degrees(np.arctan2(validate[1], validate[0])), 360)
    # asks which angle is closer to 180 different
    angle_distances = [np.abs(180-np.abs(val_angle-vert_angle)),
                       np.abs(180-np.abs(val_angle-vert_angle_rot))]

    # Gamma is the cone derived true pitch of the fish.
    if np.argmin(angle_distances) == 0:
        gamma = vert_angle_rot
    elif np.argmin(angle_distances) == 1:
        gamma = vert_angle
    # now put final vert angle from a clockwise 0-360 scale to a unit circle
    # scale with positive and negative values.
    if gamma <= 90:
        gamma = -1*gamma
    elif 90 < gamma <= 180:
        gamma = -1*(180-gamma)
    elif 180 < gamma <= 270:
        gamma = gamma-180
    elif gamma > 270:
        gamma = 360-gamma

  #If fish is facing directly away or towards the side camera, finds the pitch based on the length of the fish compared to the maximum length observed in other frames. you can find the fish length when phi is 180 or 360, which indicates the fish is fully horizontally expanded in the side camera. 

    if (phi < 5 or phi > 355) or (175 < phi < 185):
        fishlength = mag_fish
    else:
        fishlength = float('NaN')

    if (80 < phi < 100) or (260 < phi < 280):
        if mag_fish / float(short_axis) < 1.3:
            return im, 0, fishlength
        if gamma > 0 and not math.isnan(fishlen):
            if fishlen < mag_fish:
                fishlen = mag_fish
                fishlength = mag_fish
            pitch = np.degrees(np.arcsin((float(mag_fish)/fishlen)))
            return im, pitch, fishlength
        elif gamma < 0 and not math.isnan(fishlen):
            if fishlen < mag_fish:
                fishlen = mag_fish
                fishlength = mag_fish
            pitch = -1*np.degrees(np.arcsin((float(mag_fish)/fishlen)))
            return im, pitch, fishlength
        else:
            return im, float('NaN'), fishlength

  #now transform gamma, which is the 2D projection pitch, into 3D cone with opening angle theta, which is twice the angle of the fish off the Z axis. you get the x position of the fish on the cone and the height of hte cone from gamma, assuming a unit length fish (which is fine). the h and r will change depending on these two variables and phi, which says "given that the fish is at this x position and the heading angle is phi, the cone has this h and r, which yields theta"
  #transform this back into a z angle from the x axis. eq. of cone is 2arctan(r/h) where r is radius of cone and h is height. each gamma and phi uniquely determine a theta of a cone
    conegamma = np.radians(np.abs(gamma))
    phi = np.radians(phi)
    # see little red notebook for this calculation
    r_over_h = 1/(np.cos(phi)*np.tan(conegamma))
    if np.pi/2 < phi < 3*np.pi/2:
        r_over_h *= -1
    theta = 2*np.degrees(np.arctan(r_over_h))
    pitch = 90 - (theta/2.0)
    if gamma < 0:
        pitch *= -1
    return im, pitch, fishlength
  
  #can incorporate heading angle into this calculation; if heading angle is a certain value, rely on ratio between rectangle long vs short edges to determine pitch. otherwise, use pitch info. make sure that when you're using the ratio for pitch you do NOT use halftheta. use locaiton of eye contour with respect to the center of the bounding rectangle to hash out which angle call is correct. 

# MAINLINE


def fix_blackline(im):
    cols_to_replace = [833, 1056, 1135, 1762, 1489]
    for c in cols_to_replace:
        col_replacement = [int(np.mean([a, b])) for a, b in zip(
            im[:, c-1], im[:, c+1])]
        im[:, c] = col_replacement
    return im


def get_fish_data(data_directory):
  
    low_res = False
    fc = cv2.CAP_PROP_FRAME_COUNT
    curr_frame = cv2.CAP_PROP_POS_FRAMES 
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cam0id = [
        file_id for file_id in os.listdir(data_directory) if file_id[-8:]
        == 'cam0.AVI'][0]
    cam1id = [
        file_id for file_id in os.listdir(data_directory) if file_id[-8:]
        == 'cam1.AVI'][0]
    top = cv2.VideoCapture(data_directory + cam0id)
    side = cv2.VideoCapture(data_directory + cam1id)
    tailvid = cv2.VideoCapture(data_directory + 'top_contrasted.AVI')
    pitchvid = cv2.VideoCapture(data_directory + 'side_contrasted.AVI')
    framecount = int(top.get(fc))
    startframe = 0
    endframe = framecount
    #  endframe = 5000
    epoch_boundaries = np.cumsum(np.load(data_directory + 'framecounts.npy'))
    ir_freq_by_epoch = np.load(data_directory + 'ir_freqs.npy')
    mode_index = np.load(data_directory + 'mode_indices.npy').tolist()
    frame_types = np.load(data_directory + 'frametypes.npy').tolist()
    top_ir_backgrounds = np.load(data_directory + 'backgrounds_top.npy')
    side_ir_backgrounds = np.load(data_directory + 'backgrounds_side.npy')    
    top.set(curr_frame, startframe)
    side.set(curr_frame, startframe)
    # this won't work with startframe not zero because they only contain IR frames. 
    tailvid.set(curr_frame, startframe) 
    pitchvid.set(curr_frame,startframe)
    window = 500
    varbs = Variables()
    varbs.frametimes = np.load(data_directory + 'frametimes_ir.npy')
    tailcontvid = cv2.VideoWriter(data_directory + 'tailcontvid.AVI', fourcc,
                                  60, (window, window), True)
    contvid = cv2.VideoWriter(data_directory + 'conts.AVI', fourcc,
                              60, (window, window), True)
    sideconts = cv2.VideoWriter(data_directory + 'sideconts.AVI',
                                fourcc, 60, (1888, 1888), True)
    fishcenter = np.array([float('NaN'),float('NaN'), float('NaN')])
    eye1, eye2 = [], []
    fc_side = [float('NaN'), float('NaN')]
    fishlength_list = []
    avg_fishlength = float('NaN')

    # Now simply run through the videos and apply functions above to find all
    # Varbs member lists for entire experiment.

    brcounter = 0
    if startframe != 0:
        for k in range(0, startframe):
            if k in mode_index:
                brcounter += 1

    if ir_freq_by_epoch[0] < 10:
        low_res = True
        ir_freq_by_epoch = ir_freq_by_epoch[1:]

    for i in range(startframe, endframe, 1):

        if i == 0 or i in mode_index:
            ir_t_br = top_ir_backgrounds[brcounter].astype(np.uint8)
            ir_s_br = side_ir_backgrounds[brcounter].astype(np.uint8)
            # normalize each frame to the mean of background
            brmean_t = np.mean(ir_t_br)
            brmean_s = np.mean(ir_s_br)
            brcounter += 1 

        if i in epoch_boundaries:
            if ir_freq_by_epoch[0] < 10:
                low_res = True
            else:
                low_res = False
            ir_freq_by_epoch = ir_freq_by_epoch[1:]

        if i % 20 == 0:
            print i 

        if frame_types[i] == 1:
            top.read()
            side.read() 
        #dont read tail or pitchvids b/c they only contain IR frames already.
        # this just passes through the cam0 and cam1 frames that are fluorescent.

        else:    
            topim = brsub((top.read()), ir_t_br, brmean_t)
            topim = fix_blackline(topim)
            sideim = brsub((side.read()), ir_s_br, brmean_s)
            r, pitchframe = pitchvid.read()
            r, tail = tailvid.read()
            if not r:
                break
            r, pitchframe = cv2.threshold(pitchframe, 120, 255,
                                          cv2.THRESH_BINARY)
            r, tail = cv2.threshold(tail, 120, 255, cv2.THRESH_BINARY)
            rim, contours_t, hierarchy = cv2.findContours(
                cv2.cvtColor(tail, cv2.COLOR_BGR2GRAY),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rim, contours_s, hierarchy = cv2.findContours(
                cv2.cvtColor(pitchframe, cv2.COLOR_BGR2GRAY),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_t = sorted(contours_t, key=cv2.contourArea, reverse=True)
            contours_s = sorted(contours_s, key=cv2.contourArea, reverse=True)
            if (not contours_t) or (not contours_s):
                varbs.low_res_x.append(float('nan'))
                varbs.low_res_y.append(float('nan'))
                varbs.low_res_z.append(float('nan'))
                varbs.nanify(['phileft', 'phiright', 'leftxy', 'rightxy',
                              'headingangle', 'tailangle', 'swimbcoords_top',
                              'z', 'x2', 'pitch'])
                contvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                tailcontvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                sideconts.write(np.zeros([1888, 1888, 3]).astype(np.uint8))
                continue
            fishcenter, rad = cv2.minEnclosingCircle(contours_t[0]) 
            fishcenter_side, rad = cv2.minEnclosingCircle(contours_s[0])
            fish_xy_moments = cv2.moments(contours_t[0])
            fish_com_x = int(fish_xy_moments['m10']/fish_xy_moments['m00'])
            fish_com_y = int(fish_xy_moments['m01']/fish_xy_moments['m00'])
            varbs.low_res_x.append(fishcenter[0])
            varbs.low_res_y.append(1888 - fishcenter[1])
            varbs.low_res_z.append(1888 - fishcenter_side[1])
            if low_res:
                varbs.nanify(['phileft', 'phiright', 'leftxy', 'rightxy',
                              'headingangle', 'tailangle', 'swimbcoords_top',
                              'z', 'x2', 'pitch'])
                contvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                tailcontvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                sideconts.write(np.zeros([1888, 1888, 3]).astype(np.uint8))
                continue

            # finds end of movies.
            if not r:
                break

            if not math.isnan(fishcenter[0]):
                x_lb, y_lb = [lower_bound(fishcenter[0], window,
                                          topim.shape[1]),
                              lower_bound(fishcenter[1], window,
                                          topim.shape[0])]
                com_ROI_x = fish_com_x - x_lb
                com_ROI_y = fish_com_y - y_lb
                ROI = topim[y_lb:y_lb+window, x_lb:x_lb+window]
                tailROI = tail[y_lb:y_lb+window, x_lb:x_lb+window]
                swimb = np.array([com_ROI_x, com_ROI_y])
                eye1, eye2 = contourfinder(ROI, 150, 'top', swimb)
              #  conts = contourfinder(ROI,150,'top')
              #  eye1,eye2 = contourparser(conts)


      #this catches if contours are not found. adds nans to all member lists that are dependent on contour finding. goes to find next frame. 
            if not eye1 or math.isnan(fishcenter[0]): 
                varbs.nanify(['phileft', 'phiright', 'leftxy', 'rightxy',
                              'headingangle', 'tailangle',
                              'swimbcoords_top', 'z', 'x2', 'pitch'])
                contvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                tailcontvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                sideconts.write(np.zeros([1888, 1888, 3]).astype(np.uint8))
                continue

            elif eye1[0].shape[0] < 5 or eye2[0].shape[0] < 5:
                varbs.nanify(['phileft', 'phiright', 'leftxy', 'rightxy',
                              'headingangle', 'tailangle', 'swimbcoords_top',
                              'z', 'x2', 'pitch'])
                contvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                tailcontvid.write(np.zeros([window, window, 3]).astype(np.uint8))
                sideconts.write(np.zeros([1888, 1888, 3]).astype(np.uint8))
                continue

            ROIcolor = cv2.cvtColor(ROI, cv2.COLOR_GRAY2RGB)
            # each return from contourparser is a list. first element is the
            # contour itself, second is the xcoord, third is ycoord
            eye_midpoint = np.array([(eye1[1]+eye2[1])/2, (eye1[2]+eye2[2])/2])
            # can choose to base heading off position of the
            # sb plus eyes or eyes only.
            heading_vector_sb = eye_midpoint - swimb
            heading_vector = np.cross([eye1[1] - eye2[1], eye1[2] - eye2[2], 0],
                                      [0, 0, 1])[0:2]
            if np.dot(heading_vector, heading_vector_sb) < 0:
                heading_vector *= -1         
            heading_angle = 360 - np.mod(np.degrees(np.arctan2(
                heading_vector[1], heading_vector[0])), 360) 
            # 360 - is because y is from top left corner going downwards. 
            # this angle will give the heading according to unit circle angle coordinates
            foundtail, tailangles, tailimage = tailfinder(
                eye_midpoint,
                np.array([swimb[0], swimb[1]]), tailROI, heading_angle)
            tailcontvid.write(tailimage) 
            leftxy, phileft, rightxy, phiright, l_or_r = eyeanglefinder(
                eye1, eye2, heading_angle, np.array(ROI.shape)/2)
      #After operating on the ROI, return all found values to their rightful full frame coordinates. 
            leftxy_fullframe = [leftxy[0]+x_lb, leftxy[1]+y_lb]
            rightxy_fullframe = [rightxy[0]+x_lb, rightxy[1]+y_lb]
            swimb_fullframe = [swimb[0]+x_lb, swimb[1]+y_lb]
            varbs.leftxy.append(leftxy_fullframe)
            varbs.rightxy.append(rightxy_fullframe)
            varbs.swimbcoords_top.append(swimb_fullframe)
            varbs.phileft.append(phileft)
            varbs.phiright.append(phiright)
            varbs.headingangle.append(heading_angle)

       #Draw contours on frames to be written to movies to assure that you're seeing correct calls. 

            cv2.circle(ROIcolor, tuple(np.array(leftxy).astype(np.int)),
                       3, (255, 255, 0), -1) #yellow
            cv2.circle(ROIcolor, tuple(np.array(rightxy).astype(np.int)),
                       3, (0, 255, 255), -1) #light blue
            cv2.circle(ROIcolor, tuple(np.array(
                [com_ROI_x, com_ROI_y])), 3, (255, 0, 255), -1)
            if l_or_r:
            # l_or_r true means eye1 is the left eye. False means eye1 is the right eye. 
                el1 = cv2.fitEllipse(eye1[0])
                el2 = cv2.fitEllipse(eye2[0])
                cv2.ellipse(ROIcolor, el1, (255, 0, 0), 1)
                cv2.ellipse(ROIcolor, el2, (0, 255, 0), 1)
            else:
                el1 = cv2.fitEllipse(eye1[0])
                el2 = cv2.fitEllipse(eye2[0])
                cv2.ellipse(ROIcolor, el1, (0, 255, 0), 1)
                cv2.ellipse(ROIcolor, el2, (255, 0, 0), 1)
            # This writes the drawn contour frames into a movie called contvid. 
            contvid.write(ROIcolor)
            if foundtail: 
                varbs.tailangle.append(tailangles)
            else: 
                varbs.nanify(['tailangle'])

            if not math.isnan(fishcenter_side[0]):
                x2_lb, z_lb = [lower_bound(fishcenter_side[0],
                                           window, sideim.shape[1]),
                               lower_bound(fishcenter_side[1],
                                           window, sideim.shape[0])]
                sideROI = sideim[z_lb:z_lb+window,
                                 x2_lb:x2_lb+window]
                sidecontour, zpos, x2pos = contourfinder(
                    sideROI, 150, 'side', fishcenter_side - np.array(
                        [x2_lb, z_lb]))
            # position of first contour to pop up that is within an area threshold in contourfinder.
            if not sidecontour.any() or math.isnan(fishcenter_side[0]):
                varbs.nanify(['z', 'pitch', 'x2'])
                sideconts.write(np.zeros([1888, 1888, 3]).astype(np.uint8))
                continue
            fc_side = [x2pos+x2_lb, zpos+z_lb]
            pitchframe = cv2.cvtColor(pitchframe, cv2.COLOR_BGR2GRAY)
            ROIc, pitch, fishlength = pitchfinder_rect(
                pitchframe, fc_side[1], fc_side[0], heading_angle,
                avg_fishlength, contours_s[0])
            sideconts.write(ROIc)    
            if not math.isnan(fishlength) and len(fishlength_list) < 1000:
                fishlength_list.append(fishlength)
                fishlength_list.sort()
                avg_fishlength = np.mean(fishlength_list[-20:])
            # avg of the top 20 fish lengths taken over the first 15 seconds of the movie
            # used to find pitch if fish is looking directy at or away from side camera.
            # since dark for first 8000 frames, all pitch calc in light is same
            varbs.z.append(np.float_(fc_side[1])) 
            varbs.x2.append(np.float_(fc_side[0]))
            varbs.pitch.append(pitch)

    top.release()
    side.release()
    contvid.release()
    tailcontvid.release()
    tailvid.release()
    sideconts.release()
    return varbs


def wrap_ir(dr):
    output = get_fish_data(dr)
    output.fillin_and_fix()
    output.exporter(dr)

   
if __name__ == '__main__':
    dr = raw_input("Enter Directory of Data: ")
#    dr = os.getcwd() + '/'
    wrap_ir(dr)
