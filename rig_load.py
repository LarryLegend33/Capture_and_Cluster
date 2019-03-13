import pyb
import time

# enter 1000 for continuous white light freq
# (i.e. fused), and duty_cycle of 100
# otherwise,
# enter desired light blink frequency

white_light_frequency = 5
white_light_duty_cycle = 10
white_light_onset = .2
whitelight = pyb.Pin('X1')
wl_tim = pyb.Timer(2, freq=white_light_frequency)
wl_ch = wl_tim.channel(1, pyb.Timer.PWM, pin=whitelight)
uv = pyb.Pin('Y4')
uv_timer = pyb.Timer(4, freq=1000)
uv_ch = uv_timer.channel(4, pyb.Timer.PWM, pin=uv)
fl = pyb.Pin('X11', pyb.Pin.OUT_PP)
camtrig = pyb.Pin('Y1', pyb.Pin.OUT_PP)
camtrig2 = pyb.Pin('X9', pyb.Pin.OUT_PP)
ir = pyb.Pin('Y7', pyb.Pin.OUT_PP)


def set_uv(duty_cycle):
    uv_ch.pulse_width_percent(duty_cycle)


def set_whitelight(duty_cycle):
    wl_ch.pulse_width_percent(duty_cycle)

# If you want to do variable dimness, you can find the current state using
# ch.pulse_width(), which will return 84000 for dim(100) and 21000 for dim(25).


set_whitelight(0)
ir.high()


def full_experiment(fluor_in_highres, run_now):
  
# Args to epochs  1: whether you want to do fluorescence during highres
#                2: The duration in minutes of the epoch
#                3: The frequency of IR acquisiton in Hz. MUST TRANSLATE TO AN INTEGER MS PERIOD. (i.e. entering
#                   60 Hz here, will get 62.5 Hz acquisition b/c Micropython rounds your 16.66 ms request to 16. 
#                4: The amount of time in seconds between Fluorescence Frames during the epoch.

#Note that calling this function with a False for run_now will return the amount of frames the entire experiment will grab. The duration key ignores the presence of fluorescent frames because they are so rare. 

   #exp_dict = {'epoch1': [True,5,5,10],
   #            'epoch2': [fluor_in_highres,5,62.5,10],
   #            'epoch3': [True, 20,5,10],
   #            'epoch4': [fluor_in_highres,5,62.5,10],
   #            'epoch5': [True,40,5,10],
   #            'epoch6': [fluor_in_highres,5,62.5,10],
   #            }

# 10 mins at 62.5 Hz w/ no fluor is 37500 frames. 
#   exp_dict = {
 #              'epoch1': [False,10,62.5,10],
#               }

  # exp_dict = { 'epoch1': [True,60,1,10],}

    exp_dict = {'epoch1': [True, 20, 5, 10]}
    dict_file = open('log/experiment.txt', 'w')
    dict_file.write(str(exp_dict))
    dict_file.close()

    light_file = open('log/light.txt', 'w')
    light_file.write(str(white_light_frequency))
    light_file.write(str(white_light_duty_cycle))
    light_file.write(str(white_light_onset))
    light_file.close()

    if run_now:
        time.sleep(10)
        for epoch, entry in enumerate(exp_dict):
            print('in epoch loop')
            flash(*exp_dict['epoch' + str(epoch+1)])

# dictionary won't necessarily come out in order. doesnt matter here.             
    def total_frames():
        framecount = 0
        for epoch in exp_dict:
            params = exp_dict[epoch]
            framecount += params[1]*60*params[2]
        return framecount

    print(total_frames())

# provide fl_interval in seconds (ie one frame ever 10 sec),
# ir_freq for acquisition in hz. duration of epoch in minutes. for each,
# the total number of frames will be duration*60*ir_freq.
#.003 is required for exposure plus .001 leway after lights on


def flash(fluor, duration, ir_freq, fl_interval):
    ir_sleep = 1.0/ir_freq - .003
    fl_dwell = .2
# fl_count is directly related to ir_freq by...
    fl_count = int(ir_freq*fl_interval)
    numframes = int(duration*60*ir_freq)
    if fluor:
        ir.high()
        count = 0
        print('got framecount')
        for framecount in range(numframes):
            if count % fl_count != 0:
                # allows time for ir_lights to come back on after fl_frame
                time.sleep(.001)
                camtrig.high()
                camtrig2.high()
                time.sleep(.002)
                camtrig.low()
                camtrig2.low()
                time.sleep(ir_sleep)

# have to have identical post low times it seems for fl and ir frames.
# if this is .009 and bottom is .007, drops. but if both .007, fine.
            else:
                ir.low()
                fl.high()
                time.sleep(.001)
                print(count)
                camtrig.high()
                camtrig2.high()
                time.sleep(fl_dwell)
                camtrig.low()
                camtrig2.low()
                time.sleep(ir_sleep)   #if this is commented out and .005 is commented out, still drops. if this is .007, drops. 
                fl.low()
                ir.high()
            print('in fluor')
            if count == int(numframes * white_light_onset):
                set_whitelight(white_light_duty_cycle)
            if count == numframes - 1:
                set_whitelight(0)
            count += 1

    elif not fluor:
        ir.high()
        for framecount in range(numframes):
            if framecount == int(numframes * white_light_onset):
                set_whitelight(white_light_duty_cycle)
            if framecount == numframes - 1:
                set_whitelight(0)
            time.sleep(.001)
            camtrig.high()
            camtrig2.high()
            time.sleep(.002)
            camtrig.low()
            camtrig2.low()
            time.sleep(ir_sleep)
            # don't have to have a leading .001 because lights aren't changing.


def onepic(timeduration, fl_or_ir):
    if fl_or_ir == 'ir':
        ir.high()
    elif fl_or_ir == 'fl':
        fl.high()
    camtrig.high()
    camtrig2.high()
    time.sleep(timeduration / 1000)
    camtrig.low()
    camtrig2.low()
    ir.low()
    fl.low()













   
     
    








