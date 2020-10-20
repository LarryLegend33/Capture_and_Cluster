import pyb
import time

# enter 1000 for continuous white light freq
# (i.e. fused), and duty_cycle of 100.
# otherwise,
# enter desired light blink frequency. 5 and 10 to start

# freq in Hz, duty cycle in %
white_light_frequency = 1000
white_light_duty_cycle = 100
# white_light_onset = .1
white_light_onset = 0
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
fl_dwell = .2
ir_dwell = .002


def set_uv(duty_cycle):
    uv_ch.pulse_width_percent(duty_cycle)


def set_whitelight(duty_cycle):
    wl_ch.pulse_width_percent(duty_cycle)

# If you want to do variable dimness, you can find the current state using
# ch.pulse_width(), which will return 84000 for dim(100) and 21000 for dim(25).


set_whitelight(0)
ir.high()


def full_experiment(dm, run_now):
  
# Args to epochs 1: whether you want to do fluorescence during epoch
#                2: The duration in minutes of the epoch
#                3: The frequency of IR acquisiton in Hz. MUST TRANSLATE TO AN INTEGER MS PERIOD. (i.e. entering
#                   60 Hz here, will get 62.5 Hz acquisition b/c Micropython rounds your 16.66 ms request to 16. If using FL, want an integer frequency as well that evenly divides fl_dwell
#                 e.g. 5 frames fit inside fl_dwell would be 40ms period = 25 Hz. 10 would be 20 ms period = 50 Hz. Minimum 5 Hz for tracking is probably good.  
#                4: The frequency of fluor imaging during epoch. MUST CREATE A PERIOD THAT IS IN SECONDS. (e.g. .2 = 5 sec GOOD, .4 = 2.5 sec BAD, .5 = 2 sec GOOD). 
#                    

#Note that calling this function with a False for run_now will return the amount of frames the entire experiment will grab. 

# 10 mins at 62.5 Hz w/ no fluor is 37500 frames. 
#  exp_dict = {'epoch1': [False,90,1,10]}

    exp_dict = { 'epoch1': [True, 30, 5, .2]}
    dict_file = open('log/experiment.txt', 'w')
    dict_file.write(str(exp_dict))
    dict_file.close()
    light_file = open('log/light.txt', 'w')
    light_file.write(str(white_light_frequency))
    light_file.write(str(white_light_duty_cycle))
    light_file.write(str(white_light_onset))
    light_file.close()
    total_frames = 0
    if run_now:
        time.sleep(10)
        for epoch, entry in enumerate(exp_dict):
            print('in epoch loop')
            flash(*exp_dict['epoch' + str(epoch+1)])
    else:
        for epoch, entry in enumerate(exp_dict):
            total_frames += frames_in_epoch(exp_dict['epoch' + str(epoch+1)])
        print(total_frames)

# dictionary won't necessarily come out in order. doesnt matter here.
# ir_frames_per_fl_frame is amount of ir frames that can fit inside an fl_dwell (i.e. each ir frame taks 1/params[2] seconds)
# every time a fl frame happens, it takes up ir_frames_per_fl_frame worth of would-be ir frames; subtract this value (minus 1 b/c one frame is taken)
# from the ir count. 
    
# note .003 is required for exposure plus .001 leway after lights on

def frames_in_epoch(params):
    framecount = 0
    ir_frame_count = params[1]*60*params[2]
    if not params[0]:
        framecount = ir_frame_count
    else:
        ir_frames_per_fl_frame = fl_dwell / (1/params[2])
        fl_frame_count = params[1]*60*params[3]
        framecount = ir_frame_count - ((ir_frames_per_fl_frame - 1) * fl_frame_count)
    return framecount



def flash(fluor, ep_dur, ir_freq, fl_freq):
    ir_sleep = 1.0/ir_freq - (ir_dwell + .001)
    numframes = frames_in_epoch([fluor, ep_dur, ir_freq, fl_freq])
    ir_frames_per_fl_frame = fl_dwell / (1/ir_freq)
    # 1/fl_freq seconds pass between fl frames. in this time, the following number of frames are taken. add 1 b/c fl frame is taken in place of an ir frame. 
    fl_frame_modulus = (1 / fl_freq) * ir_freq - ir_frames_per_fl_frame + 1
    if fluor:
        ir.high()
        count = 0
        print('got framecount')
        for framecount in range(numframes):
            if count % fl_frame_modulus != 0:
                # allows time for ir_lights to come back on after fl_frame
                time.sleep(.001)
                camtrig.high()
                camtrig2.high()
                time.sleep(ir_dwell)
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













   
     
    








