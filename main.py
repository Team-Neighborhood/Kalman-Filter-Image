import argparse
import numpy as np
import pandas as pd
import cv2
import os
from os.path import join

import kalman

NUM_JOINT=17

np.set_printoptions(precision=3, suppress=True)

cv2.namedWindow('show', 0)
cv2.resizeWindow('show', 960, 540)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?',
                        default='data/jannabi_clip.csv',
                        type=str)
    args = parser.parse_args()
    return args

def parse_csv(config):
    pd_data = pd.read_csv(config.input, header=None)
    np_data = pd_data.values
    np_filename = np_data[:,0]
    np_points = np_data[:,1:]
    return np_filename, np_points

def point2xyv(kp):
    kp = np.array(kp)
    x = kp[0::3].astype(int)
    y = kp[1::3].astype(int)
    v = kp[2::3].astype(int) # visibility, 0 = Not visible, 0 != visible
    return x,y,v

def main():
    config = parse_args()
    filenames, np_points = parse_csv(config)
    
    ''' initialize kalman filter for 17 keypoints '''
    list_KFs = []
    for i in range(NUM_JOINT):
        KF = kalman.KF2d( dt = 1 ) # time interval: '1 frame'
        init_P = 1*np.eye(4, dtype=np.float) # Error cov matrix
        init_x = np.array([0,0,0,0], dtype=np.float) # [x loc, x vel, y loc, y vel]
        dict_KF = {'KF':KF,'P':init_P,'x':init_x}
        list_KFs.append(dict_KF)

    ''' image sequence loop '''
    for idx in range(len(filenames)):
        name = filenames[idx]
        
        list_measured = np_points[idx]
        kx,ky,kv = point2xyv(list_measured) # x, y, visiblity

        img = cv2.imread(join('./data/jannabi_clip', name), 1)

        list_estimate = [] # kf filtered keypoints

        cnt_validpoint = 0
        start = cv2.getTickCount()
        for i in range(NUM_JOINT):

            z = np.array( [kx[i], ky[i]], dtype=np.float)

            KF = list_KFs[i]['KF']
            x  = list_KFs[i]['x']
            P  = list_KFs[i]['P']
            
            x, P, filtered_point = KF.process(x, P, z)

            list_KFs[i]['KF'] = KF
            list_KFs[i]['x']  = x
            list_KFs[i]['P']  = P

            # visibility
            v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 2

            list_estimate.extend(list(filtered_point) + [v]) # x,y,v
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000 # ms
        print ('[INFO] %d kfs aver time: %.2fms'%(NUM_JOINT, time/NUM_JOINT))

        ''' draw image '''
        show = img.copy()

        mx,my,mv = point2xyv(list_measured)
        fx,fy,fv = point2xyv(list_estimate)

        color = (0,255,0)
        for i in range(NUM_JOINT):
            if fv[i] != 0: cv2.circle(show, (fx[i],fy[i]), 3, color, -1, 1)
            if mv[i] != 0: cv2.circle(show, (mx[i],my[i]), 1, (0,0,255), -1, 1)

        cv2.imshow('show', show)
        key = cv2.waitKey(20)
        if key == 27:
            break


if __name__ == '__main__':
    main()