import cv2
import numpy as np

class KF2d():
    '''
        x = [x_position, x_velocity, y_position, y_velocity]
    '''
    def __init__(self, dt=1):
        super(KF2d, self).__init__()
        self.dt = dt
        self.A = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ], dtype=np.float)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        self.Q = 0.9*np.eye(4, dtype=np.float)
        self.R = np.array([
            [100, 0],
            [0, 100]
        ], dtype=np.float)
        
        self.zero_cnt = 0
        self.flg_disappear = True
    
    def kalman_main(self, x, P, z):
        ''' prediction '''
        xp = self.A @ x
        Pp = self.A @ P @ self.A.T + self.Q
        
        ''' kalman gain '''
        # ''' original code '''
        # K = Pp @ self.H.T @ np.linalg.inv( self.H @ Pp @ self.H.T + self.R)

        # ''' optimization code : remove matrix inverse '''
        # 40% fatser than original code
        HPH = np.array([
            [Pp[0,0], Pp[0,2]],
            [Pp[2,0], Pp[2,2]],
        ])
        HPHR = HPH + self.R
        inv_HPHR = np.array([
                [HPHR[1,1], -HPHR[0,1]],
                [-HPHR[1,0], HPHR[0,0]]
                ])
        inv_HPHR /= (HPHR[0,0]*HPHR[1,1]-HPHR[0,1]*HPHR[1,0])
        K = Pp @ self.H.T @ inv_HPHR

        ''' estimation '''
        x = xp + K @ (z - self.H @ xp)
        P = Pp - K @ self.H @ Pp        # Error covariance matrix
        
        return x, P

    def preprocess(self, x, P, z):
        '''
        Not measured case
        zero count++ , replacec z into estimate
        '''
        if z[0]!=0 and z[1]!=0 and self.flg_disappear == True:
            self.flg_disappear = False
            x[0] = z[0]
            x[1] *= 0.1
            x[2] = z[1]
            x[3] *= 0.1
            P = 0 * np.eye(4, dtype=np.float)

        if x[0]!=0 and x[2]!=0 and z[0]==0 and z[1]==0:
            z[0] = x[0]
            z[1] = x[2]
            self.zero_cnt += 1
        else:
            self.zero_cnt = 0

        if self.zero_cnt >= 5:
            self.zero_cnt = 0
            self.flg_disappear = True
            x = np.array([0,0,0,0], dtype=np.float)
            P = 0 * np.eye(4, dtype=np.float)

        if abs(x[1]) > 5 or abs(x[3]) > 5:
            x[0] = z[0]
            x[1] *= 0.1
            x[2] = z[1]
            x[3] *= 0.1
            P = 0 * np.eye(4, dtype=np.float)
        
        return x, P, z

    def postprocess(self, x, P):

        if abs(x[1]) > 10 or abs(x[3]) > 10:
            output = (0,0)
        else:
            output = (int(round(x[0])), int(round(x[2])))
        if x[0] < 10 or x[2] < 10:
            output = (0,0)

        return x, P, output

    def process(self, x, P, z):
        o = None
        
        x, P, z = self.preprocess(x, P, z)
        x, P    = self.kalman_main(x, P, z)
        x, P, o = self.postprocess(x, P)

        return x, P, o