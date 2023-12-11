import os
import cv2
import numpy as np
from scipy.optimize import least_squares

class VisualOdometry():
    
    def __init__(self, data_dir):
        
        #load data
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_cam_params(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.imgs_l = self._load_images(os.path.join(data_dir, 'image_l'))
        self.imgs_r = self._load_images(os.path.join(data_dir, 'image_r'))
        
        #initialise fast feature detector
        '''
        pparameters are defined based on the below link
        link: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
        '''
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [np.divide(self.disparity.compute(self.imgs_l[0], self.imgs_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()
        
        #initialise optical flow tracking parameters
        '''
        pparameters are defined based on the below link
        link: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
        '''
        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_cam_params(file_path):
        
        with open(file_path, 'r') as f:
            
            P = [None]*2
            K = [None]*2
            for i in range(2):
                str_cam_params = np.fromstring(f.readline(), dtype = np.float64, sep = ' ')
                P[i] = np.reshape(str_cam_params, (3, 4))
                K[i] = P[i][0:3, 0:3]
            
            P_l, P_r = P
            K_l, K_r = K
            
        return K_l, P_l, K_r, P_r
    
    @staticmethod
    def _load_poses(file_path):
        
        poses = []
        
        with open(file_path, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype = np.float64, sep = ' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
                
        return poses 
    
    @staticmethod
    def _load_images(file_path):
        
        img_paths = [os.path.join(file_path, f) for f in sorted(os.listdir(file_path))]
        
        imgs = [None]*len(img_paths)
        for i, path in enumerate(img_paths):
            imgs[i] = cv2.imread(path)
        
        return imgs
    
    @staticmethod
    def _get_Tmat(R, t):
        
        T = np.eye(4, dtype = np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
    
        return T
    
    def get_tiled_kps(self, img, win_h, win_w):
        
        def get_kps(x, y):
            
            img_patch = img[y:y+win_h, x:x + win_w]
            
            kpts = self.fastFeatures.detect(img_patch)
            
            for kpt in kpts:
                kpt.pt = (kpt.pt[0] + x, kpt.pt[1] + y)
                
            if len(kpts) > 10:
                #sorted => default: ascending order => reverse = True
                kpts = sorted(kpts, key = lambda x: x.response, reverse = True)
                return kpts[:10]
            
            return kpts
            
        h, w, c = img.shape
        
        kps_list = [get_kps(x, y) for y in range(0, h, win_h) for x in range(0, w, win_w)]
        
        kp_list_flatten = np.concatenate(kps_list)
        
        return kp_list_flatten
    
    def track_kpts(self, p_img, c_img, p_kps, max_err = 4):
        
        t_p_kps = np.expand_dims(cv2.KeyPoint_convert(p_kps), axis = 1)
        t_c_kps, st, err = cv2.calcOpticalFlowPyrLK(p_img, c_img, t_p_kps, None, **self.lk_params)
        
        msk_tracking = st.astype(bool)
        
        msk_max_err = np.where(err[msk_tracking] < max_err, True, False)
        
        p_trackkpts = t_p_kps[msk_tracking][msk_max_err]
        c_trackkpts = t_c_kps[msk_tracking][msk_max_err]
        
        h, w, _ = p_img.shape
        msk_in_bound = np.where(np.logical_and(c_trackkpts[:, 1] < h, c_trackkpts[:, 0] < h), True, False)
        p_trackkpts = p_trackkpts[msk_in_bound]
        c_trackkpts = c_trackkpts[msk_in_bound]
        
        return p_trackkpts, c_trackkpts
    
    def compute_feature_kpts(self, p_trackkpts_l, c_trackkpts_l, p_disparities, c_disparities, min_disp = 0, max_disp = 100):
        
        def get_ind(kps, disp):
            pt_ind = kps.astype(int)
            disp = disp.T[pt_ind[:,0], pt_ind[:,1]]
            msk_in_bound = np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
            
            return disp, msk_in_bound
        
        #compute left feature point
        p_disp_l, msk1 = get_ind(p_trackkpts_l, p_disparities)
        c_disp_l, msk2 = get_ind(c_trackkpts_l, c_disparities)
        
        msk = np.logical_and(msk1, msk2)

        p_fpt_l, p_disp_l = p_trackkpts_l[msk], p_disp_l[msk]
        c_fpt_l, c_disp_l = c_trackkpts_l[msk], c_disp_l[msk]
    
        #compute right feature point
        p_fpt_r, c_fpt_r = np.copy(p_fpt_l), np.copy(c_fpt_l)
        p_fpt_r[:, 0] -= p_disp_l
        c_fpt_r[:, 0] -= c_disp_l
        
        return p_fpt_l, c_fpt_l, p_fpt_r, c_fpt_r
    
    def compute_3d(self, p_kp_l, p_kp_r, c_kp_l, c_kp_r):
        
        #triangulate points of i-1 th image
        #q1 shape = (4, N)
        q1 = cv2.triangulatePoints(self.P_l, self.P_r, p_kp_l.T, p_kp_r.T)
        #normalise and extract xyz data
        p_xyz = np.transpose(q1[:3]/q1[3])
        
        #triangulate points of i th image
        #q2 shape = (4, N)
        q2 = cv2.triangulatePoints(self.P_l, self.P_r, c_kp_l.T, c_kp_r.T)
        #normalise and extract xyz data
        c_xyz = np.transpose(q2[:3]/q2[3])
    
        return p_xyz, c_xyz
    
    def reproject(self, dof, p_fkp, c_fkp, p_xyz, c_xyz):
        
        '''
        purpose: optimise residuals by estimating T_mat
        '''
        
        #get rotation vector
        r = dof[:3]
        
        #get rotation matrix
        R, _ = cv2.Rodrigues(r)
        
        #get translation vector
        t = dof[3:]
        
        #get transfomation matrix
        T_mat = self._get_Tmat(R, t)
        
        #compute forward projection matrix 
        f_proj_mat = np.matmul(self.P_l, T_mat)
        
        #compute backward projection matrix
        b_proj_mat = np.matmul(self.P_l, np.linalg.inv(T_mat))
        
        ones = np.ones((p_fkp.shape[0], 1))
        p_xyz = np.hstack([p_xyz, ones])
        c_xyz = np.hstack([c_xyz, ones])
        
        #project 3d points from i-th image to (i-1)-th image
        pred_p_fkp = c_xyz.dot(f_proj_mat.T)
        pred_p_fkp = pred_p_fkp[:, :2].T/pred_p_fkp[:, 2]
  
        #project 3d points from (i-1)-th image to i-th image
        pred_c_fkp = p_xyz.dot(b_proj_mat.T)
        pred_c_fkp = pred_c_fkp[:, :2].T/pred_c_fkp[:, 2]
        
        res = np.vstack([pred_p_fkp - p_fkp.T, pred_c_fkp - c_fkp.T])
        res = res.flatten()

        return res
        
    def est_pose(self, p_fkp, c_fkp, p_xyz, c_xyz, max_iter = 100):
        
        min_err = np.inf
        early_stop_threshold = 20
        early_stop_cnt = 0

        for _ in range(max_iter):
            
            #ocassionally, # of feature points < 6
            try: 
                sample_ind = np.random.choice(range(p_fkp.shape[0]), 6, replace = False)
            except:
                sample_ind = np.random.choice(range(p_fkp.shape[0]), 6)
            sample_p_fkp, sample_c_fkp = p_fkp[sample_ind], c_fkp[sample_ind]
            sample_p_xyz, sample_c_xyz = p_xyz[sample_ind], c_xyz[sample_ind]
            
            init_guess = np.zeros(6)
            
            opt_res = least_squares(self.reproject, 
                                    init_guess, 
                                    method = 'lm', 
                                    max_nfev = 200,
                                    args = (sample_p_fkp, 
                                            sample_c_fkp, 
                                            sample_p_xyz, 
                                            sample_c_xyz))
            
            err = self.reproject(opt_res.x, p_fkp, c_fkp, p_xyz, c_xyz)

            err = err.reshape((4, p_fkp.shape[0]))
            err1 = err[0:2, :]
            err2 = err[2: , :]
            
            err1_norm = np.linalg.norm(err1, axis = 0)
            err2_norm = np.linalg.norm(err2, axis = 0)
            
            err_mean = np.mean(err1_norm) + np.mean(err2_norm)

            if err_mean < min_err:
                min_err = err_mean
                out_pose = opt_res.x
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            
            if early_stop_cnt == early_stop_threshold:
                break

        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        
        T_mat = self._get_Tmat(R, t)
        
        return T_mat
        
    def get_relative_pose(self, c_ind):
        
        p_img_l, c_img_l = self.imgs_l[c_ind - 1: c_ind + 1]  
        
        #get tiled keypoints
        p_img_l_kps = self.get_tiled_kps(p_img_l, 10, 20)
        
        #track key points based on left images (i - 1 th index vs i th index)
        p_trackkpts_l, c_trackkpts_l = self.track_kpts(p_img_l, c_img_l, p_img_l_kps, max_err = 5)

        #compute disparities (left vs right)
        c_img_r = self.imgs_r[c_ind]
        self.disparities.append(np.divide(self.disparity.compute(c_img_l, c_img_r).astype(np.float32), 16))
        
        #get right image keypoints
        p_fpt_l, c_fpt_l, p_fpt_r, c_fpt_r = self.compute_feature_kpts(p_trackkpts_l, c_trackkpts_l, self.disparities[-2], self.disparities[-1])
        
        #compute 3d points 
        p_xyz, c_xyz = self.compute_3d(p_fpt_l, p_fpt_r, c_fpt_l, c_fpt_r)
        
        #estimate pose
        T_mat = self.est_pose(p_fpt_l, c_fpt_l, p_xyz, c_xyz)
        
        return T_mat
