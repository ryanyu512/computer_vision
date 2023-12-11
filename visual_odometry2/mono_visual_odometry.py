import os
import cv2
import numpy as np

class VisualOdometry():
    
    def __init__(self, data_dir):
        #K = intrinsic matrix
        #P = intrinsic matrix with extra zero column
        self.K, self.P = self._load_cam_params(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.imgs = self._load_images(os.path.join(data_dir, 'image_l'))
        
        self.orb = cv2.ORB_create(3000)
        
        #initialise FLANN
        FLANN_INDEX_LSH = 6
        
        index_params = dict(algorithm = FLANN_INDEX_LSH, 
                            table_number = 6,
                            key_size = 12, 
                            multi_probe_level = 1)
        
        search_params = dict(checks = 50)
        
        self.flann = cv2.FlannBasedMatcher(indexParams = index_params, 
                                           searchParams = search_params)
        
    @staticmethod
    def _load_cam_params(file_path):
        
        with open(file_path, 'r') as f:
            str_cam_params = np.fromstring(f.readline(), dtype = np.float64, sep = ' ')
            P = np.reshape(str_cam_params, (3, 4))
            K = P[0:3, 0:3]
            
        return K, P
    
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
    
    def get_matches(self, c_ind, is_draw = True):
        
        p_ind = c_ind - 1
        
        #obtain descriptors and keypoints
        k1, d1 = self.orb.detectAndCompute(self.imgs[p_ind], None)
        k2, d2 = self.orb.detectAndCompute(self.imgs[c_ind], None)

        #retain best two matches
        matches = self.flann.knnMatch(d1, d2, k = 2)
        
        #filter out bad matches
        good_matches = []
        try:
            '''
            1. we keep two best two matches
            2. if m1.distance ~= m2.distance => two matches equally good => we don't know how to choose
            3. however, if m1.distance < 0.7 * m2.distance => we may guess that m1 match is much better
            '''
            for m1, m2 in matches:
                if m1.distance < 0.5 * m2.distance:
                    good_matches.append(m1)
        except:
            pass
        
        #draw match
        if is_draw:
            draw_params = dict(matchColor = -1,
                               singlePointColor = None, 
                               matchesMask = None, 
                               flags = 2)
            
            d_img = cv2.drawMatches(self.imgs[c_ind], 
                                    k1,
                                    self.imgs[p_ind], 
                                    k2,
                                    good_matches, 
                                    None, 
                                    **draw_params)
            
            cv2.imshow("img", d_img)
            cv2.waitKey(10)
        
        #q1 = key points from i - 1th image
        #q2 = key points from ith image
        q1 = np.float32([k1[gm.queryIdx].pt for gm in good_matches])
        q2 = np.float32([k2[gm.trainIdx].pt for gm in good_matches])
        
        return q1, q2
        
    def compute_pose(self, q1, q2):
        
        #compute essential matrix
        E, _ = cv2.findEssentialMat(q1, 
                                    q2, 
                                    self.K, 
                                    threshold = 1)
        
        #decompose essential matrix into R and t
        R, t = self.recover_pose(E, q1, q2)
        
        #get transformation matrix
        T_mat = self._get_Tmat(R, np.squeeze(t))
        
        return T_mat
    
    def recover_pose(self, E, q1, q2):
        
        def sum_z_cal_relative_scale(R, t):
            
            #get extrinsic matrix
            T_mat = self._get_Tmat(R, t)
            
            #get projection matrix = intrinsic matrix * extrinsic matrix
            P_mat = np.matmul(self.P, T_mat)

            #3d points expressed in homogenous form
            hom_q1 = cv2.triangulatePoints(self.P, P_mat, q1.T, q2.T)
            #transform 3d points relative to {1} => relative to {2}
            hom_q2 = np.matmul(T_mat, hom_q1)
            
            #normalise and extract 3d points
            xyz_q1 = hom_q1[:3, :]/hom_q1[3, :]
            xyz_q2 = hom_q2[:3, :]/hom_q1[3, :]
            
            #positive z implies the points are in front of the camera (make sense)
            #negative z implies the points are behind the camera (not make sense)
            
            sum_pos_z_q1 = np.sum(xyz_q1[2, :] > 0)
            sum_pos_z_q2 = np.sum(xyz_q2[2, :] > 0)
            sum_pos_z = sum_pos_z_q1 + sum_pos_z_q2
            
            #compute relative scale
            xyz_q1_norm = np.linalg.norm(xyz_q1.T[:-1] - xyz_q1.T[1:])
            xyz_q2_norm = np.linalg.norm(xyz_q2.T[:-1] - xyz_q2.T[1:])
            rel_scale = np.mean(xyz_q1_norm/xyz_q2_norm)

            return sum_pos_z, rel_scale
        
        #decompose essential matrix into rotation matrix and translation vector
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        
        z_list = []
        s_list = []
        for R, t in pairs:
            z_sum, s = sum_z_cal_relative_scale(R, t)
            z_list.append(z_sum)
            s_list.append(s)
            
        best_ind = np.argmax(z_list)
        R, t = pairs[best_ind]
        t = t * s_list[best_ind]
        
        return [R, t]   