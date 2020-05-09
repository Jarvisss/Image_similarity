import os
import measure_img_sim
from idsc import IDSC
from cvhelper import cv_toBinary,cv_imread,cv_rotate
import numpy as np
from scipy.spatial.distance import euclidean
import cv2
import sys
from PyQt5.QtCore import pyqtSignal, QThread


def getImagesInDir(root_path):
    files = os.walk(root_path)
    img_paths = []
    for path, d, filelist in files:
        for filename in filelist:
            if filename.endswith('jpg') or filename.endswith('png'):
                img_paths.append(os.path.join(path, filename))

    return img_paths



class ImageDB(QThread):
    step_signal = pyqtSignal(int)
    done_signal = pyqtSignal(list)
    def __init__(self, img_db_root):
        super(ImageDB, self).__init__()
        self.img_paths = getImagesInDir(img_db_root)
        self.method = 'scd'
        self.query_path = ''
        self.sample_rate = 100
        self.do_sample = True
        self.do_resize = True
        self.do_smooth = False
        self.rotation_invariant = False
        self.visualize = True
        self.save_countours = True
        self.resize_w = 2**9
        self.resize_h = 2**9
        self.iterations = 3
        self.rot_angle = 30
        self.img_query_group = []
        self.counter_query_group = []
        self.sim_dict = {}
        pass

    def _sample_contour_points(self, contour_points, n):
        # remove second dim
        # contour_points = np.reshape(contour_points, (len(contour_points), 2))
        # sample n points
        idx = np.linspace(0, len(contour_points) - 1, num=n).astype(np.int)
        if len(contour_points) < len(idx):
            return contour_points
        else:
            return contour_points[idx]

    def set_method(self, method):
        self.method = method

    def set_query_path(self, path):
        self.query_path = path

    def set_sample_rate(self, n):
        self.sample_rate = n

    def set_angle_step(self, angle):
        self.rot_angle = angle


    """
    run(query_path, method) -> retval
    .   @brief get image similarities of query image and images in the database, by the given method type
    .   
    .   @param req_path Input array or vector of bytes.
    .   @param method Which method to use for similarity compare
    """
    def run(self):

        if self.query_path == '':
            return
        sim_dict = {}
        sim_dict_flipped = {}
        img = cv_imread(self.query_path)
        resized_q = cv2.resize(img, (self.resize_w, self.resize_h)) if self.do_resize else img
        resized_qbinary = cv_toBinary(resized_q, self.do_smooth)
        self.img_query_group.clear()
        self.counter_query_group.clear()
        for i in range(360//self.rot_angle):
            rot = cv_rotate(img, i*self.rot_angle)
            rot_flip = cv_rotate(cv2.flip(img,1), i*self.rot_angle)
            resized_q = cv2.resize(rot, (self.resize_w, self.resize_h)) if self.do_resize else rot
            resized_qbinary = cv_toBinary(resized_q, self.do_smooth)

            resized_q_flip = cv2.resize(rot_flip, (self.resize_w, self.resize_h)) if self.do_resize else rot_flip
            resized_qbinary_flip = cv_toBinary(resized_q_flip, self.do_smooth)

            c0, _ = cv2.findContours(resized_qbinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cq = self._sample_contour_points(max(c0, key=len), self.sample_rate) if self.do_sample else max(c0, key=len)
            c0_flip, _ = cv2.findContours(resized_qbinary_flip, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cq_flip = self._sample_contour_points(max(c0_flip, key=len), self.sample_rate) if self.do_sample else max(c0, key=len)
            self.img_query_group.append(rot)
            self.img_query_group.append(rot_flip)
            self.counter_query_group.append(cq)
            self.counter_query_group.append(cq_flip)


        if self.method == 'hu':
            for i,image_path in enumerate(self.img_paths):
                self.step_signal.emit(i + 1)
                sim = measure_img_sim.hu_moment_sim(self.query_path, image_path)
                sim_dict[image_path] = sim
                sorted_list = sorted(sim_dict.items(), key=lambda x: x[1])

        elif self.method == 'emd':
            for image_path in self.img_paths:
                sim = measure_img_sim.earth_movers_distance(self.query_path, image_path)
                sim_dict[image_path] = sim
                sorted_list = sorted(sim_dict.items(), key=lambda x: x[1])
        elif self.method == 'ssim':
            for image_path in self.img_paths:
                sim = measure_img_sim.structural_sim(self.query_path, image_path)
                sim_dict[image_path] = sim
                sorted_list = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            '''
            Shape based methods
            '''
            if self.method == 'idsc':
                dist = euclidean
                idsc = IDSC()
                query_descriptor = idsc.describe(resized_qbinary)
                for image_path in self.img_paths:
                    print('comparing'+image_path)
                    npy_path = image_path.replace('.jpg','.npy')
                    if not os.path.exists(npy_path):
                        img_descriptor = idsc.describe(cv_toBinary(image_path),self.do_smooth)
                        np.save(npy_path, img_descriptor)
                    else:
                        img_descriptor = np.load(npy_path)

                    sim = dist(query_descriptor.flat, img_descriptor.flat)
                    sim_dict[image_path] = sim
                    sorted_list = sorted(sim_dict.items(), key=lambda x: x[1])

            elif self.method=='scd':
                scd = cv2.createShapeContextDistanceExtractor()
                scd.setRotationInvariant(self.rotation_invariant)
                for i, image_path in enumerate(self.img_paths):
                    print(image_path,end =" ")
                    self.step_signal.emit(i+1)
                    resized_mbinary = cv_toBinary(cv2.resize(cv_imread(image_path), (self.resize_w,self.resize_h)) if self.do_resize else cv_imread(image_path),self.do_smooth)
                    blank = 255 - np.zeros(resized_mbinary.shape,np.uint8)
                    c1, _ = cv2.findContours(resized_mbinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                    cm = self._sample_contour_points(max(c1, key=len), self.sample_rate) if self.do_sample else max(c1, key=len)
                    if self.visualize:
                        resized_mbinary = cv2.drawContours(cv2.cvtColor(resized_mbinary, cv2.COLOR_GRAY2BGR), [cm], 0, (0,0,255),thickness=1)
                        cv2.imshow('aa', resized_mbinary)
                        cv2.waitKey(1)
                    if self.save_countours:
                        print(cm.shape)
                        save_path = image_path.replace('\\', '/')
                        save_path = save_path.replace('轮廓图', 'counters')
                        name = save_path.split('/')[-1]
                        save_dir = save_path.replace(name, '')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        for i in cm:
                            cv2.circle(blank, (i[0][0],i[0][1]), radius=1, color=(0,0,0), thickness=-1)
                            # blank[int(i[0][1]), int(i[0][0])] = 0
                        cv2.imencode('.jpg', blank)[1].tofile(save_path)

                    print(len(cm), end=" ")
                    scd.setIterations(self.iterations)
                    min_dist = sys.maxsize
                    flipped = False
                    rot_angle = 0
                    for i, cq in enumerate(self.counter_query_group):
                        dist = scd.computeDistance(cq,cm)
                        if dist < min_dist:
                            min_dist = dist
                            flipped = i % 2==1
                            rot_angle = i//2*self.rot_angle

                    print(dist, "flip:", flipped, "rot_angle:", rot_angle)
                    sim_dict[image_path] = (min_dist,flipped, rot_angle)
                    sorted_list = sorted(sim_dict.items(), key=lambda x: x[1][0])

            elif self.method == 'hausdorff':
                hsd = cv2.createHausdorffDistanceExtractor()
                for i,image_path in enumerate(self.img_paths):
                    print(image_path)
                    self.step_signal.emit(i+1)
                    resized_mbinary = cv_toBinary(cv2.resize(cv_imread(image_path), (self.resize_w,self.resize_h)) if self.do_resize else cv_imread(image_path),self.do_smooth)
                    c1, _ = cv2.findContours(resized_mbinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                    cm = self._sample_contour_points(max(c1, key=len),self.sample_rate) if self.do_sample else max(c1, key=len)
                    sim = hsd.computeDistance(cq,cm)
                    sim_dict[image_path] = sim
                    sorted_list = sorted(sim_dict.items(), key=lambda x: x[1])

        self.done_signal.emit(sorted_list)
        self.quit()



if __name__ == '__main__':
    imdb = ImageDB('动物图片轮廓')
    sorted = imdb.cal_similarities('./动物图片轮廓/两栖纲/蛇/蛇-背面.jpg')
    print(sorted)
