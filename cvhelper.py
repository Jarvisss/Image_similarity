import cv2
import numpy as np
from PyQt5.QtGui import QImage

def cv_getSmoothImg(img):
    ignoreMap = cv2.GaussianBlur(img, (0, 0), 2)
    return ignoreMap

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

# there should be only one contour, if segmentation was correctly done
# if there are more, caused by noise, take the longest one
def cv_getLongestContour(img, thres=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
    im, ct, hi = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contour_points = max(ct, key=len)
    # remove second dim
    # contour_points = np.reshape(contour_points, (len(contour_points), 2))
    return contour_points

def cv_drawContourImg(img, thickness=2):
    counter_points = cv_getLongestContour(img)
    contourImg = cv2.drawContours(img, [counter_points], 0, (0,0,255),thickness=thickness)

    return contourImg

def cv_toQimage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    q_img = QImage(cv_img.data, width
                   , height, bytesPerLine, QImage.Format_RGB888)
    return q_img

def cv_toBinary(img,do_smooth,threshold=128,):
  smooth_img = cv_getSmoothImg(img) if do_smooth else img
  gray = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2GRAY)
  _, bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
  return bin


if __name__ == '__main__':
    img = cv_imread('./轮廓图/两栖纲/青蛙/青蛙-侧面.jpg')
    sm_img = cv_getSmoothImg(img)
    contour_points = cv_getLongestContour(sm_img)
    img = cv2.drawContours(img, [contour_points], -1, (0,0,255), thickness=1)
    cv2.imshow('1',img)
    cv2.waitKey()
