import warnings
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance

from imageio import imread
import numpy as np
import cv2
import time


from cvhelper import  cv_imread,cv_getSmoothImg,cv_getLongestContour

##
# Globals
##

warnings.filterwarnings('ignore')

# specify resized image sizes
height = 2**9
width = 2**9

##
# Functions
##

"""
get bbox of a gray scale image
"""
def get_bbox(img):
  up, bottom, left, right = 0, img.shape[0], 0, img.shape[1]
  for i in range(img.shape[0]):
    if np.min(img[i]) < 255:
      up = i
      break

  for i in range(img.shape[0]-1, 0, -1):
    if np.min(img[i]) < 255:
      bottom = i
      break

  for j in range(img.shape[1]):
    if np.min(img[:,j]) < 255:
      left = j
      break

  for j in range(img.shape[1]-1, 0, -1):
    if np.min(img[:,j]) < 255:
      right = j
      break

  return up,left,bottom,right

def get_img(path, norm_size=True, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # as_gray returns a 2d grayscale array
  img = imread(path, as_gray=True).astype(int)
  # resizing returns float vals 0:255; convert to ints for downstream tasks
  if norm_size:
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
  if norm_exposure:
    img = normalize_exposure(img)
  return img


def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)


def normalize_exposure(img):
  '''
  Normalize the exposure of an image.
  '''
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def earth_movers_distance(path_a, path_b):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)


def structural_sim(path_a, path_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a)
  img_b = get_img(path_b)
  sim, diff = compare_ssim(img_a, img_b, full=True)
  return sim


def pixel_sim(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  return np.sum(np.absolute(img_a - img_b)) / (height*width) / 255


def sift_sim(path_a, path_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # get the images
  img_a = cv2.imread(path_a)
  img_b = cv2.imread(path_b)

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


def hu_moment_sim(im_path1, im_path2, height=None):
  im1 = cv_imread(im_path1)
  im2 = cv_imread(im_path2)

  ct1 = cv_getLongestContour(im1)
  ct2 = cv_getLongestContour(im2)
  if height==None:
    height = im1.shape[0]
  im1 = cv2.resize(im1, (im1.shape[1] * height // im1.shape[0], height))
  im2 = cv2.resize(im2, (im2.shape[1] * height // im2.shape[0], height))

  gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


  # d1 = cv2.matchShapes(gray1, gray2, cv2.CONTOURS_MATCH_I1, 0)
  # d2 = cv2.matchShapes(gray1, gray2, cv2.CONTOURS_MATCH_I2, 0)
  # d3 = cv2.matchShapes(gray1, gray2, cv2.CONTOURS_MATCH_I3, 0)
  #
  # return (d1+d2+d3)/3
  d1 = cv2.matchShapes(ct1, ct2,cv2.CONTOURS_MATCH_I1, 0)
  return d1


if __name__ == '__main__':
  img_a = './动物图片轮廓/两栖纲/蛇/蛇-背面.jpg'
  img_b = './动物图片轮廓/两栖纲/蛇/蛇-正面.jpg'

  # get the similarity values
  start = time.time()
  structural_sim = structural_sim(img_a, img_b)
  end = time.time()
  print('ssim:',structural_sim, end-start)
  pixel_sim = pixel_sim(img_a, img_b)
  end2 = time.time()
  print('pixel_sim:',pixel_sim, end2-end)
  emd = earth_movers_distance(img_a, img_b)
  end3 = time.time()
  print('emd:',emd, end3-end2)