from scipy.sparse.csgraph import floyd_warshall
from skimage.draw import line as skline
import numpy as np
import cv2
import scipy as sp, scipy.spatial
from cvhelper import cv_toBinary


class IDSC():

  def __init__(self, n_contour_points=300, n_angle_bins=8,
               n_distance_bins=8, distance_func=sp.spatial.distance.euclidean, shortest_path_func=floyd_warshall):
    label = 'Inner Distance Shape Context'
    # n_levels = 1
    # binary_input = True
    self.n_contour_points = n_contour_points
    self.n_angle_bins = n_angle_bins
    self.n_distance_bins = n_distance_bins
    self.distance = distance_func
    self.shortest_path = shortest_path_func

  def describe(self, binary, visualize=False):
    # print('get descriptor')
    # maximum distance is the from upper left to lower right pixel,
    # so all points lie within distance
    self.max_distance = self.distance((0, 0), binary.shape)
    contour_points = self._sample_contour_points(binary,
                                                 self.n_contour_points)
    if visualize:
      img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
      for p in contour_points:
        img = cv2.circle(img, tuple(p), 1, thickness=2,
                         color=(0, 255, 0))
      cv2.imshow('contour',img)
      cv2.waitKey()

    if len(contour_points) == 0:
      print('contours missing in IDSC')
      return np.zeros(self.n_contour_points)
    dist_matrix = self._build_distance_matrix(binary, contour_points)
    context = self._build_shape_context(dist_matrix, contour_points)

    ### Visualisation ###

    # if steps is not None:

    # steps['picked points'] = img

    return context

  def _get_points_on_line(self, p1, p2):
    x, y = skline(p1[0], p1[1], p2[0], p2[1])
    return x, y

  def _sample_contour_points(self,binary, n):
    im, ct, hi = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = max(ct, key=len)
    # remove second dim
    contour_points = np.reshape(contour_points, (len(contour_points), 2))
    # sample n points
    idx = np.linspace(0, len(contour_points) - 1, num=n).astype(np.int)
    return contour_points[idx]

  '''
  Inner Distance matrix
  '''
  def _build_distance_matrix(self, binary, contour_points):
    dist_matrix = np.zeros((len(contour_points), len(contour_points)))

    # fill the distance matrix pairwise
    for i, p1 in enumerate(contour_points):
      for j, p2 in enumerate(contour_points[i + 1:]):
        lx, ly = self._get_points_on_line(p1, p2)
        values = binary[ly, lx]
        inside_shape = np.count_nonzero(values) == len(values)
        if not inside_shape:
          continue
        # if all points on line are within shape -> calculate distance
        dist = self.distance(p1, p2)
        if dist > self.max_distance:
          break
        # store distance in matrix (mirrored)
        dist_matrix[j + i, i] = dist
        dist_matrix[i, j + i] = dist

    return dist_matrix

  '''
  Basic SC Algorithm
  '''
  def _build_shape_context(self, distance_matrix, contour_points, skip_distant_points=False):
    histogram = []
    max_log_distance = np.log2(self.max_distance)
    # steps between assigned bins
    dist_step = max_log_distance / self.n_distance_bins
    angle_step = np.pi * 2 / self.n_angle_bins
    # find shortest paths in distance matrix (distances as weights)
    graph = self.shortest_path(distance_matrix, directed=False)

    # iterate all points on contour
    for i, (x0, y0) in enumerate(contour_points):
      hist = np.zeros((self.n_angle_bins, self.n_distance_bins))

      # calc. contour tangent from previous to next point
      # to determine angles to all other contour points
      (prev_x, prev_y) = contour_points[i - 1]
      (next_x, next_y) = contour_points[(i + 1) % len(contour_points)]
      tangent = np.arctan2(next_y - prev_y,
                           next_x - prev_x)

      # inspect relationship to all other points (except itself)
      # direction and distance are logarithmic partitioned into n bins
      for j, (x1, y1) in enumerate(contour_points):
        if j == i:
          continue
        dist = graph[i, j]
        # 0 or infinity determine, that there is no path to point
        if dist != 0 and dist != np.inf:
          log_dist = np.log2(dist)
        # ignore unreachable points, if requested
        elif skip_distant_points:
          continue
        # else unreachable point is put in last dist. bin
        else:
          log_dist = max_log_distance
        angle = (tangent - np.arctan2(y1 - y0, x1 - x0)) % (2 * np.pi)
        # calculate bins, the inspected point belongs to
        dist_idx = int(min(np.floor(log_dist / dist_step),
                           self.n_distance_bins - 1))
        angle_idx = int(min(angle / angle_step,
                            self.n_angle_bins - 1))
        # point fits into bin
        hist[angle_idx, dist_idx] += 1

      # L1 norm
      if hist.sum() > 0:
        hist = hist / hist.sum()
      histogram.append(hist.flatten())

    return np.array(histogram).T



if __name__ == '__main__':
  dist = sp.spatial.distance.euclidean

  # img_a = './动物图片轮廓/两栖纲/蛇/蛇-背面.jpg'
  # img_b = './动物图片轮廓/两栖纲/蛇/蛇-正面.jpg'
  contour_img_path = './轮廓图/两栖纲/青蛙/青蛙-背面.jpg'
  contour_img_path_2 = './轮廓图/两栖纲/青蛙/青蛙-侧面.jpg'
  contour_img_path_3 = './轮廓图/两栖纲/青蛙/青蛙-背面.jpg'
  idsc = IDSC()


  bin = cv_toBinary(contour_img_path)
  histo = idsc.describe(bin, True)

  bin2 = cv_toBinary(contour_img_path_2)
  histo2 = idsc.describe(bin2, True)

  distance = dist(histo.flat, histo2.flat)
  print(distance)