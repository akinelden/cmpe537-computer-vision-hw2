import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy

def stitch_images(images, n_pairs = 10, normalize=True):
  homographies = []
  prev_img = None
  for img in images:
    if prev_img is None:
      prev_img = img
      continue
    # coord1 = select_points(prev_img, n_pairs)
    # coord2 = select_points(img, n_pairs)
    
    # select pairs from images
    coords1, coords2 = select_pairs(prev_img, img, n_pairs)
    # compute Homography
    H = computeH(coords1, coords2, normalize)
    homographies.append(H)
    warped = warp(prev_img, H)
  
# def select_points(img, n):
#   plt.imshow(img)
#   coords = plt.ginput(n,show_clicks=True)
#   plt.show()
#   return np.array(coords)
  
def select_pairs(img1, img2, n):
  print("Please select pairs one by one.")
  plt.subplot(211)
  plt.imshow(img1)
  plt.subplot(212)
  plt.imshow(img2)
  coords = plt.ginput(2*n,show_clicks=True, timeout=600)
  # odd items of coords belong to img1, event ones belong to img2
  coords1 = np.array(coords[0::2])
  coords2 = np.array(coords[1::2])
  # swap columns since matplotlib x axis refers to columns and y axis refers to rows in image matrix
  return (coords1[:,[1,0]], coords2[:,[1,0]])

def computeH(im1Points, im2Points, normalize=True):
  Points1, Points2 = im1Points, im2Points
  T1, T2 = None, None
  if normalize:
    Points1, T1 = normalize_points(im1Points)
    Points2, T2 = normalize_points(im2Points)
  A = get_matrix_A(Points1, Points2)
  _U, _S, VT = scipy.linalg.svd(A)
  H = VT.T[:, -1].reshape(3,3)
  # H_ = Vh[:, -1].reshape(3,3).T
  if normalize:
    H = np.linalg.inv(T2) @ H @ T1
  return H

def warp(image, H):
  img_arr = np.array(image)
  (xmin, ymin, xmax, ymax) = get_borders(img_arr, H)
  mgrid = np.mgrid[xmin:xmax, ymin:ymax]
  xs = mgrid[0].reshape(-1)
  # xs = np.tile(np.arange(xmin,xmax), ymax-ymin)
  # ys = np.repeat(np.arange(ymin, ymax), xmax-xmin)
  ys = mgrid[1].reshape(-1)
  back_transformed = np.linalg.inv(H) @ np.array([xs, ys, np.ones(len(xs))]) # 3xN
  R, G, B = interpolate_points(back_transformed[:2,:], img_arr)
  warped = np.stack([R.reshape(xmax-xmin,-1), G.reshape(xmax-xmin,-1), B.reshape(xmax-xmin,-1)], 2)
  return warped

def get_borders(image, H):
  corners = np.c_[(0,0,1),(0,image.shape[1],1),(image.shape[0], 0,1),(image.shape[0],image.shape[1],1)]
  temp_xyz = np.array(H @ corners)
  temp_xy = np.array(temp_xyz[:2] / temp_xyz[2], ddtype=np.int32)
  min_pts = temp_xy.min(1)
  max_pts = temp_xy.max(1)
  return (min_pts[0], min_pts[1], max_pts[0], max_pts[1])

def interpolate_points(transformed_pts, img):
  # transformed_pts : 2xN np array
  # img : 3d np.array of image
  rgb = []
  xs = np.arange(img.shape[0])
  ys = np.arange(img.shape[1])
  txs = transformed_pts[0]
  tys = transformed_pts[1]
  for i in range(3):
    zs = img[:,:,i].reshape(-1)
    intp_f = interpolate.interp2d(xs, ys, zs, kind="cubic", fill_value=0)
    rgb.append(intp_f(txs[0:1000], tys[0:1000]))
  return rgb

def normalize_points(points):
  n_pts = points.shape[0]
  mean_pt = points.mean(0)
  squared_euclidean = np.linalg.norm(points-mean_pt,2,1)**2
  s = np.sqrt(2*n_pts/squared_euclidean.sum())
  T = np.array([[s,0,-s*mean_pt[0]],
                [0,s,-s*mean_pt[1]],
                [0,0,1]])
  return (np.c_[points, np.ones(n_pts)] @ T.T)[:,:2], T

def get_matrix_A(points1, points2):
  rows = []
  for (x1,y1), (x2,y2) in zip(points1, points2):
    rows.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
    rows.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
  return np.array(rows)

