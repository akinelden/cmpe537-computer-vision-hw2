import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import interpolate
import scipy

def select_pairs(images, n, file=""):
  """
  images: list of images in order
  n: number of point pairs
  """
  points = []
  for img1, img2 in zip(images[:len(images)-1], images[1:]):
    print("Please select pairs one from above and one from below image sequentially.")
    plt.title("Please select pairs one from above and one from below image sequentially.")
    plt.subplot(211)
    plt.imshow(img1)
    plt.subplot(212)
    plt.imshow(img2)
    pairs = plt.ginput(2*n,show_clicks=True, timeout=600)
    # odd items of coords belong to img1, event ones belong to img2
    coords1 = np.array(pairs[0::2])
    coords2 = np.array(pairs[1::2])
    # swap columns since matplotlib x axis refers to columns and y axis refers to rows in image matrix
    a = np.c_[coords1[:,[1,0]], coords2[:,[1,0]]]
    points.append(np.c_[coords1[:,[1,0]], coords2[:,[1,0]]])
  arr = np.array(points).reshape(-1,4)
  if file != "":
    np.savetxt(file, arr, delimiter=",")
  return arr

def load_pairs_from_csv(file):
  return np.loadtxt(open(file, "rb"), delimiter=",")

def stitch_images(images, pt_pairs, normalize=True):
  """
  images: List of images to stitch (left to right)
  n_pairs: Number of pairs to select for each image pair
  normalize: Whether to normalize data points
  pts_file: The name of the file in which to save point pairs. If empty string, points are not saved
  """
  if len(images) < 2:
    raise Exception("At least 2 images are required")
  med = int(len(images)/2)
  if pt_pairs.shape[0] % (len(images)-1) != 0:
    raise Exception("Format of the pt_pairs array is wrong. Equal number of correspondence points should be provided for each image pair")
  n_pairs = int(pt_pairs.shape[0] / (len(images)-1))
  # calculate homographies for left images
  left_homographies = []
  for i in reversed(range(med)):
    source = images[i]
    target = images[i+1]
    # points of these images are between [i*n_pairs, (i+1)*n_pairs] since points are from left to right
    pts = pt_pairs[i*n_pairs:(i+1)*n_pairs]
    H = computeH(pts[:,:2], pts[:,2:], normalize)
    # if this is not the first homography, multiply with last matrix to get homography Hn1 (Hn,n-1 @ Hn-1,1 = Hn,1)
    if (len(left_homographies) > 0):
      H = left_homographies[0] @ H
    left_homographies.insert(0,H)

  # calculate homographies for right images
  right_homographies = [] 
  for i in range(med+1,len(images)):
    # source is right, target is left image for right images
    source = images[i]
    target = images[i-1]
    # points of these images are between [(i-1)*n_pairs, i*n_pairs] since points are from right to left
    pts = pt_pairs[(i-1)*n_pairs:i*n_pairs]
    H = computeH(pts[:,2:], pts[:,:2], normalize)
    # if this is not the first homography, multiply with last matrix to get homography Hn1 (Hn,n-1 @ Hn-1,1 = Hn,1)
    if (len(right_homographies) > 0):
      H = right_homographies[-1] @ H
    right_homographies.append(H)
  
  warped_tuples = []
  print("Starting to warp left images")
  for img, H in zip(images[:med], left_homographies):
    warped_tuples.append(warp(img, H))
  warped_tuples.append((np.array(images[med]), 0, 0))
  print("Starting to warp right images")
  for img, H in zip(images[med+1:], right_homographies):
    warped_tuples.append(warp(img, H))
  
  print("Starting to stitch and blend images")
  stitched = None; rmax_offset = 0; cmax_offset = 0
  for right_img, r_offset, c_offset in warped_tuples:
    # if first image, initialize values
    if stitched is None:
      stitched = right_img
      rmax_offset = r_offset
      cmax_offset = c_offset
      continue
    stitched = stitch_and_blend(stitched, right_img, rmax_offset-r_offset, cmax_offset-c_offset)
    rmax_offset = max(rmax_offset, r_offset)
    # c_offset will always max at left most image
  print("Done")
  return stitched

def computeH(im1Points, im2Points, normalize=True):
  """
  im1Points: points of source image
  im2Points: points of target image
  normalize: whether to normalize
  """
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
  """
  image: Image as PIL or ndarray (it will convert to numpy array)
  H: homography matrix as ndarray
  """
  print("Warping..")
  img_arr = np.array(image)
  (xmin, ymin, xmax, ymax) = get_borders(img_arr, H)
  mgrid = np.mgrid[xmin:xmax, ymin:ymax]
  xs = mgrid[0].reshape(-1)
  ys = mgrid[1].reshape(-1)
  back_transformed = np.linalg.inv(H) @ np.array([xs, ys, np.ones(len(xs))]) # 3xN
  R, G, B = interpolate_points(back_transformed[:2]/back_transformed[2], img_arr)
  warped = np.stack([R.reshape(xmax-xmin,-1), G.reshape(xmax-xmin,-1), B.reshape(xmax-xmin,-1)], 2)
  warped = np.clip(np.around(warped), 0, 255).astype(np.uint8)
  return warped, -1*xmin, -1*ymin

def get_borders(image, H):
  corners = np.c_[(0,0,1),(0,image.shape[1],1),(image.shape[0], 0,1),(image.shape[0],image.shape[1],1)]
  temp_xyz = np.array(H @ corners)
  temp_xy = np.array(temp_xyz[:2] / temp_xyz[2], dtype=np.int32)
  min_pts = temp_xy.min(1)
  max_pts = temp_xy.max(1)
  return (min_pts[0], min_pts[1], max_pts[0], max_pts[1])

def interpolate_points(back_trans_pts, img):
  """
  back_trans_pts : 2xN np array
  img : 3d np.array of image
  """
  print("Interpolating..")
  rgb = []
  xs = np.arange(img.shape[0])
  ys = np.arange(img.shape[1])
  txs = back_trans_pts[0]
  tys = back_trans_pts[1]
  # bounds of the transformed image:
  outer_txs = (txs < 0) | (txs > img.shape[0])
  outer_tys = (tys < 0) | (tys > img.shape[1])
  for i in range(3):
    zs = img[:,:,i]
    spline = interpolate.RectBivariateSpline(xs, ys, zs)
    intp_values = spline.ev(txs, tys)
    intp_values[outer_txs] = 0
    intp_values[outer_tys] = 0
    rgb.append(intp_values)
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

def stitch_and_blend(left_img, right_img, row_offset, col_offset):
  """
  left_img: Left image as np.array
  right_img: Right image as np.array
  row_offset: difference between starting row of right and left images (can be negative if right image is higher than left)
  col_offset: difference between starting column of right and left images (must be nonnegative always since left image is at left)
  """
  print("Stitching images..")
  if col_offset < 0:
    raise Exception("col_offset cannot be negative")
  rmax = max(left_img.shape[0], row_offset + right_img.shape[0]) if row_offset > 0 else max(right_img.shape[0], left_img.shape[0] - row_offset)
  cmax = max(left_img.shape[1], col_offset + right_img.shape[1])
  stitched = np.zeros((rmax, cmax, 3), dtype=np.uint8)
  
  if row_offset > 0: # left image is higher
    lr1, lr2, lc1, lc2 = (0, left_img.shape[0], 0, left_img.shape[1])
    rr1, rr2, rc1, rc2 = (row_offset, row_offset+right_img.shape[0], col_offset, col_offset+right_img.shape[1])
  else: # right image is higher
    lr1, lr2, lc1, lc2 = (-row_offset, left_img.shape[0]-row_offset, 0, left_img.shape[1])
    rr1, rr2, rc1, rc2 = (0, right_img.shape[0], col_offset, col_offset+right_img.shape[1])

  # first put left image
  stitched[lr1:lr2, lc1:lc2] = left_img
  # compare left and right image intensities
  rgb_weight = np.array([0.299, 0.587, 0.114])
  gray_stitched = np.dot(stitched[rr1:rr2, rc1:rc2], rgb_weight)
  gray_right = np.dot(right_img, rgb_weight)
  filtered = gray_right > gray_stitched
  # set right image to only where its intensity is higher
  stitched[rr1:rr2, rc1:rc2][filtered] = right_img[filtered]
  return stitched