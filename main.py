import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import interpolate
import image_stitch

path = "north_campus/left-1.jpg"
left1 = Image.open(path)
path = "north_campus/left-2.jpg"
left2 = Image.open(path)

coords1 = np.array([[1018.50277818,  421.4985378 ],
       [ 896.37966077,  389.91497295],
       [1208.00416728,   36.17904664],
       [ 700.56155871,  156.19659307],
       [1182.7373154 ,  427.81525077],
       [ 847.951528  ,   42.49575961]])

coords2 =np.array([[340.50891943, 426.42557391],
       [249.96936687, 399.05315105],
       [534.2214505 ,  70.58407662],
       [ 24.67327095, 137.9623483 ],
       [487.89888873, 424.32000292],
       [193.11895014,  34.78936979]])

coords1 = coords1[:, [1,0]]
coords2 = coords2[:, [1,0]]

H = image_stitch.computeH(coords1, coords2, False)
warped_img, x_offset, y_offset = image_stitch.warp(np.array(left2), H)
plt.imshow(warped_img)