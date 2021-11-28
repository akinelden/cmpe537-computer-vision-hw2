import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import interpolate
import image_stitch

# Experiments for north campus

path = "north_campus/left-1.jpg"
left1 = Image.open(path)
path = "north_campus/left-2.jpg"
left2 = Image.open(path)
path = "north_campus/middle.jpg"
middle = Image.open(path)
path = "north_campus/right-1.jpg"
right1 = Image.open(path)
path = "north_campus/right-2.jpg"
right2 = Image.open(path)

images = [left1, middle, right1]

pts = image_stitch.load_pairs_from_csv("data/nc-1.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("5 Correspondence Points with 3 Images")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/nc-2.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 Correspondence Points with 3 Images")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/nc-3.csv")
stitched = image_stitch.stitch_images(images, pts, False)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 3 wrong matches without normalization step")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/nc-3.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 3 wrong matches")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/nc-5.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 5 wrong matches")
plt.imshow(stitched)
plt.show()

images = [left2, left1, middle, right1, right2]
pts = image_stitch.load_pairs_from_csv("data/nc-7.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("Stitch all")
plt.imshow(stitched)
plt.show()

# ----------------------------------------------
# Experiments for CMPE building

path = "cmpe-building/left-2.jpg"
left2 = Image.open(path)
path = "cmpe-building/left-1.jpg"
left1 = Image.open(path)
path = "cmpe-building/middle.jpg"
middle = Image.open(path)
path = "cmpe-building/right-1.jpg"
right1 = Image.open(path)
path = "cmpe-building/right-2.jpg"
right2 = Image.open(path)

images = [left1, middle, right1]

pts = image_stitch.load_pairs_from_csv("data/cb-1.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("5 Correspondence Points with 3 Images")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/cb-2.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 Correspondence Points with 3 Images")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/cb-3.csv")
stitched = image_stitch.stitch_images(images, pts, False)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 3 wrong matches without normalization step")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/cb-3.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 3 wrong matches")
plt.imshow(stitched)
plt.show()

pts = image_stitch.load_pairs_from_csv("data/cb-5.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("12 correspondence points with 5 wrong matches")
plt.imshow(stitched)
plt.show()

images = [left2, left1, middle, right1, right2]
pts = image_stitch.load_pairs_from_csv("data/cb-7.csv")
stitched = image_stitch.stitch_images(images, pts)
plt.figure(figsize=(15,30))
plt.title("Stitch all")
plt.imshow(stitched)
plt.show()