
import cv2
import numpy as np
import glob

#read has watter images
imgs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('./Images/HasWater/*.jpg')]

scaled = [cv2.resize(im, (258,258)) for im in imgs]

sizes = np.array([im.shape[0] for im in imgs])
med = np.median(sizes)
avg = np.mean(sizes)
std = np.std(sizes)
maxi = np.max(sizes)
mini = np.min(sizes)

print(
    f"""
median = {med}
mean = {avg}
std = {std}
max = {maxi}
min = {mini}
""")

