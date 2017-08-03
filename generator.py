# coding=utf-8
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage
import skimage.io
import skimage.transform

import string
characters = "0123456789+-*()"
print(characters)
width, height, n_len, n_class = 170, 80, 4, len(characters)+1

path = "/home/wjg/datasets/image_contest_generated/"

generator = ImageCaptcha(width=width, height=height)
f = open(path + "labels.txt", "w")
labels = []
for i in range(100000):
    random_str = ''.join([random.choice(characters) for j in range(5)])
    img = np.array(generator.generate_image(random_str))
    skimage.io.imsave(path + str(i) + ".png", img)
    f.write(random_str + "\n")