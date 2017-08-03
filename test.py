import time
import random
import utils
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab
import skimage
import skimage.io
import skimage.transform

f = open("batch_losses.txt", mode='r')
lines = f.readlines()

plt.figure()
plt.plot(range(lines.__len__()), [float(line) for line in lines], '--', label='train loss', )
plt.grid(True)
plt.xlabel('time')
plt.ylabel('acc-loss')
plt.legend(loc="upper right")
plt.show()

str = '[num for num in range(10)]'
print eval(str)