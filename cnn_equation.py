from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model as plot
from IPython.display import Image
import utils
from tqdm import tqdm
import time

import string

height, width, n_len, n_class = 80, 170, 8, 16
dir_path = "/home/wjg/datasets/image_contest_level_1/"
labels = utils.load_labels(dir_path + "labels.txt")

origin_population = range(100000)


def gen_randomint():
    sampled_population = random.sample(origin_population, origin_population.__len__())
    while sampled_population.__len__() > 0:
        yield sampled_population.pop()

randomint_generator = gen_randomint()

def gen(batch_size=100):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]

    while True:
        for i in range(batch_size):
            num = next(randomint_generator)
            X[i] = utils.load_image(dir_path + str(num) + ".png")
            for j in range(n_len):
                y[j][i, :] = 0
                y[j][i, labels[num][j]] = 1
        yield X, y


input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(input=input_tensor, output=x)

try:
    model.load_weights('cnn_equation.h5')
    print 'Use pretrained model.'
except:
    print 'Model file not found or corrupted, create a new one.'
finally:
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    plot(model, to_file="model_equation.png", show_shapes=True)
    Image('model_equation.png')


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
    return batch_acc / batch_num

accuracy = 0.0

while accuracy < 0.999:
    model.fit_generator(gen(), samples_per_epoch=900, nb_epoch=1,
                        nb_worker=2, pickle_safe=True,
                        validation_data=gen(), nb_val_samples=100)
    accuracy = evaluate(model)
    print "accuracy =", accuracy
    f = open("accuracy_equation.txt", "a")
    f.writelines(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + "\t" + str(accuracy) + "\n")
    f.close()
    model.save('cnn_equation.h5')
    randomint_generator = gen_randomint()

    time.sleep(300)     # have a rest!