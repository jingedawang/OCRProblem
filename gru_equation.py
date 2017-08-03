from keras.models import *
from keras.layers import *
from keras import backend as K

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import utils

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 7, 16

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
             init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])

try:
    model.load_weights('gru_equation.h5')
    # model.load_weights('gru_equation-5rd-99.8%acc.h5')
    print 'load gru_equation.h5 successfully.'
except:
    print 'Model file gru_equation.h5 not found or corrupted, create a new one.'

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

dir_path = "/home/wjg/datasets/image_contest_level_1/"
# dir_path = "/home/wjg/datasets/image_contest_generated/"
labels, labels_length = utils.load_labels(dir_path + "labels.txt")
# labels, labels_length = utils.load_labels2(dir_path + "labels.txt")
origin_population = range(100000)

def sample_train_and_validation(origin_population):
    sampled_origin_population = random.sample(origin_population, origin_population.__len__())
    return sampled_origin_population[:90000], sampled_origin_population[90000:]

train_population, validation_population = sample_train_and_validation(origin_population)

def gen_randomint(isTrain=True):
    if isTrain:
        while train_population.__len__() > 0:
            yield train_population.pop()
    else:
        while validation_population.__len__() > 0:
            yield validation_population.pop()


def gen(batch_size=128, isTrain=True):
    randomint_generator = gen_randomint(isTrain)
    X = np.zeros((batch_size, width, height, 3), dtype=np.float32)
    y = np.zeros((batch_size, n_len), dtype=np.int8)
    label_length = np.zeros((batch_size, 1), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            num = next(randomint_generator)
            X[i] = utils.load_image(dir_path + str(num) + ".png").transpose(1, 0, 2)
            y[i] = labels[num]
            label_length[i][0] = labels_length[num]
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2),
               label_length], np.ones(batch_size)

def evaluate(model, batch_num=16):
    batch_acc = 0
    generator = gen(isTrain=False)
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        ctc_decode = K.ctc_decode(y_pred[:,2:,:],
                                  input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        if out.shape[1] == n_len:
            current_batch_acc = ((y_test == out).sum(axis=1) == n_len).mean()
            print "batch " + str(i) + ": " + str(current_batch_acc)
            batch_acc += current_batch_acc
    return batch_acc / batch_num


from keras.callbacks import *


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print
        print 'acc: %f%%' % acc
        f = open("accuracy_equation.txt", "a")
        f.writelines(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\t" + 'acc: %f%%' % acc + "\n")
        f.close()



evaluator = Evaluate()

dir_val_path = "/home/wjg/datasets/image_contest_level_1_validate/"
def calculate():
    f = open("accuracy_validation_4th_2.txt", "a")
    index = 199936
    while index < 200000:
        X = np.zeros((64, width, height, 3), dtype=np.float32)
        for i in range(64):
            X[i] = utils.load_image(dir_val_path + str(index + i) + ".png").transpose(1, 0, 2)
        y_pred = base_model.predict(X)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        # print out[0]
        if out.shape[1] == n_len or out.shape[1] == n_len - 2:
            results = []
            for i in range(out.shape[0]):
                label_pred = []
                for num in out[i]:
                    if num != -1:
                        label_pred.append(utils.get_char_from_type(num))
                    else:
                        break
                if label_pred.__len__() != 5 and label_pred.__len__() != 7:
                    print "Predict failed!"
                    f.write("Predict failed!\n")
                    continue
                equation = ''.join(label_pred)
                try:
                    result = eval(equation)
                except:
                    result = 'eval failed!'
                print "index " + str(index + i) + ": " + ''.join(label_pred) + " " + str(result)
                # results.append(result)
                f.write(equation + " " + str(result) + "\n")
        else:
            print "Predict failed! Shape is " + str(out.shape[1])
            f.write("Predict failed! Shape is " + str(out.shape[1]) + "\n")
        index += 128
    f.close()

calculate()

# class SaveMyModel(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         model.save('gru_equation.h5')
#         print "model saved to gru_equation.h5"
#
# save_my_model = SaveMyModel()
#
# images = utils.load_entire_image(dir_path, [str(i) + ".png" for i in range(10000)])
# labels, labels_length = utils.load_labels(dir_path + "labels.txt")
#
# model.fit(np.array(images), np.array(labels), batch_size=128, nb_epoch=100, validation_split=0.05, callbacks=[save_my_model])

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        f = open("batch_losses.txt", "a")
        f.write(str(logs.get('loss')) + '\n')
        f.close()

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        f = open("val_acc.txt", "a")
        f.write(str(logs.get('val_acc')) + '\n')
        f.close()

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

history = LossHistory()

# validation_population = random.sample(origin_population, origin_population.__len__())
# acc = evaluate(base_model, batch_num=780) * 100
# print 'acc: %f%%' % acc

epoch_count = 0
print "========================= epoch " + str(epoch_count) + " ========================="

while False:
    model.fit_generator(gen(batch_size=128, isTrain=True), samples_per_epoch=512, nb_epoch=1,
                        callbacks=[evaluator, history],
                        nb_worker=2, pickle_safe=True)
    model.save('gru_equation.h5')
    print "model saved to gru_equation.h5"
    train_population, validation_population = sample_train_and_validation(origin_population)    # resample

    # if epoch_count % 5 == 0:
    #     time.sleep(300)  # have a rest!
    epoch_count += 1

    print "========================= epoch " + str(epoch_count) + " ========================="