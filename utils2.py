import skimage
import skimage.io
import skimage.transform
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    img = skimage.transform.resize(img, (80, 170))
    return img

def load_entire_image(path, filename_list):
    imgs = []
    # load image
    for filename in filename_list:
        img = skimage.io.imread(path + filename)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        img = skimage.transform.resize(img, (80, 170))
        # img = img.transpose(1, 0, 2)
        imgs.append(img)
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    return imgs

def load_entire_image2(path, filename_list):
    images = np.zeros([filename_list.__len__(), 80, 170, 3], dtype=np.uint8)
    # load image
    print filename_list.__len__()
    for i in range(filename_list.__len__()):
        img = skimage.io.imread(path + filename_list[i])
        # img = img / 255.0
        # assert (0 <= img).all() and (img <= 1.0).all()
        print img
        print '##########################'
        img = skimage.transform.resize(img, (80, 170), preserve_range=True)

        # img = img.transpose(1, 0, 2)
        # imgs.append(img)
        print img
        images[i] = img
        print '###########################'
        print images[i]
        plt.imshow(images[i])
        pylab.show()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    return images

def next_batch(images, labels, start, length):
    return np.array(images[start : start + length]), np.array(labels[start : start + length])

def load_labels(path):
    f = open(path, "r")
    lines = f.readlines()
    labels = []
    labels_length = []
    for line in lines:
        label = [0] * 7
        isEnd = False
        for i in range(7):
            if line[i] == ' ':
                isEnd = True
            if isEnd:
                label[i] = -1
            else:
                char_type = get_char_type(line[i])
                label[i] = char_type
        labels.append(label)
        if isEnd:
            labels_length.append(5)
        else:
            labels_length.append(7)
    return labels, labels_length

def load_labels2(path):
    f = open(path, "r")
    lines = f.readlines()
    labels = []
    labels_length = []
    for line in lines:
        label = [0] * 5
        isEnd = False
        for i in range(5):
            if line[i] == ' ':
                isEnd = True
            if isEnd:
                label[i] = 15
            else:
                char_type = get_char_type(line[i])
                label[i] = char_type
        labels.append(label)
        labels_length.append(5)
    return labels, labels_length

char_type_dict = {
        "0" : 0, "1" : 1, "2" : 2, "3" : 3, "4" : 4, "5" : 5, "6" : 6, "7" : 7, "8" : 8, "9" : 9,
        "+" : 10, "-" : 11, "*" : 12,
        "(" : 13, ")" : 14,
        " " : 15
    }
type_char_dict = {v:k for k,v in char_type_dict.items()}

def get_char_type(c):
    return char_type_dict[c]

def get_char_from_type(type):
    return type_char_dict[type]

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
