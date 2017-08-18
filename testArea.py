
# coding:utf8
import cv2
import numpy as np


def preprocess(gray):
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)

    return dilation2


def findTextRegion(img):
    region = []
    areamaps = {}
    # 1. 查找轮廓
    ig, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 3000):
            continue
        areamaps[(x, y, w, h)] = area
        # 面积小的都筛选掉
        if (area < 3000):
            continue
        height = h
        width = w
        if width<height:
            continue
        if height * 10 < width:
            continue
        region.append((x, y, w, h))
    return region, areamaps

# 定义缩放resize函数
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image

    # 宽度是0
    if width is None:
        # 则根据高度计算缩放比例
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果高度为0
    else:
        # 根据宽度计算缩放比例
        r = width / float(w)
        dim = (width, int(h * r))

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回缩放后的图像
    return resized

def expanderArea(area):
    addw = 25
    addh = 15
    pw = 4000
    ph = 1500
    x,y,w,h =area
    # print("befor:")
    # if x!=0 or y!=0:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)
    # print(area)
    if w is  0 and h is 0 :
        return area
    # 重新计算x
    if x>addw:
        if x+w+addw>pw:
            w=pw-x+addw
        else:
            w=w+2*addw
        x = x-addw
    else:
        w=w+x+addw
        x=0
        # if x+addw>pw:
        #     w=pw-4000
        # else:
        #     w=w+2*addw
    #  重新计算y
    if y > addh:
        if y + h + addh > ph:
            h = ph - y + addh
        else:
            h = h + 2 * addh
        y = y - addh
    else:
         h = h + y + addh
         y = 0

    # print("after:")
    # if x!=0 or y!=0:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # print((x,y,w,h))
    return x,y,w,h

def detect(img,index):
    img = resize(img)
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region, areamaps = findTextRegion(dilation)
    def getValue(item):
        c, d = item;
        return d;

    # print(areamaps.items())
    # , reverse=True   lambda item: item[1]
    sortedmap = sorted(areamaps.items(), key=getValue, reverse=True)
    # print(sortedmap)
    if len(sortedmap)<2:
        print("error%d "%index)
        return ((0,0,0,0),(0,0,0,0),(0,0,0,0))
        # return "error"
    if len(sortedmap)==2:
        sortedmap.append(((0,0,0,0),0))
    big, ar1 = sortedmap[0]
    small1, ar2 = sortedmap[1]
    small2, ar3 = sortedmap[2]
    big = expanderArea(big)
    small1 = expanderArea(small1)
    small2 = expanderArea(small2)
    # print("result")
    if small1[1] > small2[1]:
        if(small1[1]>4000+small2[1]):
            # print(r"两个区域")
            # print(small1, ar2)
            # print(big, ar1)
            return (0,0,0,0),small1,big
        # print(small2, ar3)
        # print(small1, ar2)
        # print(big, ar1)
        return small2,small1,big
        # return small2, ar3,small1, ar2,big, ar1
    else:
        if (small1[1]+ 4000 < small2[1]):
            # print(r"两个区域")
            # print(small2, ar2)
            # print(big, ar1)
            return ((0,0,0,0),0),small2,big
        # print(small1, ar2)
        # print(small2, ar3)
        # print(big, ar1)
        return small1,small2,big
        # return small1, ar2,small2, ar3,big, ar1

    # print("areasssss:")
    # print(areas)
    # print(sorted(areas))
    # i=0
    # for i in range(3):


    # 4. 用绿线画出这些找到的轮廓
    # for x, y, w, h in region:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)

    # 带轮廓的图片
    # cv2.imwrite("contours.png", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def write_result(writer,str1):
    str1=writer.write(str(str1)+"\n")
    return str
if __name__ == '__main__':
    # 读取文件
    writer=open('split_result.txt', 'w')
    for i in range(100000):
        print(i)
        img = cv2.imread('/home/wjg/datasets/image_contest_level_2/%d.png'%i)
        small1, small2, big = detect(img, i)
        write_result(writer, (small1, small2, big))
        # if small1[0] == 0 and small1[1] == 0 and small1[2] == 0 and small1[3] == 0:
        #     continue
        if max(small1) > 0:
            cv2.imwrite('/home/wjg/datasets/image_contest_level_2_split/%d_small1.png'%i, img[small1[1]:small1[1]+small1[3], small1[0]:small1[0]+small1[2], :])
        if max(small2) > 0:
            cv2.imwrite('/home/wjg/datasets/image_contest_level_2_split/%d_small2.png' % i, img[small2[1]:small2[1] + small2[3], small2[0]:small2[0] + small2[2], :])
        if max(big) > 0:
            cv2.imwrite('/home/wjg/datasets/image_contest_level_2_split/%d_big.png' % i, img[big[1]:big[1] + big[3], big[0]:big[0] + big[2], :])

    writer.close()
