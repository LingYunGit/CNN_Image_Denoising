# -*- coding: utf-8 -*-
import argparse
from datetime import datetime as dt

import numpy as np
import scipy.misc

from Denoising.opensource.models import VGG16, I2V

mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
    for i in range(3):
        img[0,:,:,i] += mean[i]
    return img

#减掉均值
def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

#读取图片，并将按比例调整图片大小
def read_image(path, w=None):
    img = scipy.misc.imread(path)
    # Resize if ratio is specified
    if w:
        r = w / np.float32(img.shape[1])
        img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
    img = img.astype(np.float32)
    img = img[None, ...]
    return img

#零均值
def read_image_sub_mean(path,w=None):
    # Subtract the image mean
    return sub_mean(read_image(path,w))

# def add_noisy(img, mean=0.0, stddev=1.0):
#     return img


#保存图片
def save_image(im, it):
    img = im.copy()
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    tstr = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    scipy.misc.imsave("./im_na_%s_%05d.png"%(tstr,it), img)

def save_image_add_mean(im,it):
    img = im.copy()
    # Add the image mean
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    tstr = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    scipy.misc.imsave("./im_na_%s_%05d.png"%(tstr,it), img)

#分割图片
# batch_size为分割图像的大小
# stride为滑动窗口移动的间隔
def clip_image(image,batch_size,stride):
    hang = image.shape[1]
    lie = image.shape[2]
    if hang<batch_size or lie<batch_size:
        print 'Wranging : image too small'
        return image
    #计算批数
    hangshu = (hang-batch_size)/stride+1
    lieshu = (lie-batch_size)/stride+1

    new_image = np.zeros((hangshu*lieshu,batch_size,batch_size,image.shape[3]))

    for l in xrange(lieshu):
        for h in xrange(hangshu):
            new_image[l*hangshu+h]=image[0][h*stride:h*stride+batch_size,l*stride:l*stride+batch_size,...]
    return  new_image


#读取命令的参数
def parseArgs():
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--model', '-m', default='vgg',
                        help='Model type (vgg, i2v)')
    parser.add_argument('--modelpath', '-mp', default='vgg',
                        help='Model file path')
    parser.add_argument('--content', '-c', default='images/sd.jpg',
                        help='Content image path')
    parser.add_argument('--style', '-s', default='images/style.jpg',
                        help='Style image path')
    parser.add_argument('--width', '-w', default=800, type=int,
                        help='Output image width')
    parser.add_argument('--iters', '-i', default=5000, type=int,
                        help='Number of iterations')
    parser.add_argument('--alpha', '-a', default=1.0, type=float,
                        help='alpha (content weight)')
    parser.add_argument('--beta', '-b', default=200.0, type=float,
                        help='beta (style weight)')
    args = parser.parse_args()
    return args.content, args.style, args.modelpath, args.model, args.width, args.alpha, args.beta, args.iters

#获取模型
def getModel(image, params_path, model):
    if model == 'vgg':
        return VGG16(image, params_path)
    elif model == 'i2v':
        return I2V(image, params_path)
    else:
        print 'Invalid model name: use `vgg` or `i2v`'
        return None