#coding=utf-8
import cv2
import h5py
import numpy as np
import random

ROOT_VOC_DIRECTORY = './VOC2012/'


#读取所有图片的名字
def readImageName():
    namesFile=ROOT_VOC_DIRECTORY+'all_image_names.txt'
    allImageNames = []
    fd = file( namesFile, "r" )
    for line in fd.readlines():
        allImageNames.append(line.strip().lstrip())
    fd.close()
    return allImageNames

#添加椒盐噪声
def SaltAndPepper(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

def GaessNoisy(src,sigma):
    NoiseImg = src.copy()
    s = np.random.normal(0, 1, size=src.shape)*sigma
    NoiseImg = np.add(NoiseImg,s)
    NoiseImg.astype(dtype=np.uint8)
    return NoiseImg


def readImage2Data(pathcImageNmaes,patchSize,stride):
    patchImageInput = np.empty(shape=[0,patchSize,patchSize,1])
    patchImageLabel = np.empty(shape=[0,patchSize,patchSize,1])
    for i in xrange(len(pathcImageNmaes)):
        img = cv2.imread(ROOT_VOC_DIRECTORY+'JPEGImages/'+pathcImageNmaes[i],0)

        #添加椒盐噪声
        #noisyImage = SaltAndPepper(img,0.2)
        #添加高斯噪声
        noisyImage = GaessNoisy(img,100)
        #re to 0-1

        img = img/255.0
        noisyImage = noisyImage/255.0
        cv2.imshow('i',img)
        cv2.imshow('img',noisyImage)
        cv2.waitKey()
        row = (img.shape[0]-patchSize)/stride
        line = (img.shape[1]-patchSize)/stride
        imgPatch = np.zeros(shape=[row*line,patchSize,patchSize,1])
        imgPatchLabel = np.zeros(shape=[row*line,patchSize,patchSize,1])
        for r in xrange(row):
            for l in xrange(line):
                imgPatch[r*line+l,...,0]=img[r*stride:r*stride+patchSize,l*stride:l*stride+patchSize]
                imgPatchLabel[r*line+l,...,0]=noisyImage[r*stride:r*stride+patchSize,l*stride:l*stride+patchSize]
        patchImageInput = np.vstack((patchImageInput,imgPatch))
        patchImageLabel = np.vstack((patchImageLabel,imgPatchLabel))
    return patchImageInput,patchImageLabel



def writeData2H5(data,label,batchSize,fileNum):
    file = h5py.File('train{0}.h5'.format(fileNum),'w')
    data = data[0:(data.shape[0]-data.shape[0]%batchSize),...]
    label = label[0:(label.shape[0]-label.shape[0]%batchSize),...]
    #random
    '''
    randomM = list(xrange(data.shape[0]))
    random.shuffle(randomM)
    data = data[randomM,...]
    label = label[randomM,...]
    '''
    carh5data = file.create_dataset('data',data=data,shape=data.shape,dtype=np.float32)
    carh5label = file.create_dataset('label',data=label,shape=label.shape,dtype=np.float32)
    file.close()


if __name__ == '__main__':
    allImageNames = readImageName()
    PATCH_SIZE = 55     #片段大小
    STRIDE_SIZE = 30    #步长大小
    BATCH_SIZE = 128    #训练一个批次的大小
    NUM_ONE_H5 = 5   #一个h5文件放图片的数量
    NUM_H5 = 1        #h5文件数量
    NUM_BEGIN = 0     #第几个h5文件开始
    NUM_H5_MAX = len(allImageNames)/NUM_ONE_H5
    for i in range(NUM_BEGIN,min(NUM_H5,NUM_H5_MAX),1):
        patchImageName = allImageNames[i*NUM_ONE_H5:(i+1)*NUM_ONE_H5]
        data,label = readImage2Data(patchImageName,PATCH_SIZE,STRIDE_SIZE)
        writeData2H5(data,label,BATCH_SIZE,i)

