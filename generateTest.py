#coding=utf-8
import h5py
import cv2
#获取数据
def loadH5data(filenum):
    file = h5py.File('./train{0}.h5'.format(filenum),'r')
    data_train = file['data'][:]
    data_label = file['label'][:]
    print data_label.shape
    return data_train,data_label

data,label = loadH5data(0)
print data.shape
print label.shape
for i in xrange(30):
    img = data[i,...]
    noisy = label[i,...]
    print img
    cv2.imshow('im',img)
    cv2.imshow('no',noisy)
    cv2.waitKey(1000)
