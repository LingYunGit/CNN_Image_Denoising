#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
def conv2d(input,w,b,name):
    conv2  = tf.nn.conv2d(input,w,[1,1,1,1],padding='SAME',name='{0}_Conv'.format(name))#,name=(name+' Conv')
    conv2d_out = tf.nn.bias_add(conv2,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return conv2d_out
def deConv2d(input,w,b,output_shape,name):
    deconv2d = tf.nn.deconv2d(input,w,output_shape,[1,1,1,1],padding='SAME',name='{0}_DeConv'.format(name))
    deconv2d_out = tf.nn.bias_add(deconv2d,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return deconv2d_out

#设置权重
weight={
    'c1w':tf.Variable(tf.random_normal([ 5 , 5 , 1 , 32 ],mean=0,stddev=0.001),trainable=True,name='Weight_Conv1'),
    'c2w':tf.Variable(tf.random_normal([ 3 , 3 , 32 , 64 ],mean=0,stddev=0.001),trainable=True,name='Weight_Conv2'),
    'c3w':tf.Variable(tf.random_normal([ 1 , 1 , 64 , 64 ],mean=0,stddev=0.001),trainable=True,name='Weight_Conv3'),
    'c4w':tf.Variable(tf.random_normal([ 3 , 3 , 32 , 64 ],mean=0,stddev=0.001),trainable=True,name='Weight_Conv4'),
    'c5w':tf.Variable(tf.random_normal([ 5 , 5 , 1 , 32 ],mean=0,stddev=0.001),trainable=True,name='Weight_Conv5'),
}
#设置偏置
biases={
    'c1b':tf.Variable(tf.zeros([32]),trainable=True,name='Biases_Conv1'),
    'c2b':tf.Variable(tf.zeros([64]),trainable=True,name='Biases_Conv2'),
    'c3b':tf.Variable(tf.zeros([64]),trainable=True,name='Biases_Conv3'),
    'c4b':tf.Variable(tf.zeros([32]),trainable=True,name='Biases_Conv4'),
    'c5b':tf.Variable(tf.zeros([1]),trainable=True,name='Biases_Conv5'),
}


all_image_name = []
ImageType = 1
if ImageType==1:
    IMAGE_HEIGHT= 512
    IMAGE_WIDTH = 512
    all_image_name.append('barbara')
    all_image_name.append('boat')
    all_image_name.append('couple')
    all_image_name.append('fingerprint')
    all_image_name.append('hill')
    all_image_name.append('Lena512')
    all_image_name.append('man')
elif ImageType==2:
    IMAGE_HEIGHT= 256
    IMAGE_WIDTH = 256
    all_image_name.append('Cameraman256')
    all_image_name.append('house')
    all_image_name.append('montage')
    all_image_name.append('peppers256')
else:
    IMAGE_HEIGHT= 375
    IMAGE_WIDTH = 500
    all_image_name.append('voc_001')
    all_image_name.append('voc_002')
    all_image_name.append('voc_003')

#构建网络
img_input = tf.placeholder(tf.float32,name='img_input')
img_label = tf.placeholder(tf.float32,name='img_label')
conv1 = conv2d(img_input,weight['c1w'],biases['c1b'],'con1')
conv1 = tf.nn.relu(conv1,name='con1_relu')
conv2 = conv2d(conv1,weight['c2w'],biases['c2b'],'con2')
conv2 = tf.nn.relu(conv2,name='con2_relu')
conv3 = conv2d(conv2,weight['c3w'],biases['c3b'],'con3')
conv3 = tf.nn.relu(conv3,name='con3_relu')
conv4 = deConv2d(conv3,weight['c4w'],biases['c4b'],[1,IMAGE_HEIGHT,IMAGE_WIDTH,32],'con4')
conv4 = tf.nn.relu(conv4,name='con4_relu')
conv5 = deConv2d(conv4,weight['c5w'],biases['c5b'],[1,IMAGE_HEIGHT,IMAGE_WIDTH,1],'con5')
#conv5 = tf.nn.relu(conv5,name='con5_relu')
loss = tf.nn.l2_loss(img_label-conv5,name='l2-loss')
#计算PSNR
psnr_input = tf.placeholder(tf.float32,name='img_input')
psnr_output = tf.placeholder(tf.float32,name='img_denoising')
sub2 = tf.pow(psnr_output-psnr_input,2)
#均方差
mse = tf.reduce_sum(sub2)/(IMAGE_WIDTH*IMAGE_HEIGHT)
psnr = 20*tf.log(255/tf.sqrt(mse))/tf.log(tf.constant(10,tf.float32))

#设置要保存的变量
#设置要保存的变量
saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()

all_image_noisy = []
all_image_clear = []
all_image_denoisy = []


for i in xrange(len(all_image_name)):
    #输入测试图像
    content_image = cv2.imread('../image/image/{0}_noisy.png'.format(all_image_name[i]),0)
    if content_image.ndim ==2:
        content_image = content_image[None,...,None]/255.0
    all_image_noisy.append(content_image)
    img_normal = cv2.imread('../image/image/{0}.png'.format(all_image_name[i]),0)
    all_image_clear.append(img_normal)

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('./logs/test_logs',sess.graph_def)
    #恢复变量
    saver.restore(sess,'./save/model_cnn_train254.ckpt')
    if ImageType==1:
        file_result = file('./save/result.txt','w')
    else:
        file_result = file('./save/result.txt','a')
    #预测
    for j in xrange(len(all_image_name)):
        content_image = all_image_noisy[j]
        start_time = time.time()
        pred = sess.run(conv5, feed_dict={img_input:content_image.astype(np.float64)})
        end_time = time.time()
        cost_time = end_time-start_time
        psnr_noisy = sess.run(psnr,feed_dict={psnr_input:all_image_clear[j],psnr_output:np.clip(all_image_noisy[j][0, ...,0]*255,0,255).astype(np.uint8)})
        psnr_denoisy = sess.run(psnr,feed_dict={psnr_input:all_image_clear[j],psnr_output:np.clip(pred[0, ...,0]*255,0,255).astype(np.uint8)})
        cv2.imwrite('./save/image_de/{}_denoisy.png'.format(all_image_name[j]),np.clip(pred[0, ...,0]*255,0,255).astype(np.uint8))
        all_image_denoisy.append(pred)
        print '%-12s'%(all_image_name[j]),'[ noisy   psnr : ','%.4f'%psnr_noisy,' , denoisy psnr : ','%.4f'%psnr_denoisy,' , cost time : ','%.6f'%cost_time,'s ]'
        file_result.write(all_image_name[j]+','+str(psnr_noisy)+','+str(psnr_denoisy)+','+str(cost_time)+'\n')
    file_result.close()

#保存图像
#for pc in xrange(pred.shape[3]):
    #cv2.imwrite('./save/image/conv3_{0}.png'.format(pc), im_out_denoisy)



