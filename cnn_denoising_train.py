#coding:utf-8

"""
训练用于去噪的卷积神经网络,训练集需要提前转换成.h5文件,通过h5py读取,训练后保存数据
"""

import tensorflow as tf
import h5py
import time
import cv2
import numpy as np

TIME_FORMAT = '%Y-%m-%d %X'#格式化输出时间

def conv2d(input,w,b,name):
    conv2  = tf.nn.conv2d(input,w,[1,1,1,1],padding='SAME',name='{0}_Conv'.format(name))#,name=(name+' Conv')
    conv2d_out = tf.nn.bias_add(conv2,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return conv2d_out

def deConv2d(input,w,b,output_shape,name):
    deconv2d = tf.nn.deconv2d(input,w,output_shape,[1,1,1,1],padding='SAME',name='{0}_DeConv'.format(name))
    deconv2d_out = tf.nn.bias_add(deconv2d,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return deconv2d_out

batch_size = 128
num_iters = 100
learning_rate = 0.0001
h5file_num = 5
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

#获取数据
def loadH5data(filenum):
    file = h5py.File('../DataSet/train{0}.h5'.format(filenum),'r')
    data_train = file['data'][:]
    data_label = file['label'][:]
    hw_train = data_train.shape[2]
    hw_label = data_label.shape[2]
    data_train = data_train.reshape((data_train.shape[0],hw_train,hw_train,1))
    data_label = data_label.reshape((data_label.shape[0],hw_label,hw_label,1))
    return data_train,data_label

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

#test PSNR
#输入测试图像
psnr_image_normal = cv2.imread('../image/image/{0}.png'.format('Lena512'),0)
psnr_image_noisy = cv2.imread('../image/image/{0}_noisy.png'.format('Lena512'),0)
if psnr_image_noisy.ndim ==2:
    psnr_image_noisy = psnr_image_noisy[None,...,None]
PSNR_IMAGE_HEIGHT=psnr_image_noisy.shape[1]
PSNR_IMAGE_WIDTH = psnr_image_noisy.shape[2]
psnr_image_noisy = psnr_image_noisy/255.0

#构建网络
img_input = tf.placeholder(tf.float32,name='img_input')
img_label = tf.placeholder(tf.float32,name='img_label')
conv1 = conv2d(img_input,weight['c1w'],biases['c1b'],'con1')
conv1 = tf.nn.relu(conv1,name='con1_relu')
#conv1 = tf.nn.avg_pool(conv1,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool1')
conv2 = conv2d(conv1,weight['c2w'],biases['c2b'],'con2')
conv2 = tf.nn.relu(conv2,name='con2_relu')
#conv2 = tf.nn.avg_pool(conv2,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool2')
conv3 = conv2d(conv2,weight['c3w'],biases['c3b'],'con3')
conv3 = tf.nn.relu(conv3,name='con3_relu')
#conv3 = tf.nn.avg_pool(conv3,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool3')
conv4 = deConv2d(conv3,weight['c4w'],biases['c4b'],[batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,32],'con4')
conv4 = tf.nn.relu(conv4,name='con4_relu')
#conv4 = tf.nn.avg_pool(conv4,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool4')
conv5 = deConv2d(conv4,weight['c5w'],biases['c5b'],[batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,1],'con5')
loss = tf.nn.l2_loss(img_label-conv5,name='l2-loss')
global_step = tf.Variable(0,trainable=False,name='Step')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


psnr_conv4 = deConv2d(conv3,weight['c4w'],biases['c4b'],[1,PSNR_IMAGE_HEIGHT,PSNR_IMAGE_WIDTH,32],'dcon4')
psnr_conv4 = tf.nn.relu(psnr_conv4,name='dcon4_relu')
#conv4 = tf.nn.avg_pool(conv4,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool4')
psnr_conv5 = deConv2d(psnr_conv4,weight['c5w'],biases['c5b'],[1,PSNR_IMAGE_HEIGHT,PSNR_IMAGE_WIDTH,1],'dcon5')




#PSNR Network
psnr_img_input = tf.placeholder(tf.float32,name='psnr_img_input')
psnr_img_output = tf.placeholder(tf.float32,name='psnr_img_denoising')
sub2 = tf.pow(psnr_img_output-psnr_img_input,2)
#均方差
mse = tf.reduce_sum(sub2)/(psnr_image_normal.shape[0]*psnr_image_normal.shape[1])
psnr = 20*tf.log(255/tf.sqrt(mse))/tf.log(tf.constant(10,tf.float32))
psnr_summary = tf.placeholder(tf.float32,name='psnr_summary')

tf.scalar_summary('Loss',loss)
tf.scalar_summary('PSNR',psnr_summary)
summary_op = tf.merge_all_summaries()

#设置要保存的变量
saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()
step = 0

max_psnr = 0

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('./logs/cnn_train_logs',sess.graph_def)
    #saver.restore(sess,'./save/model_iter30_lr0001.ckpt')
    for iter in xrange(num_iters):
        #print 'Iter:',iter
        for h5_num in xrange(h5file_num):
            data_train,data_label = loadH5data(h5_num+1)
            data_batch_num = data_train.shape[0]/batch_size
            for batch in xrange(data_batch_num):
                temp_input = data_train[batch*batch_size:(batch+1)*batch_size,(55-IMAGE_WIDTH)/2:55-(55-IMAGE_WIDTH)/2,(55-IMAGE_WIDTH)/2:55-(55-IMAGE_WIDTH)/2,...]
                temp_label = data_label[batch*batch_size:(batch+1)*batch_size,(55-IMAGE_WIDTH)/2:55-(55-IMAGE_WIDTH)/2,(55-IMAGE_WIDTH)/2:55-(55-IMAGE_WIDTH)/2,...]
                _,loss_out = sess.run([train_step,loss],feed_dict={img_input:temp_input,img_label:temp_label})
                step = step+1;
                if step%5==0:
                    pred = sess.run(psnr_conv5, feed_dict={img_input:psnr_image_noisy})
                    psnr_out_denoisy = np.clip(pred[0, ...,0]*255,0,255).astype(np.uint8)
                    psnr_num = sess.run(psnr,feed_dict={psnr_img_input:psnr_image_normal,psnr_img_output:psnr_out_denoisy})
                    summary_str = sess.run(summary_op,feed_dict={img_input:temp_input,img_label:temp_label,psnr_summary:psnr_num})
                    summary_writer.add_summary(summary_str,step)
                    print ''+str(iter)+','+str(h5_num)+','+time.strftime(TIME_FORMAT,time.localtime())+"," + str(sess.run(global_step)) + "," + "{:.6f}".format(loss_out)+','+'{:.6f}'.format(learning_rate)+','+str(psnr_num)
                    if psnr_num>max_psnr:
                        max_psnr=psnr_num
                        #print '    save model ',iter
                        saver.save(sess,'./save/model_cnn_train{0}_best.ckpt'.format(iter))
        #保存变量
        if iter%5==0:
            print 'save paramenter seccess'
            saver.save(sess,'./save/model_cnn_train{0}.ckpt'.format(iter))

print 'End!'





