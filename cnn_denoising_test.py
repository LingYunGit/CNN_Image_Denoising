#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def conv2d(input,w,b,name):
    conv2  = tf.nn.conv2d(input,w,[1,1,1,1],padding='SAME',name='{0}_Conv'.format(name))#,name=(name+' Conv')
    conv2d_out = tf.nn.bias_add(conv2,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return conv2d_out
def deConv2d(input,w,b,output_shape,name):
    deconv2d = tf.nn.deconv2d(input,w,output_shape,[1,1,1,1],padding='SAME',name='{0}_DeConv'.format(name))
    deconv2d_out = tf.nn.bias_add(deconv2d,b,name='{0}_Bias_Add'.format(name))#,name=(name+' Bias_Add')
    return deconv2d_out
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

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

test_imgname = 'Lena512'
#输入测试图像
content_image = cv2.imread('../image/image/{0}_noisy.png'.format(test_imgname),0)
if content_image.ndim ==2:
    content_image = content_image[None,...,None]
IMAGE_HEIGHT=content_image.shape[1]
IMAGE_WIDTH = content_image.shape[2]

#构建网络
img_input = tf.placeholder(tf.float32,name='img_input')
img_label = tf.placeholder(tf.float32,name='img_label')
withMaxPool = True
if(withMaxPool):
    conv1 = conv2d(img_input,weight['c1w'],biases['c1b'],'con1')
    conv1 = tf.nn.relu(conv1,name='con1_relu')
    #conv1 = tf.nn.avg_pool(conv1,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool1')
    conv2 = conv2d(conv1,weight['c2w'],biases['c2b'],'con2')
    conv2 = tf.nn.relu(conv2,name='con2_relu')
    #conv2 = tf.nn.avg_pool(conv2,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool2')
    conv3 = conv2d(conv2,weight['c3w'],biases['c3b'],'con3')
    conv3 = tf.nn.relu(conv3,name='con3_relu')
    #conv3 = tf.nn.avg_pool(conv3,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool3')
    conv4 = deConv2d(conv3,weight['c4w'],biases['c4b'],[1,IMAGE_HEIGHT,IMAGE_WIDTH,32],'con4')
    conv4 = tf.nn.relu(conv4,name='con4_relu')
    #conv4 = tf.nn.avg_pool(conv4,ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1],padding='SAME',name='Pool4')
    conv5 = deConv2d(conv4,weight['c5w'],biases['c5b'],[1,IMAGE_HEIGHT,IMAGE_WIDTH,1],'con5')
else:
    conv1 = conv2d(img_input,weight['c1w'],biases['c1b'],'con1')
    conv1 = tf.nn.relu(conv1,name='con1_relu')
    conv2 = conv2d(conv1,weight['c2w'],biases['c2b'],'con2')
    conv2 = tf.nn.relu(conv2,name='con2_relu')
    conv3 = conv2d(conv2,weight['c3w'],biases['c3b'],'con3')
    conv3 = tf.nn.relu(conv3,name='con3_relu')
    conv4 = conv2d(conv3,weight['c4w'],biases['c4b'],'con4')
    conv4 = tf.nn.relu(conv4,name='con4_relu')
    conv5 = conv2d(conv4,weight['c5w'],biases['c5b'],'con5')
loss = tf.nn.l2_loss(img_label-conv5,name='l2-loss')

#添加噪声
content_image_place = tf.placeholder(tf.float32,name ='clear_image')
content_image_place_shape = tf.placeholder(tf.int32,name ='clear_image_shape')
noisy = tf.truncated_normal(content_image_place_shape, stddev=25)
noisy_image = tf.add(content_image_place,noisy,name='add_noise')

#设置要保存的变量
#设置要保存的变量
saver = tf.train.Saver(tf.all_variables())
init = tf.initialize_all_variables()



with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('./logs/test_logs',sess.graph_def)
    #恢复变量
    saver.restore(sess,'./save/model_cnn_train178.ckpt')

    #添加噪声
    #noisy_im = sess.run(noisy_image,feed_dict={content_image_place_shape:content_image.shape,content_image_place:content_image/1.0})

    test_x = content_image/255.0
    #预测
    pred_c1,pred_c4,pred = sess.run([conv1,conv4,conv5], feed_dict={img_input:test_x})
    #pred = sess.run(conv1,feed_dict={img_input:test_x})
    #保存图像
    for pc in xrange(pred_c1.shape[3]):
        plt.imsave('./save/image/conv1_{0}.png'.format(pc), np.clip(pred_c1[0, ...,pc]*255,0,255).astype(np.uint8), cmap='gray')
    for pc in xrange(pred_c4.shape[3]):
        plt.imsave('./save/image/conv4_{0}.png'.format(pc), np.clip(pred_c4[0, ...,pc]*255,0,255).astype(np.uint8),cmap='gray')


im_out_noisy = np.clip(test_x[0, ...,0]*255,0,255).astype(np.uint8)
im_out_denoisy = np.clip(pred[0, ...,0]*255,0,255).astype(np.uint8)
#保存图像
for pc in xrange(pred.shape[3]):
    cv2.imwrite('./save/image/conv5_{0}.png'.format(pc), im_out_denoisy)
#原始图像
img_normal = cv2.imread('../image/image/{0}.png'.format(test_imgname),0)
#Clear Image
img_input = tf.placeholder(tf.float32,name='img_input')
img_output = tf.placeholder(tf.float32,name='img_denoising')
sub2 = tf.pow(img_output-img_input,2)
#均方差
mse = tf.reduce_sum(sub2)/(img_normal.shape[0]*img_normal.shape[1])
psnr = 20*tf.log(255/tf.sqrt(mse))/tf.log(tf.constant(10,tf.float32))
with tf.Session() as sess:
    print 'noisy   psnr : ',sess.run(psnr,feed_dict={img_input:img_normal,img_output:im_out_noisy})
    print 'denoisy psnr: ',sess.run(psnr,feed_dict={img_input:img_normal,img_output:im_out_denoisy})

#显示图像
plt.subplot(131)
plt.title('noisy')
plt.imshow(img_normal,cmap='gray')
#Noisy Image
plt.subplot(132)
plt.title('noisy')
plt.imshow(im_out_noisy,cmap='gray')
#Denoisy Image
plt.subplot(133)
plt.title('pred')
plt.imshow(im_out_denoisy,cmap='gray')
plt.show()
