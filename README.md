# CNN_Image_Denoising
code:[《基于深度卷积神经网络的图像去噪研究》](http://kns.cnki.net/kcms/detail/detail.aspx?filename=JSJC201703042&amp;dbcode=CJFD&amp;dbname=CJFD2017&amp;v=)


## 制作训练集

- `generateData.py` 用于制作训练数据集，使用voc作为图片来源，可添加高斯、椒盐等不同程度噪声

## 训练去噪网络

- `cnn_denoising_train.py` 训练网络，需要使用上面生成的h5数据集
- `cnn_denoising_test.py` 测试
- `test_muli_image.py` 测试标准图

