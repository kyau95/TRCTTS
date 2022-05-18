import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.layers import Activation, BatchNormalization, concatenate, Conv2D, Lambda, UpSampling2D
import numpy as np

resnet = ResNet50(input_shape=(512, 512, 3), weights='imagenet', include_top=False)
tf.keras.backend.clear_session()
x = resnet.get_layer('conv5_block3_out').output

x = UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_1')(x)
x = concatenate([x, resnet.get_layer('conv4_block6_out').output], axis=3)
x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)

x = UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_2')(x)
x = concatenate([x, resnet.get_layer('conv3_block4_out').output], axis=3)
x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)

x = UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_3')(x)
x = concatenate([x, resnet.get_layer('conv2_block3_out').output], axis=3)
x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)

x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
x = Activation('relu')(x)

pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
output = concatenate([pred_score_map,rbox_geo_map, angle_map], axis=3, name='pred_map')


model = tf.keras.models.Model(inputs=resnet.input, outputs= output,name='EAST')
for layers in resnet.layers:
    layers.trainable=False

