"""
Currently NO training done
Need to implement necessary functions for training and loss
Goal is to not use a pretrained model and use my own
"""

import keras
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, ZeroPadding2D, Activation
from keras import regularizers
from keras.losses import Loss
import keras.backend as K
import tensorflow as tf
import numpy as np

RESIZE_FACTOR = 2
def resize_bilinear(x):
    return tf.image.resize_bilinear(
        x, 
        size=[K.shape(x)[1] * RESIZE_FACTOR, K.shape(x)[2] * RESIZE_FACTOR]
    )

def resize_output_shape(in_shape):
    shape = list(in_shape)
    if len(shape) != 4:
        return
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)

EPSILON = 1e-5
MOMENTUM = 0.997

# EAST Model
resnet = tf.keras.applications.ResNet50(input_shape=(512, 512, 3), weights='imagenet', include_top=False)
tf.keras.backend.clear_session()
x = resnet.get_layer('conv5_block3_out').output

x = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_1')(x)
x = tf.keras.layers.concatenate([x, resnet.get_layer('conv4_block6_out').output], axis=3)
x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_2')(x)
x = tf.keras.layers.concatenate([x, resnet.get_layer('conv3_block4_out').output], axis=3)
x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',name='resize_3')(x)
x = tf.keras.layers.concatenate([x, resnet.get_layer('conv2_block3_out').output], axis=3)
x = tf.keras.layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)

x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(EPSILON))(x)
x = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
x = tf.keras.layers.Activation('relu')(x)

pred_score_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
rbox_geo_map = tf.keras.layers.Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
angle_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
angle_map = tf.keras.layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
output = tf.keras.layers.concatenate([pred_score_map,rbox_geo_map, angle_map], axis=3, name='pred_map')

model = tf.keras.models.Model(inputs=resnet.input, outputs= output,name='EAST')
for layers in resnet.layers:
  layers.trainable=False  


# loss functions based on paper https://arxiv.org/abs/1704.03155
# dice coefficient loss (CLS)
def dice_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, loss_weight, small_text_weight, score_y_true, score_y_pred):
  mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
  intersection = tf.reduce_sum(score_y_true * score_y_pred * mask)
  union = tf.reduce_sum(score_y_true * mask) + tf.reduce_sum(score_y_pred * mask) + EPSILON
  loss = 1. - (2. * intersection / union)
  return loss * loss_weight

def rbox_loss(small_training_mask, text_training_mask, small_text_weight, target_score_map, geo_y_true, geo_y_pred):
  # d1->top, d2->right, d3->bottom, d4->left
  d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=geo_y_true, num_or_size_splits=5, axis=3)
  d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=geo_y_pred, num_or_size_splits=5, axis=3)
  area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
  area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
  w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
  h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
  area_intersect = w_union * h_union
  area_union = area_gt + area_pred - area_intersect
  L_AABB = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))
  L_theta = 1 - tf.cos(theta_pred - theta_gt)
  L_g = L_AABB + 20 * L_theta
  _training_mask = tf.minimum(small_training_mask + small_text_weight, 1) * text_training_mask

  return tf.reduce_mean(L_g * target_score_map * _training_mask)

class TotalLoss(Loss):
    def __init__(self, 
                 from_logits=False, 
                 reduction=tf.keras.losses.Reduction.AUTO):
        super(TotalLoss, self).__init__(reduction=reduction)
    
    def call(self, y_true, y_pred):
        # geo map and score maps
        y_true_cls = y_true[:, :, :, 0]
        y_true_geo = y_true[:, :, :, 1:6]
        y_pred_cls = y_pred[:, :, :, 0]
        y_pred_geo = y_pred[:, :, :, 1:6]
        train_mask = y_true[:, :, :, :6]

        d_loss = dice_loss(y_true_cls, y_pred_cls, train_mask)
        d_loss *= 0.01 # scaling

        r_loss = rbox_loss(y_true_cls, y_true_geo, y_pred_geo, train_mask)
        return 100 * (r_loss + d_loss)

"""
Current error compiling
"""
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=TotalLoss())

# model.evaluate()