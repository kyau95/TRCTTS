"""
Currently NO training done
Need to implement necessary functions for training and loss
Goal is to not use a pretrained model and use my own
"""

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
from keras import regularizers
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

EPSILON = 1e-6
MOMENTUM = 0.99
KERN_SIZE = (3, 3)
class EastModel:
    def __init__(self, input_size=512):
        input_image = Input(shape=(None, None, 3), name='input_image')
        small_text_mask = Input(shape=(None, None, 1))
        text_region_mask = Input(shape=(None, None, 1))
        target_score_map = Input(shape=(None, None, 1))
        resnet = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)
        x = resnet.get_layer('activation_49').output

        x = Lambda(resize_bilinear, name='resize_1')(x)
        x = concatenate([x, resnet.get_layer('activation_40').output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(EPSILON))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, KERN_SIZE, padding='same', kernel_regularizer=regularizers.l2(EPSILON))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        x = concatenate([x, resnet.get_layer('activation_22').output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, KERN_SIZE, padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(resnet.get_layer('activation_10').output)], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(EPSILON))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, KERN_SIZE, padding='same', kernel_regularizer=regularizers.l2(EPSILON))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, KERN_SIZE, padding='same', kernel_regularizer=regularizers.l2(EPSILON))(x)
        x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x) 
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, small_text_mask, text_region_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = small_text_mask
        self.text_region_boundary_training_mask = text_region_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map


def dice_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, loss_weight, small_text_weight, score_y_true, score_y_pred):
  eps = 1e-5
  _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
  intersection = tf.reduce_sum(score_y_true * score_y_pred * _training_mask)
  union = tf.reduce_sum(score_y_true * _training_mask) + tf.reduce_sum(score_y_pred * _training_mask) + eps
  loss = 1. - (2. * intersection / union)

  return loss * loss_weight

def rbox_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, small_text_weight, target_score_map, geo_y_true, geo_y_pred):
  # d1 -> top, d2->right, d3->bottom, d4->left
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
  _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask

  return tf.reduce_mean(L_g * target_score_map * _training_mask)