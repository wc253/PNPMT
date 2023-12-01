from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, PReLU, \
    GlobalAveragePooling2D, Concatenate, Multiply, Conv2DTranspose, Layer, Add, Permute, RepeatVector, Dropout, \
    AveragePooling2D, subtract, Input
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1, L2, L1L2


##################################################################################
# FFDNet
##################################################################################
def block_ffdnet(y, norm_style='', dropout=0, weight=0.01, dim=96):
    if norm_style == 'l1':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(y)
    elif norm_style == 'l2':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(y)
    elif norm_style == 'l1_l2':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(y)
    else:
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    if dropout > 0:
        y = Dropout(dropout)(y)
    return y


def channel_down(input_tensor, noiseLevel, upscale_factor=2):
    (batch_size, in_height, in_width, channels) = input_tensor.shape
    out_height = int(in_height / upscale_factor)
    out_width = int(in_width / upscale_factor)
    input_reshape = Reshape((out_height, upscale_factor, out_width, upscale_factor, channels), )(input_tensor)
    channels *= upscale_factor ** 2
    input_reshape = Permute((2, 4, 1, 3, 5))(input_reshape)
    input_reshape = Reshape((out_height, out_width, channels))(input_reshape)
    noiseLevel = RepeatVector(out_width * out_height)(noiseLevel)
    noiseLevel = Reshape((out_height, out_width, 1))(noiseLevel)
    input_cat = Concatenate()([input_reshape, noiseLevel])
    return input_cat


def channel_model(input_tensor, block_num=12, upscale_factor=2, norm_style='', dropout=0, weight=0.01, dim=96):
    if norm_style == 'l1':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(input_tensor)
    elif norm_style == 'l2':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(input_tensor)
    elif norm_style == 'l1_l2':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(input_tensor)
    else:
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    for i in range(block_num):
        x = block_ffdnet(x, norm_style=norm_style, dropout=dropout, weight=weight, dim=dim)

    if norm_style == 'l1':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L1(weight))(x)
    elif norm_style == 'l2':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L2(weight))(x)
    elif norm_style == 'l1_l2':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L1L2(l1=weight, l2=weight))(x)
    else:
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Activation('tanh')(x)

    return x


def channel_up(input_tensor, upscale_factor=2):
    (batch_size, in_height, in_width, channels) = input_tensor.shape
    channels /= upscale_factor ** 2
    channels = int(channels)
    input_reshape = Reshape((upscale_factor, upscale_factor, in_height, in_width, channels))(input_tensor)
    input_reshape = Permute((3, 1, 4, 2, 5))(input_reshape)
    out_height = int(in_height * upscale_factor)
    out_width = int(in_width * upscale_factor)
    input_reshape = Reshape((out_height, out_width, channels))(input_reshape)
    return input_reshape


def fddnet(channel_input, channel_noise, block_num, norm, dropout, weight, dim):
    channel_connect = channel_down(channel_input, channel_noise)
    channel_fea = channel_model(channel_connect, block_num=block_num, norm_style=norm, dropout=dropout, weight=weight,
                                dim=dim)
    channel_output = channel_up(channel_fea)
    return channel_output

