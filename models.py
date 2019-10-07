from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv1D, Multiply, Lambda, Activation, Reshape, Add
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.layers.convolutional.subpixelupscaling import SubPixelUpscaling
import keras.backend as K
from typing import Tuple

def build_PatchGAN_Discriminator(input_shape: Tuple[int, int]):
    '''
    PatchGAN(Discriminator)のKerasモデルをつくる

    Parameters
    ----------
    input_shape: Tuple[int, int]
        MCEPsを転置したshape
        例: (36, 128)
    
    Returns
    -------
    model: Model
        Kerasのモデル
    
    Note
    ----
    input_shapeは(36, n)とかが良い
    ちなみに(35, n)ではうまくいかない
    '''
    def gated_liner_units(layer_inputs, gates):
        act = Activation('sigmoid')(layer_inputs)
        act = Multiply()([layer_inputs, act])
        return act
    def downsample2d(layer_inputs, filters, kernel_size, strides):
        h1            = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm       = InstanceNormalization(epsilon=1e-06)(h1)
        h1_gates      = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm_gates = InstanceNormalization(epsilon=1e-06)(h1_gates)
        h1_glu        = gated_liner_units(h1_norm, h1_norm_gates)
        return h1_glu
    input_layer   = Input(shape=input_shape)
    input_reshape = Reshape(input_shape+(1,))(input_layer)
    h1            = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(input_reshape)
    h1_gates      = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(input_reshape)
    h1_glu        = gated_liner_units(h1, h1_gates)
    d1            = downsample2d(h1_glu, 256, (3, 3), (2, 2))
    d2            = downsample2d(d1, 512, (3, 3), (2, 2))
    d3            = downsample2d(d2, 1024, (3, 3), (2, 2))
    d4            = downsample2d(d3, 1024, (1, 5), (1, 1))
    out           = Conv2D(1, (1, 3), strides=(1, 1), padding='same')(d4)
    model = Model(inputs=input_layer, outputs=out)
    return model

def build_212CNN_Generator(input_shape):
    '''
    2-1-2CNN(Generator)のKerasモデルをつくる

    Parameters
    ----------
    input_shape: Tuple[int, int]
        MCEPsを転置したshape
        例: (36, 128)
    
    Returns
    -------
    model: Model
        Kerasのモデル
    
    Note
    ----
    input_shapeは(36, n)とかが良い
    ちなみに(35, n)ではうまくいかない
    '''
    def gated_liner_units(layer_inputs, gates):
        act = Activation('sigmoid')(layer_inputs)
        act = Multiply()([layer_inputs, act])
        return act
    def downsample2d(layer_inputs, filters, kernel_size, strides):
        h1            = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm       = InstanceNormalization(epsilon=1e-06)(h1)
        h1_gates      = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm_gates = InstanceNormalization(epsilon=1e-06)(h1_gates)
        h1_glu        = gated_liner_units(h1_norm, h1_norm_gates)
        return h1_glu
    def residual1d(layer_inputs, filters, kernel_size, strides):
        h1            = Conv1D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm       = InstanceNormalization(epsilon=1e-06)(h1)
        h1_gates      = Conv1D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_norm_gates = InstanceNormalization(epsilon=1e-06)(h1_gates)
        h1_glu        = gated_liner_units(h1_norm, h1_norm_gates)
        h2            = Conv1D(filters//2, kernel_size, strides=strides, padding='same')(h1_glu)
        h2_norm       = InstanceNormalization(epsilon=1e-06)(h2)
        h3            = Add()([layer_inputs, h2_norm])
        return h3
    def upsample2d(layer_inputs, filters, kernel_size, strides):
        h1               = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_shuffle       = SubPixelUpscaling(scale_factor=2)(h1)
        h1_norm          = InstanceNormalization(epsilon=1e-06)(h1_shuffle)
        h1_gates         = Conv2D(filters, kernel_size, strides=strides, padding='same')(layer_inputs)
        h1_shuffle_gates = SubPixelUpscaling(scale_factor=2)(h1_gates)
        h1_norm_gates    = InstanceNormalization(epsilon=1e-06)(h1_shuffle_gates)
        h1_glu           = gated_liner_units(h1_norm, h1_norm_gates)
        return h1_glu
    input_layer   = Input(shape=input_shape)
    input_reshape = Reshape(input_shape+(1,))(input_layer)
    h1            = Conv2D(128, (5, 15), strides=(1, 1), padding='same')(input_reshape)
    h1_gates      = Conv2D(128, (5, 15), strides=(1, 1), padding='same')(input_reshape)
    h1_glu        = gated_liner_units(h1, h1_gates)
    d1            = downsample2d(h1_glu, 256, (5, 5), (2, 2))
    d2            = downsample2d(d1, 256, (5, 5), (2, 2))
    d3            = Reshape((-1, 2304))(d2)
    resh1         = Conv1D(256, 1, strides=1, padding='same')(d3)
    resh1_norm    = InstanceNormalization(epsilon=1e-06)(resh1)
    r1            = residual1d(resh1_norm, 512, 3, 1)
    r2            = residual1d(r1, 512, 3, 1)
    r3            = residual1d(r2, 512, 3, 1)
    r4            = residual1d(r3, 512, 3, 1)
    r5            = residual1d(r4, 512, 3, 1)
    r6            = residual1d(r5, 512, 3, 1)
    resh2         = Conv1D(2304, 1, strides=1, padding='same')(r6)
    resh2_norm    = InstanceNormalization(epsilon=1e-06)(resh2)
    resh3         = Reshape((9, -1, 256))(resh2_norm)
    u1            = upsample2d(resh3, 1024, 5, 1)
    u2            = upsample2d(u1, 512, 5, strides=1)
    conv_out      = Conv2D(1, (5, 15), strides=(1, 1), padding='same')(u2)
    out           = Reshape((-1, 128))(conv_out)
    out           = Activation('tanh')(out)
    model         = Model(inputs=input_layer, outputs=out)
    return model
