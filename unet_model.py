""" Trains U-Net
"""

import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import (Input, 
                          merge, 
                          Convolution2D, 
                          MaxPooling2D, 
                          Deconvolution2D, 
                          UpSampling2D,
                          Dropout, 
                          Cropping2D,
                          Lambda,
                          Activation,
                          merge)
from keras.layers.advanced_activations import PReLU
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler)
from keras import backend as K
from custom_image import (ImageDataGenerator, 
                          standardize,
                          random_transform)
from custom_layers import (CroppingChannels,
                          DepthSoftmax)
from net_utils import *


def downblock_seg(input_net, nb_filters, init):
    
    db = BatchNormalization()(input_net)
    db = PReLU()(db)
    db = Convolution2D(nb_filters, 3, 3, border_mode='same', init=init, subsample=(2,2))(db)

    db_2 = BatchNormalization()(db)
    db_2 = PReLU()(db_2)
    db_2 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=init)(db_2)

    db_2 = merge([db, db_2], mode='sum')

    return db_2

def upblock_seg(input_net, nb_filters, deconv_output, init='he_normal'):

    sb = Convolution2D(nb_filters, 1, 1, init=init)(input_net)
    sb = BatchNormalization()(sb)
    sb = PReLU()(sb)

    sb = Deconvolution2D(nb_filters, 2, 2,
        output_shape=deconv_output,
        subsample=(2, 2),
        init=init)(sb)

    sb = BatchNormalization()(sb)
    sb = PReLU()(sb)

    return sb


def merge_seg(input_net, skip_net, nb_filters, init='he_norma'):

    ms = merge([input_net, skip_net], mode='concat', concat_axis=3)

    ms = Convolution2D(nb_filters, 3, 3, border_mode='same', init=init)(ms)
    ms = BatchNormalization()(ms)
    ms = PReLU()(ms)

    return ms

def downblock_vgg(input_net, nb_filters, init):
    db = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='valid', init=init)(input_net)
    db = BatchNormalization()(db)
    db = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='valid', init=init)(db)
    db = BatchNormalization()(db)
    db_pooled = MaxPooling2D(pool_size=(2, 2))(db)

    return (db, db_pooled)

def upblock_vgg(input_net, 
                skip_net=None,
                deconv_output=None, 
                crop_width=None, 
                nb_filters=64, 
                init='he_normal',
                name=None):
    """Provides a vgg-like up-block of model steps
    Args:
        input_net:  previous model output which is fed in
        skip_net:   previous model output which is skips across and is appended to filters,
            if None then ignored
        deconv_output: output of the transposed convolution layer
        crop_width: amount to crop skip_net by to match with deconv_output
        nb_filters: number of filters
        init: weight initialization
    Output:
        ub: complete model which describes up-block
    """
    if name:
        name_crop2D = name + '_crop2D'
        name_deconv = name + '_deconv2D'
        name_bn_1 = name + '_bn_1'
        name_conv2D_1 = name + '_conv2D_1'
        name_bn_2 = name + '_bn_2'
        name_conv2D_2 = name + '_conv2d_2'
        name_bn_3 = name + '_bn_3'

    if skip_net is not None:
        ub = Cropping2D(cropping=((crop_width, crop_width), (crop_width, crop_width)),
                name=name_crop2D)(skip_net)
        deconv = Deconvolution2D(nb_filters, 2, 2,
                output_shape=deconv_output,
                subsample=(2, 2),
                init=init,
                name=name_deconv)(input_net)
        ub = merge([deconv, ub], mode='concat', concat_axis=3)
    else:
        ub = Deconvolution2D(nb_filters, 2, 2,
                output_shape=deconv_output,
                subsample=(2, 2),
                init=init,
                name=name_deconv)(input_net)
    
    ub = BatchNormalization(name=name_bn_1)(ub)
    ub = Convolution2D(nb_filters, 3, 3,
                activation='relu',
                border_mode='valid',
                init=init,
                name=name_conv2D_1)(ub)
    ub = BatchNormalization(name=name_bn_2)(ub)
    ub = Convolution2D(nb_filters, 3, 3,
                activation='relu',
                border_mode='valid',
                init=init,
                name=name_conv2D_2)(ub)
    ub = BatchNormalization(name=name_bn_3)(ub)

    return ub

def get_aenet(batch_size=1,
              rows=572,
              cols=572,
              num_channels=20,
              nb_filters=64,
              loss='jaccard',
              net_type='vgg',
              init='he_normal',
              lr=1e-3):
    """Returns keras model of a VGG-like autoencoder for a first pass.
    Args:
        batch_size: batch_size of samples
        rows: rows of image
        cols: cols of image
        num_channels: number channels in input images
        lr: learning rate of optimizer
        nb_filters: base number of filters used (gets increased in successive
            layers)
        loss: One of 'jaccard', 'logjaccard', or 'logloss', gives choice of 
            objective fn
        init: weight initialization, default 'he_normal'
    Output:
        model: CNN model
    """
    # Tensorflow does not allow using None for batch_size in deconv layers
    # we can input it symbolically or we can just hard-code batch size
    # see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/vf8eH9YMwVA
    # and https://github.com/fchollet/keras/issues/3478

    # Tensorflow does not allow using None for batch_size in deconv layers
    # we can input it symbolically or we can just hard-code batch size
    # see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/vf8eH9YMwVA
    # and https://github.com/fchollet/keras/issues/3478
    
    
    inputs = Input((rows, cols, num_channels))

    # keeps only the 4 high-res channels (RGB and P image)
    hires_channels = CroppingChannels(cropping=(0,17))(inputs)

    (conv1_skip, conv1) = downblock_vgg(hires_channels, nb_filters, init)
    (conv2_skip, conv2) = downblock_vgg(conv1, nb_filters*2, init)
    (conv3_skip, conv3) = downblock_vgg(conv2, nb_filters*4, init)
    (conv4_skip, conv4) = downblock_vgg(conv3, nb_filters*8, init)
    (conv5_skip, conv5) = downblock_vgg(conv4, nb_filters*16, init)

    conv6 = upblock_vgg(conv5_skip,  
                        deconv_output=[batch_size, 56, 56, nb_filters*8], 
                        crop_width=4, 
                        nb_filters=nb_filters*8, 
                        init=init)
    conv7 = upblock_vgg(conv6,  
                        deconv_output=[batch_size, 104, 104, nb_filters*4], 
                        crop_width=16, 
                        nb_filters=nb_filters*4, 
                        init=init)
    conv8 = upblock_vgg(conv7,  
                        deconv_output=[batch_size, 200, 200, nb_filters*2], 
                        crop_width=40, 
                        nb_filters=nb_filters*2, 
                        init=init)
    conv9 = upblock_vgg(conv8,  
                        deconv_output=[batch_size, 392, 392, nb_filters], 
                        crop_width=88, 
                        nb_filters=nb_filters, 
                        init=init)
    # this is explicity an auto-encoder, so we make sure that they are the same
    conv10 = Convolution2D(3, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')
    return model



def get_unet(batch_size=1, 
             shape_in = (572, 572, 20),
             shape_out = (388, 388, 11),
             lr=1e-3, 
             nb_filters=64,
             loss='jaccard',
             net_type='vgg',
             init='he_normal',
             ae_weights=None,
             classifier_weights=None,
             mask_weight=None,
             end_activation='softmax'):
    """Returns keras model of a small U-Net CNN for prototyping. See:
    https://arxiv.org/pdf/1505.04597v1.pdf
    Args:
        batch_size: batch_size of samples
        rows: rows of image
        cols: cols of image
        num_channels: number channels in input images
        lr: learning rate of optimizer
        nb_filters: base number of filters used (gets increased in successive
            layers)
        loss: One of 'jaccard', 'logjaccard', or 'logloss', gives choice of 
            objective fn
        init: weight initialization, default 'he_normal'
        ae_weights: filepath to pre-trained weights of autoencoder
        classifier_weights: filepath to pretrained weights of classifier
            (the 'encoding' portion of the net)
    Output:
        model: CNN model
    """

    # Tensorflow does not allow using None for batch_size in deconv layers
    # we can input it symbolically or we can just hard-code batch size
    # see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/vf8eH9YMwVA
    # and https://github.com/fchollet/keras/issues/3478

    # Tensorflow does not allow using None for batch_size in deconv layers
    # we can input it symbolically or we can just hard-code batch size
    # see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/vf8eH9YMwVA
    # and https://github.com/fchollet/keras/issues/3478

    # make model from scratch if not passing in weights
    if classifier_weights is None:
       
        inputs = Input(shape_in)

        # keeps only the 3 high-res channels (RGB and P image)
        hires_channels = CroppingChannels((0,17))(inputs)

        (conv1_skip, conv1) = downblock_vgg(inputs, nb_filters, init)
        (conv2_skip, conv2) = downblock_vgg(conv1, nb_filters*2, init)
        (conv3_skip, conv3) = downblock_vgg(conv2, nb_filters*4, init)
        (conv4_skip, conv4) = downblock_vgg(conv3, nb_filters*8, init)
        (conv5_skip, conv5) = downblock_vgg(conv4, nb_filters*16, init)

    # If we pass in weights, we build out model around the pretrained weights
    else:
        vgg = load_model(classifier_weights,
                         custom_objects={'CroppingChannels': CroppingChannels})

        inputs = vgg.get_layer('input_1').output

        conv1_skip = vgg.get_layer('batchnormalization_2').output
        conv2_skip = vgg.get_layer('batchnormalization_4').output
        conv3_skip = vgg.get_layer('batchnormalization_6').output
        conv4_skip = vgg.get_layer('batchnormalization_8').output
        conv5_skip = vgg.get_layer('batchnormalization_10').output

        conv1 = vgg.get_layer('maxpooling2d_1').output
        conv2 = vgg.get_layer('maxpooling2d_2').output
        conv3 = vgg.get_layer('maxpooling2d_3').output
        conv4 = vgg.get_layer('maxpooling2d_4').output


    conv6 = upblock_vgg(conv5_skip, conv4_skip, 
                        deconv_output=[batch_size, 56, 56, nb_filters*8], 
                        crop_width=4, 
                        nb_filters=nb_filters*8, 
                        init=init,
                        name='ub_vgg_1')
    conv7 = upblock_vgg(conv6, conv3_skip, 
                        deconv_output=[batch_size, 104, 104, nb_filters*4], 
                        crop_width=16, 
                        nb_filters=nb_filters*4, 
                        init=init,
                        name='ub_vgg_2')
    conv8 = upblock_vgg(conv7, conv2_skip,  
                        deconv_output=[batch_size, 200, 200, nb_filters*2], 
                        crop_width=40, 
                        nb_filters=nb_filters*2, 
                        init=init,
                        name='ub_vgg_3')
    conv9 = upblock_vgg(conv8, conv1_skip, 
                        deconv_output=[batch_size, 392, 392, nb_filters], 
                        crop_width=88, 
                        nb_filters=nb_filters, 
                        init=init,
                        name='ub_vgg_4')

    # adds in the other 16 channels, and crops them appropriately
    # Done like this, we are essentially using the RGB channels for feature
    # finding, and then adding in quasi-pixel-based classification for the
    # final segmentation map
    lowres_channels = Cropping2D(cropping=((92,92), (92,92)),
            name='lowres_spatialcrop')(inputs)
    lowres_channels = CroppingChannels(cropping=(3,0), 
            name='lowres_channelcrop')(lowres_channels)
    # appends them to current stack
    conv10 = merge([conv9, lowres_channels], mode='concat', concat_axis=3)
    conv10 = Convolution2D(nb_filters*2, 1, 1, 
            name='conv10_conv2d_1')(conv10)
    conv10 = Convolution2D(nb_filters*2, 1, 1,
            name='conv10_conv2d_2')(conv10)
    
    conv11 = Convolution2D(shape_out[2], 1, 1,
            name='conv11_conv2d_1')(conv10)
    if end_activation == 'softmax':
        conv11 = DepthSoftmax()(conv11)
    elif end_activation == 'sigmoid':
        conv11 = Activation('sigmoid')(conv11)

    model = Model(input=inputs, output=conv11)

    if mask_weight is not None:
        if loss == 'jaccard':
            loss_fn = jaccard_coef_loss
        elif loss == 'logloss':
            loss_fn = pixelwise_logloss
        elif loss == 'logjaccard':
            loss_fn = jaccard_coef_logloss
        else:
            raise ValueError('The loss must be one of \'jaccard\',' 
                            +'\'logjaccard\', or \'logloss\'')
    else:
        # if the user passes a list of mask weights, we use a modified
        # loss function
        loss_fn = partial(jaccard_coef_loss_weighted, weights=mask_weight)
    

    model.compile(optimizer=Adam(lr=lr),
            loss=loss_fn,
            metrics=[jaccard_coef])
    return model
    
def get_classnet(batch_size=8, 
                   shape_in=(572, 572, 20),
                   lr=1e-3, 
                   nb_filters=64,
                   init='he_normal'):
    """Returns keras model of a small U-Net CNN for prototyping. See:
    https://arxiv.org/pdf/1505.04597v1.pdf
    Args:
    Output:
        model: CNN model
    """
    # Tensorflow does not allow using None for batch_size in deconv layers
    # we can input it symbolically or we can just hard-code batch size
    # see https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/vf8eH9YMwVA
    # and https://github.com/fchollet/keras/issues/3478
    
    
    inputs = Input(shape_in)

    # keeps only the 4 high-res channels (RGB and P image)
    hires_channels = CroppingChannels(cropping=(0,17))(inputs)

    (conv1_skip, conv1) = downblock_vgg(hires_channels, nb_filters, init)
    (conv2_skip, conv2) = downblock_vgg(conv1, nb_filters*2, init)
    (conv3_skip, conv3) = downblock_vgg(conv2, nb_filters*4, init,)
    (conv4_skip, conv4) = downblock_vgg(conv3, nb_filters*8, init)
    (conv5_skip, conv5) = downblock_vgg(conv4, nb_filters*16, init)

    
    conv6 = Convolution2D(nb_filters*16, 1, 1, activation='relu', init=init)(conv5_skip)
    conv6 = Convolution2D(11, 1, 1, activation='relu', init=init)(conv6)
    conv6 = GlobalAveragePooling2D()(conv6)
    conv6 = Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv6)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def get_segnet(batch_size=1, 
             shape_in = (512, 512, 20),
             shape_out = (512, 512, 11),
             lr=5e-5, 
             nb_filters=64,
             loss='jaccard',
             init='he_normal',
             end_activation='softmax'):
    """Returns keras model of a sophisticated u-net type for medical
    segmentation, see:
    https://arxiv.org/pdf/1701.03056.pdf
    Args:
        batch_size: batch_size of samples
        shape_in: input shape
        shape_out: output shape
        num_channels: number channels in input images
        lr: learning rate of optimizer
        nb_filters: base number of filters used (gets increased in successive
            layers)
        loss: One of 'jaccard', 'logjaccard', or 'logloss', gives choice of 
            objective fn
        init: weight initialization, default 'he_normal'
    Output:
        model: CNN model
    """

    inputs = Input(shape_in)

    # beginning convolutions
    conv_1 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=init)(inputs)
    conv_2 = Convolution2D(nb_filters, 3, 3, border_mode='same', init=init)(conv_1)

    # down-sampling
    db_1 = downblock_seg(conv_2, nb_filters, init)
    db_2 = downblock_seg(db_1, nb_filters*2, init)
    db_3 = downblock_seg(db_2, nb_filters*4, init)

    # up-sampling
    deconv_output = [batch_size, shape_in[0]//4, shape_in[1]//4, nb_filters*4]
    ub_1 = upblock_seg(db_3, nb_filters*4, deconv_output=deconv_output, init=init)
    ms_1 = merge_seg(ub_1, db_2, nb_filters*2, init)

    deconv_output = [batch_size, shape_in[0]//2, shape_in[1]//2, nb_filters*2]
    ub_2 = upblock_seg(ms_1, nb_filters*2, deconv_output=deconv_output, init=init)
    ms_2 = merge_seg(ub_2, db_1, nb_filters*2, init)

    deconv_output = [batch_size, shape_in[0], shape_in[1], nb_filters]
    ub_3 = upblock_seg(ms_2, nb_filters, deconv_output=deconv_output, init=init)
    ms_3 = merge_seg(ub_3, conv_2, nb_filters, init)

    # bringing outputs of different layers together in map
    final_1 = Convolution2D(shape_out[2], 1, 1, init=init)(ms_1)
    final_1 = UpSampling2D(size=(2,2))(final_1)
    
    final_2 = Convolution2D(shape_out[2], 1, 1, init=init)(ms_2)
    final_2 = merge([final_2, final_1], mode='sum')
    final_2 = UpSampling2D(size=(2,2))(final_2)

    final_3 = Convolution2D(shape_out[2], 1, 1, init=init)(ms_3)
    final_3 = merge([final_3, final_2], mode='sum')
    
    #activated = Activation('softmax')(final_3)
    if end_activation='softmax':
        activated = DepthSoftmax()(final_3)
    else:
        activated = Activation('sigmoid')(final_3)

    model = Model(input=inputs, output=activated)

    loss_fn = jaccard_coef_loss
    model.compile(optimizer=Adam(lr=lr),
        loss=loss_fn,
        metrics=[jaccard_coef])

    return model
