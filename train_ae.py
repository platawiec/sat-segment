import numpy as np
import sys
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, Deconvolution2D, Dropout, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from custom_image import ImageDataGenerator, standardize, random_transform, random_crop, center_crop
from unet_model import get_aenet
from generator_utils import get_ae_generators
import tensorflow as tf

def main():
    #parser = OptionParser()
    #parser.add_option("-s", "--split_random",
    #                  action='store',
    #                  type='int',
    #                  dest='split_random',
    #                  default=1)

    #options, _ = parser.parse_args()
    #split_random = options.split_random

    batch_size = 1
    nb_worker = 4

    model = get_aenet(batch_size=batch_size,
                       lr=1e-3)
    

    #learner = Leaner(model)

    train_generator = get_ae_generators(batch_size=1,
                                                        augment=True,
                                                        nb_iter=50,
                                                        preload=True)

    weight_filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'

    checkpointer = ModelCheckpoint(filepath=weight_filepath, verbose=1, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3)


    model.fit_generator(
            train_generator,
            samples_per_epoch=5000,
            nb_epoch=20,
            validation_data=train_generator,
            nb_val_samples=32,
            callbacks=[checkpointer, reduce_lr],
            nb_worker=nb_worker)


    model.save('aenet.h5')
    model.save_weights('aenet_weights.h5')



if __name__ == '__main__':
    sys.exit(main())
