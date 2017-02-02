from keras import backend as K
from keras import activations
from keras import initializations
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils.np_utils import conv_output_length
from keras.utils.np_utils import conv_input_length

class CroppingChannels(Layer):
    """CroppingChannels

    Crops the channels from a 4D input tensor, instead of the spatial
    coordinates as in e.g. Cropping2D

    # Input shape
        4D Tensor with shape: 
        (samples, depth_axis_to_crop, first_axis, second_axis)

    # Output shape
        4D Tensor with shape:
        (samples, depth_cropped_axis, first_axis, second_axis)

    # Arguments
        cropping: tuple of int (length 2). How many units should be trimmed
        off beginning and end of depth dimension

    """

    def __init__(self, cropping=(0, 0), dim_ordering='default', **kwargs):
        super(CroppingChannels, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.cropping = tuple(cropping)
        assert len(self.cropping) == 2, 'cropping must be a tuple length of 2'
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1] - self.cropping[0] - self.cropping[1],
                    input_shape[2],
                    input_shape[3])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3] - self.cropping[0] - self.cropping[1])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.dim_ordering == 'th':
            return x[:,
                     self.cropping[0]:input_shape[1]-self.cropping[1],
                     :,
                     :]
        elif self.dim_ordering == 'tf':
            return x[:,
                     :,
                     :,
                     self.cropping[0]:input_shape[3]-self.cropping[1]]
                     

    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(CroppingChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthSoftmax(Layer):
    """DepthSoftmax

    Performs a pixel-wise softmax computation on the model

    # Input shape
        4D Tensor with shape: 
        (samples, depth_axis, first_axis, second_axis)

    # Output shape
        Same as input

    """
    def __init__(self, dim_ordering='default', **kwargs):
        super(DepthSoftmax, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    def call(self, x, mask=None):
        ndim = K.ndim(x)
        if ndim == 4:
            e = K.exp(x - K.max(x, axis=-1, keepdims=True))
            s = K.sum(e, axis=-1, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply DepthSoftmax to'
                              'a tensor that is not 4D. '
                              'Here, ndim=' + str(ndim))
                     
