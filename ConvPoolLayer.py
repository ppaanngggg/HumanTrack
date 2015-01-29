import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy as np

class ConvPoolLayer:
    def __init__(self,rng,theano_rng,
                 input,is_dropout,
                 filter_shape,image_shape,poolsize=(2,2),
                 W=None,b=None):
        """
        conv and pool layer

        :param rng: random number generator
        :param input: image tensor, of shape image_shape
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :param poolsize: tuple or list of length 2
        :return:
        """

        if is_dropout==True:
            self.input=input*2*theano_rng.binomial(
                size=input.shape
            ).astype('float32')
        else:
            self.input=input

        # init weights
        if W:
            self.W=W
        else:
            # feature maps * filter height * filter width
            fan_in = np.prod(filter_shape[1:])
            # "num output feature maps * filter height * filter width" / pooling size
            fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)
            # init weights randomly
            W_bound = np.sqrt(6./(fan_in+fan_out))
            self.W=theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        # init bias
        if b:
            self.b=b
        else:
            b_values=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
            self.b=theano.shared(value=b_values,borrow=True)

        # convolve input feature with filters
        conv_out=conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        #downsample
        pooled_out=downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # tanh( (conv and pool) +b)
        self.output=T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))

        # store params
        self.params=[self.W,self.b]

