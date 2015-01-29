import theano
import theano.tensor as T
import numpy as np

class HiddenLayer:
    def __init__(self,rng,theano_rng,
                 input,is_dropout,
                 n_in,n_out,
                 W=None,b=None):
        """
        hidden layer of MLP

        :param rng: random number generator
        :param input: dmatrix (n_examples, n_in)
        :param n_in: dimensionality of input
        :param n_out: dimensionality of output
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
            W_values=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            self.W=theano.shared(value=W_values,name='W',borrow=True)

        # init bias
        b_values=np.zeros((n_out,),dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,name='b',borrow=True)
        # tanh(XW+b)
        self.output=T.tanh(T.dot(self.input,self.W)+self.b)
        self.params = [self.W, self.b]

