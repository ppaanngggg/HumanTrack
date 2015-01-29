import numpy as np
import theano
import theano.tensor as T

class OutputLayer:
    def __init__(self,theano_rng,
                 input,is_dropout,
                 n_in,n_out,
                 W=None,b=None):

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
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        # init bias
        if b:
            self.b=b
        else:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        # Return the mean of the negative log-likelihood of the prediction
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))