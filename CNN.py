import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import cPickle
import pickle
from time import time
from random import randint

from ConvPoolLayer import ConvPoolLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer


class CNN:
    def __init__(self,
                 mode=None,
                 params_path=None,
                 train_set=None, valid_set=None,
                 nkerns=(32, 8, 24), batch_size=500,
                 learning_rate=0.05,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nkerns = nkerns

        self.rng = np.random.RandomState(int(time()))
        self.theano_rng = RandomStreams(int(time()))

        tmp_params = [None] * 8
        if params_path:
            # load params
            f = open(params_path, 'rb')
            # layer3.params,layer2.params,layer1.params,layer0.params=pickle.load(f)
            tmp_params = pickle.load(f)

        if mode == 'train':
            self.train_set_x = theano.shared(np.asarray(train_set[0] / 255., dtype=theano.config.floatX), borrow=True)
            self.train_set_y = theano.shared(np.asarray(train_set[1], dtype=theano.config.floatX), borrow=True)
            self.train_set_y = T.cast(self.train_set_y, 'int32')

            self.valid_set_x = theano.shared(np.asarray(valid_set[0] / 255., dtype=theano.config.floatX), borrow=True)
            self.valid_set_y = theano.shared(np.asarray(valid_set[1], dtype=theano.config.floatX), borrow=True)
            self.valid_set_y = T.cast(self.valid_set_y, 'int32')

            print '------ build train model'

            x, layer0, layer1, layer2, layer3, local_params = self.build_layers(True, tmp_params, self.batch_size)
            print [id(param) for param in local_params]
            self.build_train_model(
                x,
                layer0, layer1, layer2, layer3,
                self.train_set_x, self.train_set_y
            )

            print '------ build valid model'

            x, layer0, layer1, layer2, layer3, local_params = self.build_layers(False, local_params, 500)
            self.params = local_params
            print [id(param) for param in self.params]
            self.build_valid_model(
                x,
                layer0, layer1, layer2, layer3,
                self.valid_set_x, self.valid_set_y)

        if mode == 'predict':
            x, layer0, layer1, layer2, layer3, local_params = self.build_layers(False, tmp_params, self.batch_size)
            self.build_predict_model(x, layer0, layer1, layer2, layer3)

    def build_layers(self, is_dropout, params, batch_size):
        # input images
        x = T.matrix('x')

        # (60,30) => (54,24) => (18,8)
        # output shape is (batch_size,nkerns[0],28,13)
        layer0_input = x.reshape((batch_size, 3, 60, 30))
        layer0 = ConvPoolLayer(
            self.rng,
            theano_rng=self.theano_rng,
            input=layer0_input,
            is_dropout=False,
            filter_shape=(self.nkerns[0], 3, 7, 7),
            image_shape=(batch_size, 3, 60, 30),
            poolsize=(3, 3),
            W=params[6], b=params[7]
        )
        # (18,8) => (14,4) => (7,4)
        # output shape is (batch_size,nkerns[1],12,9)
        layer1_input = layer0.output
        layer1 = ConvPoolLayer(
            self.rng,
            theano_rng=self.theano_rng,
            input=layer1_input,
            is_dropout=False,
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            image_shape=(batch_size, self.nkerns[0], 18, 8),
            poolsize=(2, 1),
            W=params[4], b=params[5]
        )
        # a fully-connected sigmoidal layer
        # (nkerns[0]*9*4+nkerns[1]*7*4, n_out)
        pooled_out = downsample.max_pool_2d(
            input=layer0.output,
            ds=(2, 2),
            ignore_border=True
        )
        layer2_input = T.horizontal_stack(
            pooled_out.flatten(2), layer1.output.flatten(2)
        )
        layer2 = HiddenLayer(
            self.rng,
            theano_rng=self.theano_rng,
            input=layer2_input,
            is_dropout=is_dropout,
            n_in=self.nkerns[0] * 9 * 4 + self.nkerns[1] * 7 * 4,
            n_out=self.nkerns[2],
            W=params[2], b=params[3]
        )
        # output classify result
        layer3_input = layer2.output
        layer3 = OutputLayer(
            theano_rng=self.theano_rng,
            input=layer3_input,
            is_dropout=is_dropout,
            n_in=self.nkerns[2], n_out=2,
            W=params[0], b=params[1]
        )
        final_params = layer3.params + layer2.params + layer1.params + layer0.params
        return x, layer0, layer1, layer2, layer3, final_params

    def build_train_model(self, x,
                          layer0, layer1, layer2, layer3,
                          train_set_x, train_set_y):
        index = T.lscalar()
        # input images
        local_x = x
        # labels
        y = T.ivector('y')
        # learning rate
        learning_rate = T.fscalar()
        # minimize the cost
        cost = layer3.negative_log_likelihood(y)
        # create the params and grad list
        params = layer3.params + layer2.params + layer1.params + layer0.params
        print [id(param) for param in params]
        grads = T.grad(cost, params)
        # train model
        self.train_model = theano.function(
            [index, learning_rate],
            # [cost,
            # layer0.input,layer0.output,layer1.input,layer1.output,
            # layer2.input,layer2.output,layer3.input,layer3.p_y_given_x,
            # layer3.errors(y)],
            cost,
            updates=[
                (param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)
            ],
            givens={
                local_x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )

    def build_valid_model(self, x, layer0, layer1, layer2, layer3, valid_set_x, valid_set_y):
        index = T.lscalar()
        # input images
        local_x = x
        # labels
        y = T.ivector('y')
        # compute the error of test and valid set
        self.validate_model = theano.function(
            [index],
            # [layer3.errors(y),
            # layer0.input,layer0.output,layer1.input,layer1.output,
            #  layer2.input,layer2.output,layer3.input,layer3.p_y_given_x],
            layer3.errors(y),
            givens={
                local_x: valid_set_x[index * 500: (index + 1) * 500],
                y: valid_set_y[index * 500: (index + 1) * 500]
            }
        )

    def build_predict_model(self, x, layer0, layer1, layer2, layer3):
        # input images
        local_x = x
        # classify result
        y_pred = layer3.y_pred
        self.predict_model = theano.function(
            [local_x], y_pred, allow_input_downcast=True
        )


    def train(self):
        # compute number of batches
        num_train = self.train_set_x.get_value(borrow=True).shape[0]
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = 5

        # look as this many examples regardless
        patience = num_train / 5
        # wait this much longer when a new best is found
        patience_increase = num_train / 100

        validation_losses = [self.validate_model(i) for i
                             in range(n_valid_batches)]
        best_validation_loss = validation_losses
        print '  init loss :', \
            [int(loss * 10000.) / 100. for loss in validation_losses], \
            np.mean(validation_losses) * 100, '%'
        best_epoch = 0
        epoch = 0

        while patience > epoch:
            epoch += 1
            old_params = [param.get_value() for param in self.params]

            for num_index in range(n_train_batches):
                local_index = randint(0, n_train_batches - 1)
                ret = self.train_model(local_index, self.learning_rate)

            # compute zero-one loss on validation set
            validation_losses = [self.validate_model(i) for i
                                 in xrange(n_valid_batches)]
            print '  epoch', epoch, 'validation error ', \
                [int(loss * 10000.) / 100. for loss in validation_losses], \
                np.mean(validation_losses) * 100., '%'

            # if we got the best validation score until now
            if (validation_losses[0] < best_validation_loss[0] and
                    sum(validation_losses[1:]) < sum(best_validation_loss[1:])) or \
                    (sum(validation_losses)<0.97*sum(best_validation_loss)):
                # save best validation score and iteration number
                best_validation_loss = validation_losses
                best_epoch = epoch

                print '      save params'
                f = open('params_'+str(int(best_validation_loss[0]*10000.)/100.)+
                         '_'+str(int(sum(best_validation_loss[1:])*10000./4.)/100.)+
                         '_'+str(int(sum(best_validation_loss)*10000./5.)/100.), 'wb')
                pickle.dump(self.params, f)
                f.close()

                patience += patience_increase

            new_params = [param.get_value() for param in self.params]
            total_change = 0.
            for i in range(len(new_params)):
                total_change += np.sum(np.abs(old_params[i] - new_params[i]))
            print '    total params change', total_change
            if epoch % 200 == 0:
                self.learning_rate *= 0.98
                print '    change learning rate to', self.learning_rate

        print 'Optimization complete.'
        print 'Best validation score of', np.mean(best_validation_loss) * 100., '%', \
            'obtained at iteration', best_epoch + 1

    def predict(self, x):
        return self.predict_model(x)


if __name__ == '__main__':
    f = open('dataset/train_set', mode='rb')
    train_set = cPickle.load(f)
    f.close()

    f = open('dataset/valid_set', mode='rb')
    valid_set = cPickle.load(f)
    f.close()

    print train_set[0].shape
    print valid_set[0].shape

    cnn = CNN(
        'train',
        'params',
        train_set,
        valid_set,
        batch_size=5000
    )
    cnn.train()
