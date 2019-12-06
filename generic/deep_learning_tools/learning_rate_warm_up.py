from tensorflow import keras
from tensorflow.keras import backend as K


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """
    Warm up learning rate scheduler

     Using too large learning rate may result in numerical instability especially at the very beginning of the training,
     where parameters are randomly initialized. The warm up strategy increases the learning rate from 0 to the initial
     learning rate linearly during the initial N epochs or m batches. Even though Keras came with the
     LearningRateScheduler capable of updating the learning rate for each training epoch, to achieve finer updates for
     each batch, here is how you can implement a custom Keras callback to do that.
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warm up learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('Batch {}: WarmUpLearningRateScheduler setting learning rate to {}.'.format(
                    self.batch_count + 1, lr))