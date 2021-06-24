import keras
from keras import backend as K


class peel_the_layer(keras.Model):
    def __init__(self, return_sequences=False):
        self.return_sequences = return_sequences
        super(peel_the_layer, self).__init__()

    def build(self, input_shape):
        units=1

        self.w=self.add_weight(name="att_weights", shape=(input_shape[-1], units), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[-2], units), initializer="zeros")
        super(peel_the_layer,self).build(input_shape)

    def call(self, x):
        ##x is the input tensor..each word that needs to be attended to
        ##Below is the main processing done during training
        ##K is the Keras Backend import
        e = K.tanh(K.dot(x,self.w)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        ##return the ouputs. 'a' is the set of attention weights
        ##the second variable is the 'attention adjusted o/p state' or context
        if self.return_sequences:
            return a, output
        else:
            return a, K.sum(output, axis=1)
