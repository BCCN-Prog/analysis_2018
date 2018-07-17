import tensorflow as tf
from tensorflow.contrib.rnn import Conv1DLSTMCell, Conv2DLSTMCell, Conv3DLSTMCell, ConvLSTMCell, MultiRNNCell
import numpy as np

class ConvLSTMWeather( object ):

    def __init__( self, batch_size, seq_len, time_span, space_size, num_vars, encoder_dim_expansion=8, kernel_shape=None, temporal_conv=True, spatial_conv=False ):

        self.batch_size = batch_size
        self.num_vars = num_vars
        self.space_size = space_size
        self.seq_len = seq_len

        if not temporal_conv and not spatial_conv:
            raise Exception( 'Why would you use a ConvLSTM if you do not convolve anything?' )
        
        if kernel_shape is None:
            if temporal_conv or spatial_conv:
                cl_cell = Conv1DLSTMCell
                if temporal_conv:
                    kernel_shape = [ 9, 1 ] # 9 hours
                else:
                    kernel_shape = [ 1, 3 ] # 3 stations

            else:
                cl_cell = Conv2DLSTMCell
                kernel_shape = [ 9, 3 ]

        input_shape1 = [ time_span, space_size, num_vars ]
        output_shape1 = num_vars * encoder_dim_expansion

        self.enc1 = Conv2DLSTMCell(
                input_shape=input_shape1,
                output_channels=output_shape1,
                kernel_shape=kernel_shape )

        input_shape2 = [ time_span, space_size, output_shape1 ]

        self.enc2 = Conv2DLSTMCell(
                input_shape=input_shape2,
                output_channels=num_vars,
                kernel_shape=[3,1] )

        self.enc = tf.contrib.rnn.MultiRNNCell( [self.enc1, self.enc2] )

        self.dec = Conv2DLSTMCell(
                input_shape=input_shape1,
                output_channels=num_vars,
                kernel_shape=[1,1] )

        self.input = tf.placeholder( tf.float32, ( None, seq_len, time_span, space_size, num_vars ) )
        self.labels = tf.placeholder( tf.float32, ( None, seq_len, time_span, space_size, num_vars ) )

        enc_output, self.enc_state = tf.nn.dynamic_rnn( self.enc, self.input, time_major=False,
                dtype=tf.float32#, initial_state=self.enc.zero_state( batch_size, tf.float32 )
                )

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper( enc_output, [seq_len] )
        #    decoder_emb_inp, decoder_lengths, time_major=True)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.dec, helper, self.enc_state[1] )
        # Dynamic decoding
        # returns: (final_outputs, final_state, final_sequence_lengths)
        self.outputs, self.state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = self.outputs.rnn_output

        self.loss = tf.losses.mean_squared_error(
                self.labels,
                logits )

        self.cost = tf.reduce_sum( self.loss )

        self.train_step = tf.train.AdamOptimizer( 1e-4 ).minimize( self.cost )

        self.sess = tf.Session()

    def train( self, train_data, train_labels, num_steps ):
        self.sess.run( tf.global_variables_initializer() )
        for i in range( num_steps ):
            for j in range( train_data.shape[0] ):
                self.train_step.run(
                        feed_dict={
                            self.input : train_data[j:j+1],
                            self.labels : train_labels[j:j+1] },
                        session=self.sess )

                if i % 100 == 0 and j % 100 == 0:
                    cur_cost = self.cost.eval(
                            feed_dict={
                                self.input : train_data[j:j+1],
                                self.labels : train_labels[j:j+1]},
                            session=self.sess )
                    print( 'Step {}. Cost: {}'.format( i, cur_cost ) )


    def infer( self, input_data ):
        return self.state[-1][0].eval( feed_dict={self.input : input_data}, session=self.sess )
        for i in range( num_steps - 1 ):
            output = self.state[0].eval( feed_dict={self.input : output}, session=self.sess )
        return output

def gen_sine( freq, size ):
    return np.sin( np.arange( 0, size * 2 * np.pi / freq, 2 * np.pi / freq ) )

