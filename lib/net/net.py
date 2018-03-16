from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

DEFAULT_PADDING = 'SAME'
BN_EPSILON = 0.001


class Net(object):
    """
    Base Net class
    """
    def __init__(self, net_conf):
        """
        common params: a params dict
        net_params: a params dict
        """
        self.pretrained_collection = []
        self.trainable_collection = []

    def load(self, data_path, session, saver,ignore_missiong=False):
        if data_path.endswith(".ckpt"):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            sess.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model "  \
                                + subkey + " to " + key)
                        except ValueError:
                            print ("ignore " + key)
                            if not ignore_missiong:
                                raise





    def make_variable(self, name, shape, initializer
        , trainable=True, pretrainable=True):
        var = tf.get_variable(name, shape
            , initializer=initializer
            , trainable=trainable)
        if pretrainable:
            self.pretrained_collection.append(var)
        if trainable:
            self.trainable_collection.append(var)

        return var

    def make_weight(self, name, shape, means=0.0
        , stddev=0.01, trainable=True, pretrainable=True):
        """
        create weights variable
        Args:
            name: variable_scope name
            shape: the shape of weights
            means: means of initializer
            stddev: stddev of initializer
            trainable: variables can be trained
        """
        initializer = tf.truncated_normal_initializer(means, stddev=stddev)
        weight = self.make_variable(name, shape, initializer
            ,trainable, pretrainable)
        return weight

    def make_biase(self, name, shape, init=0.0
        , trainable=True, pretrainable=True):
        """
        create  biases variable
        Args:
            name: variable_scope name
            shape: the shape of biases
            init: init value
            trainable: can be trained
        """
        initializer = tf.constant_initializer(init)
        bias = self.make_variable(name, shape, initializer
            , trainable, pretrainable)
        return bias


    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def conv(self, input, k_h, k_w, c_o, s_h, s_w
        , name, padding=DEFAULT_PADDING
        , group=1, trainable=True, pretrainable=True):
        """
        convolutional layerBN_EPSILON
        Args:
            input: 4-D tensor [batch_size, height, width, depth]
            name: variable_scope name
            k_h, k_w: kernel size
            c_o: depth of output
            s_h, s_w: stride size
            activiation: activiation after convolutional layer
            padding: mode of padding
            group: the number of gpus
            trainable: wheather variables can be trained
        """

        self.validate_padding(padding=padding)
        c_i = int(input.get_shape()[-1])
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k:tf.nn.conv2d(i, k, [1, s_h, s_w, 1]  \
            , padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_weight('weights'
                , [k_h, k_w, c_i / group, c_o]
                , trainable=trainable, pretrainable=pretrainable)
            biases = self.make_biase("biases", [c_o]
                , trainable=trainable, pretrainable=pretrainable)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k)     \
                    for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            conv = tf.nn.bias_add(conv, biases, name=scope.name)
            #conv = self.batch_norm(conv, True)
            conv = self.leaky_relu(conv)
            return conv


    def batch_normalization_layer(self, input, name="bn"
        , trainable=True, pretrainable=False):
        """
        Helper function to do batch normalization
        Args:
            input: 4D-tensor
            dimen: input
        """
        out_dim = int(input.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
            beta = self.make_variable('beta', [out_dim]
                , initializer=tf.constant_initializer(0.0)
                , trainable=trainable, pretrainable=pretrainable)
            gamma = self.make_variable('gamma', [out_dim]
                , initializer=tf.constant_initializer(1.0)
                , trainable=trainable, pretrainable=pretrainable)
            bn = tf.nn.batch_normalization(input, mean, variance
                , beta, gamma, BN_EPSILON)
            return bn

    def batch_norm(self, x, train, esp=1e-05, decay=0.9
        , affine=True, name="batchNorm"):
        params_shape = tf.shape(x)[-1:]
        moving_mean = self.make_variable('mean', params_shape
            , initializer=tf.zeros_initializer, trainable=False)
        movin_variance = tf.get_variable('variance', params_shape
            ,initializer=tf.ones_initializer, trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1]
                , name='moments')
            with tf.control_dependencies(
                [assign_moving_average(moving_mean, mean, decay)
                    , assign_moving_average(movin_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(train, mean_var_with_update
            , lambda: (moving_mean, movin_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape
                , initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape
                , initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance
                , beta, gamma, esp)
        else:
            x = tf.nn.batch_normalization(x, mean, variance
                , None, None, esp)
        return x

    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    def max_pool(self, input, k_h, k_w, s_h, s_w
        , name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        """
        Args:
            input: 4-D tensor, [batch_size, height, weidth, depth]
            ksize: kernel size, [k_height, k_width]
            s_h, s_w: strides, int32
        Return:
            output: 4-D tensor [batch_size, height / s_h, width / s_w, depth]
        """
        return tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1]
            , strides=[1, s_h, s_w, 1], padding=padding
            , name=name)

    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(
            input, depth_radius=radius, alpha=alpha
            , beta=beta, bias=bias, name=name)

    def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
        """
        if x > 0:
            return x
        else:
            return alpha * x
        Args:
            x: Tensor
            alpha: float
        Retur:
            y: Tensor
        """
        x = tf.cast(x, dtype=dtype)

        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x


    def fc(self, name, input, num_in, num_out
        , is_leaky=False, trainable=True, pretrainable=True):
        """
        Fully connected layer
        Args:
            name: variable_scope name
            input: [batch_size, ???]
            num_in: size of input , int32
            num_out: size of output, int32
            activiation: activiation after fully connected layer
            trainable: wheather the variables can be trained
        Return:
            output: 2-D tensor, [batch_size, num_out]
        """
        with tf.variable_scope(name) as scope:
            reshape = tf.reshape(input, [tf.shape(input)[0], -1])
            weight = self.make_weight("weights"
                , [num_in, num_out], trainable=trainable
                , pretrainable=pretrainable)
            bias = self.make_biase('biases', [num_out]
                , trainable=trainable, pretrainable=pretrainable)
            fc = tf.matmul(reshape, weight) + bias
            if is_leaky:
                fc = self.leaky_relu(fc)
            else:
                fc = tf.identity(fc)
            return fc

    def inference(self):
        raise NotImplementedError
