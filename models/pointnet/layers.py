from torch import nn

class Conv1D(nn.Module):
    def __init__(self):
        super(Conv1D, self).__init__()

    def forward(self, x):
        num_in_channels = inputs.get_shape()[-1].value



def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

    Args:
      inputs: 3-D tensor variable BxLxC
      num_output_channels: int
      kernel_size: int
      scope: string
      stride: int
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
