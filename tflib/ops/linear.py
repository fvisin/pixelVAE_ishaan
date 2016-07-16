import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None
        ):
    """
    initialization: None, `lecun`, `he`, `orthogonal`, `("uniform", range)`
    """

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    if initialization == 'lecun' or \
        (initialization == None and input_dim != output_dim):

        weight_values = uniform(
            np.sqrt(1./input_dim),
            (input_dim, output_dim)
        )

    elif initialization == 'glorot':

        weight_values = uniform(
            np.sqrt(2./(input_dim+output_dim)),
            (input_dim, output_dim)
        )

    elif initialization == 'he':

        weight_values = uniform(
            np.sqrt(2./input_dim),
            (input_dim, output_dim)
        )

    elif initialization == 'glorot_he':

        weight_values = uniform(
            np.sqrt(4./(input_dim+output_dim)),
            (input_dim, output_dim)
        )

    elif initialization == 'orthogonal' or \
        (initialization == None and input_dim == output_dim):
        
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
             # TODO: why normal and not uniform?
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype('float32')
        weight_values = sample((input_dim, output_dim))
    
    elif initialization[0] == 'uniform':
    
        weight_values = np.random.uniform(
            low=-initialization[1],
            high=initialization[1],
            size=(input_dim, output_dim)
        ).astype('float32')

    else:

        raise Exception('Invalid initialization!')

    weight = lib.param(
        name + '.W',
        weight_values
    )

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.linalg.norm(weight_values, axis=0)

        target_norms = lib.param(
            name + '.g',
            norm_values
        )

        norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
        weight = weight * (target_norms / norms)

    result = tf.matmul(inputs, weight)

    if biases:
        result = tf.nn.bias_add(
            result,
            lib.param(
                name + '.b',
                np.zeros((output_dim,), dtype='float32')
            )
        )

    return result