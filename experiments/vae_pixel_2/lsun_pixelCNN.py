"""
Multilayer VAE + Pixel CNN
Ishaan Gulrajani
"""

import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(high_priority=False)
except ImportError:
    pass

import lib.lsun_downsampled

import lib
import lib.debug
import lib.mnist_binarized
import lib.mnist_256ary
import lib.train_loop
import lib.ops.mlp
import lib.ops.conv_encoder
import lib.ops.conv_decoder
import lib.ops.kl_unit_gaussian
import lib.ops.kl_gaussian_gaussian
import lib.ops.conv2d
import lib.ops.diagonal_bilstm
import lib.ops.relu
import lib.ops.softmax_nll
import lib.ops.softmax_and_sample
import lib.ops.embedding

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import scipy.misc
from scipy.misc import imsave
import lasagne

import time
import functools

save_params = True
save_prefix = 'reduced_pixel_cnn_'
PIXEL_CNN_REDUCED_RF = True

theano.config.dnn.conv.algo_fwd = 'time_on_shape_change'
theano.config.dnn.conv.algo_bwd_filter = 'time_on_shape_change'
theano.config.dnn.conv.algo_bwd_data = 'time_on_shape_change'

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

# bigrun
# DIM_1        = 64
# DIM_PIX_1    = 64
# DIM_2        = 64
# DIM_3        = 128
# LATENT_DIM_1 = 128
# DIM_PIX_2    = 256
# DIM_4        = 256
# DIM_5        = 2048
# LATENT_DIM_2 = 1024

DIM_1        = 128
DIM_PIX_1    = 128

# evenbigger
# DIM_1        = 128
# DIM_PIX_1    = 128
# DIM_2        = 256
# DIM_3        = 512
# LATENT_DIM_1 = 256
# DIM_PIX_2    = 256
# DIM_4        = 1024
# DIM_5        = 2048
# LATENT_DIM_2 = 1024

# bigrun v2
# DIM_1        = 64
# DIM_PIX_1    = 128
# DIM_2        = 128
# DIM_3        = 256
# LATENT_DIM_1 = 128
# DIM_PIX_2    = 256
# DIM_4        = 512
# DIM_5        = 2048
# LATENT_DIM_2 = 1024

ALPHA_ITERS = 10000
ALPHA2_ITERS = 20000
ALPHA3_ITERS = 50000
BETA_ITERS = 1000

VANILLA = False
LR = 1e-3

LSUN_DOWNSAMPLE = True

TIMES = ('iters', 100, 1000*1000, 10000)

BATCH_SIZE = 64
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32
train_data, dev_data = lib.lsun_downsampled.load(BATCH_SIZE, LSUN_DOWNSAMPLE)

lib.print_model_settings(locals().copy())

theano_srng = RandomStreams(seed=234)


def Dec1(images):
    # images = ((T.cast(images, 'float32') / 128) - 1) * 5
    # Approximately in [-2, 2]
    images = (T.cast(images, 'float32') - 130.87536969086216) / 67.922177095784917

    masked_images = T.nnet.relu(lib.ops.conv2d.Conv2D(
        'Dec1.Pix1',
        input_dim=N_CHANNELS,
        output_dim=DIM_1,
        filter_size=7,
        inputs=images,
        mask_type=('a', N_CHANNELS)
    ))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix3', input_dim=DIM_1, output_dim=DIM_PIX_1, filter_size=3, inputs=masked_images, mask_type=('b', N_CHANNELS)))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix4', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))

    if not PIXEL_CNN_REDUCED_RF:
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix5', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix6', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix7', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix8', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix9', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix10', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix11', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix12', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))
        output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix13', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=3, inputs=output, mask_type=('b', N_CHANNELS)))

    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix14', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS)))
    output = T.nnet.relu(lib.ops.conv2d.Conv2D('Dec1.Pix15', input_dim=DIM_PIX_1, output_dim=DIM_PIX_1, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS)))

    output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=DIM_PIX_1, output_dim=256*N_CHANNELS, filter_size=1, inputs=output, mask_type=('b', N_CHANNELS), he_init=False)

    return output.reshape((-1, 256, N_CHANNELS, HEIGHT, WIDTH)).dimshuffle(0,2,3,4,1)

total_iters = T.iscalar('total_iters')
images = T.itensor4('images')  # shape: (batch size, n channels, height, width)

alpha = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(ALPHA_ITERS))


def split(mu_and_logsig):
    mu, logsig = mu_and_logsig[:, ::2], mu_and_logsig[:, 1::2]
    logsig = T.log(T.nnet.softplus(logsig))
    return mu, logsig


def clamp_logsig(logsig):
    beta = T.minimum(1, T.cast(total_iters, theano.config.floatX) / lib.floatX(BETA_ITERS))
    return T.nnet.relu(logsig, alpha=beta)

# Layer 1

outputs1 = Dec1(images)

reconst_cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(outputs1.reshape((-1, 256))),
    images.flatten()
).mean()

cost = reconst_cost

# input image
dec1_fn_targets = T.itensor4('dec1_fn_targets')
# coordinates
dec1_fn_ch = T.iscalar()
dec1_fn_y = T.iscalar()
dec1_fn_x = T.iscalar()
dec1_fn_logit = Dec1(dec1_fn_targets)[:, dec1_fn_ch, dec1_fn_y, dec1_fn_x]
dec1_fn = theano.function(
    [dec1_fn_targets, dec1_fn_ch, dec1_fn_y, dec1_fn_x],
    lib.ops.softmax_and_sample.softmax_and_sample(dec1_fn_logit),
    on_unused_input='warn'
)

dev_images = dev_data().next()[0]


def generate_and_save_samples(tag):
    def color_grid_vis(X, nh, nw, save_path):
        # from github.com/Newmu
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
        for n, x in enumerate(X):
            j = n / nw
            i = n % nw
            img[j*h:j*h+h, i*w:i*w+w, :] = x
        imsave(save_path, img)

    samples = np.zeros(
        (64, N_CHANNELS, HEIGHT, WIDTH),
        dtype='int32'
    )

    print "Generating samples"
    for y in xrange(HEIGHT):
        for x in xrange(WIDTH):
            for ch in xrange(N_CHANNELS):
                next_sample = dec1_fn(samples, ch, y, x)
                samples[:, ch, y, x] = next_sample

    samples[16:32:4] = dev_images[:4]

    print "Saving samples"
    color_grid_vis(
        samples,
        8,
        8,
        os.path.join(os.getcwd(), 'samples', save_prefix +
                     'samples_{}.png'.format(tag))
    )


def generate_and_save_samples_twice(tag):
    generate_and_save_samples(tag)
    generate_and_save_samples(tag+"_2")

# Train!

lib.train_loop.train_loop(
    inputs=[total_iters, images],
    inject_total_iters=True,
    cost=cost,
    prints=[
        ('alpha', alpha),
        ('reconst', reconst_cost),
        # ('mic_kl1', mic_kl_1),
    ],
    optimizer=functools.partial(lasagne.updates.adam, learning_rate=LR),
    train_data=train_data,
    # test_data=dev_data,
    save_params=save_params,
    save_prefix=save_prefix,
    callback=generate_and_save_samples,
    times=TIMES
)
