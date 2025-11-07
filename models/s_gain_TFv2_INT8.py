# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S-GAIN.

Todo: This version uses TensorFlow 2.x and INT8 precision.

Todo: description of the model
Todo: run S-GAIN on GPU
Todo: rebuild S-GAIN in tensorflow v2 with sparse tensors and INT8 precision

We adapted the original GAIN code for our work:
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets",
           ICML, 2018.
Paper Link: https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
"""

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from keras import Sequential, Input
from keras.src.layers import Dense

from utils.inits_TFv2_INT8 import normal_xavier_init
from utils.utils import binary_sampler, uniform_sampler, sample_batch_index, normalization, renormalization, rounding


# -- Models -----------------------------------------------------------------------------------------------------------

def generator(miss_data_x, mask, theta):
    """

    :param miss_data_x:
    :param mask:
    :param theta:

    :return:
    """
    W1, W2, W3, b1, b2, b3 = theta

    model = Sequential()

    # Input layer
    inputs = Input(name='G_inputs')  # Todo
    model.add(inputs)

    # Layer 1
    L1 = Dense(W1.shape[1], activation='relu', name='G_L1')
    L1.build((W1.shape[1], W1.shape[0]))
    L1.set_weights([W1, b1])
    L1.kernel.name = 'G_W1'
    L1.bias.name = 'G_b1'
    model.add(L1)

    # Layer 2
    L2 = Dense(W2.shape[1], activation='relu', name='G_L2')
    L2.build((W2.shape[1], W2.shape[0]))
    L2.set_weights([W2, b2])
    L2.kernel.name = 'G_W2'
    L2.bias.name = 'G_b2'
    model.add(L2)

    # Layer 3 (MinMax normalized output)
    L3 = Dense(W3.shape[1], activation='sigmoid', name='G_L3')
    L3.build((W3.shape[1], W3.shape[0]))
    L3.set_weights([W3, b3])
    L3.kernel.name = 'G_W3'
    L3.bias.name = 'G_b3'
    model.add(L3)


def discriminator(miss_data_x, hint, theta):
    """

    :param miss_data_x:
    :param hint:
    :param theta:

    :return:
    """
    W1, W2, W3, b1, b2, b3 = theta

    model = Sequential()

    # Input layer
    inputs = Input()  # Todo
    model.add(inputs)

    # Layer 1
    L1 = Dense(W1.shape[1], activation='relu', name='D_L1')
    L1.build((W1.shape[1], W1.shape[0]))
    L1.set_weights([W1, b1])
    L1.kernel.name = 'D_W1'
    L1.bias.name = 'D_b1'
    model.add(L1)

    # Layer 2
    L2 = Dense(W2.shape[1], activation='relu', name='D_L2')
    L2.build((W2.shape[1], W2.shape[0]))
    L2.set_weights([W2, b2])
    L2.kernel.name = 'D_W2'
    L2.bias.name = 'D_b2'
    model.add(L2)

    # Layer 3 (MinMax normalized output)
    L3 = Dense(W3.shape[1], activation='sigmoid', name='D_L3')
    L3.build((W3.shape[1], W3.shape[0]))
    L3.set_weights([W3, b3])
    L3.kernel.name = 'D_W3'
    L3.bias.name = 'D_b3'
    model.add(L3)


# -- S-GAIN -----------------------------------------------------------------------------------------------------------

def s_gain(miss_data_x, batch_size=128, hint_rate=0.9, alpha=100, iterations=10000,
           generator_initialization='dense', generator_sparsity=0, discriminator_initialization='dense',
           discriminator_sparsity=0, verbose=False, no_model=True, monitor=None):
    """Impute the missing values in miss_data_x.

    :param miss_data_x: the data with missing values
    :param batch_size: the number of samples in the mini-batch
    :param hint_rate: the hint probability
    :param alpha: the hyperparameter
    :param iterations: the number of training iterations (epochs)
    :param generator_initialization: the initialization and pruning and regrowth strategy of the generator
    :param generator_sparsity: the probability of sparsity in the generator
    :param discriminator_initialization: the initialization and pruning and regrowth strategy of the discriminator
    :param discriminator_sparsity: the probability of sparsity in the discriminator
    :param verbose: enable verbose output to console
    :param no_model: don't save the trained model
    :param monitor: the monitor object used for the measurements

    :return:
    - imputed_data_x: the imputed data
    """

    if verbose: print('Starting S-GAIN...')

    # Start monitors
    if monitor is not None:
        monitor.init_monitor()
        monitor.start_all_monitors()

    # Reshape the missing data
    shape = miss_data_x.shape
    reshaped = False
    if len(shape) > 2:
        if verbose: print('Reshaping missing data matrix to 2D...')
        new_shape = shape[0], np.prod(shape[1:])
        miss_data_x = np.reshape(miss_data_x, new_shape)
        reshaped = True

    # Define the mask matrix
    data_mask = 1 - np.isnan(miss_data_x)

    # Parameters
    no, dim = miss_data_x.shape
    h_dim = int(dim)

    # Normalization
    if verbose: print('Normalizing data...')
    norm_data_x, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data_x, False)

    # -- S-GAIN architecture ------------------------------------------------------------------------------------------

    # Input placeholders
    X = tf.placeholder(tf.float32, shape=[None, dim])  # Data vector
    M = tf.placeholder(tf.float32, shape=[None, dim])  # Mask vector
    H = tf.placeholder(tf.float32, shape=[None, dim])  # Hint vector

    # Generator variables: Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(normal_xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(normal_xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(normal_xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Discriminator variables: Data + Hint as inputs
    D_W1 = tf.Variable(normal_xavier_init([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(normal_xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(normal_xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # -- S-GAIN functions ---------------------------------------------------------------------------------------------

    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.add(tf.matmul(inputs, G_W1), G_b1))
        G_h2 = tf.nn.relu(tf.add(tf.matmul(G_h1, G_W2), G_b2))
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.add(tf.matmul(G_h2, G_W3), G_b3))
        return G_prob

    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        # MinMax normalized output
        D_prob = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)
        return D_prob

    # -- S-GAIN structure ---------------------------------------------------------------------------------------------

    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    # -- S-GAIN solver ------------------------------------------------------------------------------------------------

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # -- S-GAIN training ----------------------------------------------------------------------------------------------

    if verbose: print('Training S-GAIN...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for it in tqdm(range(iterations)):
        if monitor is not None: monitor.log_imputation_time()

        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_mask[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb})

    if monitor is not None: monitor.log_imputation_time()
    if verbose: print('Finished training.')

    # Return the imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_mask
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data_x = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    imputed_data_x = data_mask * norm_data_x + (1 - data_mask) * imputed_data_x

    # -----------------------------------------------------------------------------------------------------------------

    # Renormalization
    if verbose: print('Re-normalizing data...')
    imputed_data_x = renormalization(imputed_data_x, norm_parameters)

    # Rounding
    if verbose: print('Rounding data...')
    imputed_data_x = rounding(imputed_data_x, miss_data_x)

    # Only impute missing data
    data_mask = np.zeros(miss_data_x.shape, dtype=int)
    data_mask[np.isnan(miss_data_x)] = 1
    imputed_data_x = np.where(data_mask, imputed_data_x, miss_data_x)

    # Reshaping
    if reshaped:
        if verbose: print('Reshaping the imputed data to the original shape...')
        imputed_data_x = np.reshape(imputed_data_x, shape)

    # Stop monitor
    if monitor is not None:
        monitor.log_imputation_time()
        monitor.log_rmse(imputed_data_x)
        monitor.stop_all_monitors()

        # Save model
        if not no_model: monitor.set_model(sess.run(theta_G), sess.run(theta_D))

    if verbose: print('Stopped S-GAIN.')

    return imputed_data_x
