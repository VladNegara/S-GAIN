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

This version uses TensorFlow 1.x and FP32 precision.

We adapted the original GAIN code for our work:
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets",
           ICML, 2018.
Paper Link: https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
"""

import numpy as np
import tensorflow.compat.v1 as tf

from tqdm import tqdm

from utils.inits_TFv1_FP32 import normal_xavier_init, random_init, erdos_renyi_init, erdos_renyi_random_weights_init
from utils.metrics import get_sparsity
from utils.utils import binary_sampler, uniform_sampler, sample_batch_index, normalization, renormalization, rounding

tf.disable_v2_behavior()


def s_gain(miss_data_x, batch_size=128, hint_rate=0.9, alpha=100, iterations=10000,
           generator_sparsity=0, generator_modality='dense', discriminator_sparsity=0, discriminator_modality='dense',
           verbose=False, no_model=True, monitor=None):
    """Impute the missing values in miss_data_x.

    :param miss_data_x: the data with missing values
    :param batch_size: the number of samples in mini-batch
    :param hint_rate: the hint probability
    :param alpha: the hyperparameter
    :param iterations: the number of training iterations (epochs)
    :param generator_sparsity: the probability of sparsity in the generator
    :param generator_modality: the initialization and pruning and regrowth strategy of the generator
    :param discriminator_sparsity: the probability of sparsity in the discriminator
    :param discriminator_modality: the initialization and pruning and regrowth strategy of the discriminator
    :param verbose: enable verbose output to console
    :param no_model: don't save the trained model
    :param monitor: the monitor object used for the measurements

    :return:
    - imputed_data_x: the imputed data
    """

    if verbose: print('Starting GAIN...')

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
    if generator_modality in ('dense', 'random'):
        G_W1 = normal_xavier_init([dim * 2, h_dim])
        G_W2 = normal_xavier_init([h_dim, h_dim])
        G_W3 = normal_xavier_init([h_dim, dim])

        if generator_modality == 'random':
            G_W1, G_W2, G_W3 = random_init([G_W1, G_W2, G_W3], generator_sparsity)

    elif generator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
        G_Ws = {
            'G_W1': np.zeros([dim * 2, h_dim]),
            'G_W2': np.zeros([h_dim, h_dim]),
            'G_W3': np.zeros([h_dim, dim])
        }

        if generator_modality == 'ER':
            G_W1, G_W2, G_W3 = erdos_renyi_init(G_Ws, generator_sparsity).values()
        elif generator_modality == 'ERK':
            return None
        elif generator_modality == 'ERRW':
            G_W1, G_W2, G_W3 = erdos_renyi_random_weights_init(G_Ws, generator_sparsity).values()
        else:  # ERKRW
            return None

    elif generator_modality == 'SNIP':
        return None

    elif generator_modality == 'GraSP':
        return None

    elif generator_modality == 'RSensitivity':
        return None

    else:  # This should not happen
        print(f'Invalid generator modality: {generator_modality}. Exiting the program.')
        return None

    G_W1 = tf.Variable(G_W1)
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(G_W2)
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(G_W3)
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Discriminator variables: Data + Hint as inputs
    if discriminator_modality in ('dense', 'random'):
        D_W1 = normal_xavier_init([dim * 2, h_dim])
        D_W2 = normal_xavier_init([h_dim, h_dim])
        D_W3 = normal_xavier_init([h_dim, dim])

        if discriminator_modality == 'random':
            D_W1, D_W2, D_W3 = random_init([D_W1, D_W2, D_W3], discriminator_sparsity)

    elif discriminator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
        D_Ws = {
            'D_W1': np.zeros([dim * 2, h_dim]),
            'D_W2': np.zeros([h_dim, h_dim]),
            'D_W3': np.zeros([h_dim, dim])
        }

        if discriminator_modality == 'ER':
            D_W1, D_W2, D_W3 = erdos_renyi_init(D_Ws, discriminator_sparsity).values()
        elif discriminator_modality == 'ERK':
            return None
        elif discriminator_modality == 'ERRW':
            D_W1, D_W2, D_W3 = erdos_renyi_random_weights_init(D_Ws, discriminator_sparsity).values()
        else:  # ERKRW
            return None

    elif discriminator_modality == 'SNIP':
        return None

    elif discriminator_modality == 'GraSP':
        return None

    elif discriminator_modality == 'RSensitivity':
        return None

    else:  # This should not happen
        print(f'Invalid discriminator modality: {discriminator_modality}. Exiting the program.')
        return None

    D_W1 = tf.Variable(D_W1)
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(D_W2)
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(D_W3)
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

    if monitor: monitor.log_loss(np.nan, np.nan, np.nan)

    # -- S-GAIN solver ------------------------------------------------------------------------------------------------

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # -- S-GAIN training ----------------------------------------------------------------------------------------------

    if verbose: print('Training S-GAIN...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for it in tqdm(range(iterations)):
        if monitor:
            monitor.log_imputation_time()

            G_sparsities = get_sparsity(sess.run(theta_G))
            D_sparsities = get_sparsity(sess.run(theta_D))
            monitor.log_sparsity(G_sparsities, D_sparsities)

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

        if monitor: monitor.log_loss(G_loss_curr, D_loss_curr, MSE_loss_curr)

        # Terminate failed imputation immediately
        if np.isnan(G_loss_curr):
            if verbose: print('Imputation failed! Terminating...')
            break

    if monitor:
        monitor.log_imputation_time()

        G_sparsities = get_sparsity(sess.run(theta_G))
        D_sparsities = get_sparsity(sess.run(theta_D))
        monitor.log_sparsity(G_sparsities, D_sparsities)

    if verbose: print('Finished training.')

    def get_image_imputation():
        """ Check the imputation of a square image from the model and saves it
        """
        import random

        random_img = random.randint(0, len(miss_data_x)-1)
        img_np = miss_data_x[random_img]

        # Regenerate mask
        mask_np = 1 - np.isnan(img_np)
        mask_np = mask_np.astype(np.float32)

        # Simple normalization (0–1) ignoring NaNs
        img_filled = np.nan_to_num(img_np, nan=0.0)
        img_norm = img_filled / 255.0

        # Add noise where pixels are missing
        X_input = mask_np * img_norm + (1 - mask_np) * np.random.uniform(0, 0.01, size=img_norm.shape)
        X_input = X_input.reshape(1, -1).astype(np.float32)

        # Run inference
        imputed = sess.run(G_sample, feed_dict={X: X_input, M: mask_np.reshape(1, -1)})[0]
        # Convert imputed result back to 0–255 range
        imputed = np.clip(imputed * 255.0, 0, 255)
        # print(miss_data_x[0])
        # print(imputed)

        from PIL import Image

        orig = np.nan_to_num(img_np, nan=0.0).reshape(28, 28)
        imputed_img = imputed.reshape(28, 28)

        # Stack side-by-side for comparison
        comparison = np.hstack([orig, imputed_img])
        comparison_img = Image.fromarray(comparison.astype(np.uint8))
        # comparison_img.show()
        comparison_img.save("imputed_comparison.png")


    # Using fashion_mnist dataset
    # if len(miss_data_x[0]) == 784:
    #     get_image_imputation()

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

    # Reshaping
    if reshaped:
        if verbose: print('Reshaping the imputed data to the original shape...')
        imputed_data_x = np.reshape(imputed_data_x, shape)

    # Stop monitor
    if monitor:
        monitor.log_imputation_time()
        monitor.log_rmse(imputed_data_x)
        monitor.stop_all_monitors()

        # Save model
        if not no_model: monitor.set_model(sess.run(theta_G), sess.run(theta_D))

    if verbose: print('Stopped S-GAIN.')

    return imputed_data_x
