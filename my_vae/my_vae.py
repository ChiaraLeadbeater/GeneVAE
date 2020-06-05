# Candidate 8228T

# Requires Tensorflow 1.4 and above for the BatchNormalisation and LeakyReLu

# Debugging is not convenient in tensorflow because of the static nature of the computational graph
# (viz. the dynamic nature of the computational graph in torch)
# The print_tensor statements make the code less readable from the mathematical/algorithmic point of view
# But, they are are almost indispensable when the almost-unavoidable problems occur following code alterations
# I, therefore, decided to leave them in once the model was debugged

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input, Lambda
from keras.layers import BatchNormalization, Dropout, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.regularizers import l1_l2
from keras.callbacks import LambdaCallback, Callback, TensorBoard, ReduceLROnPlateau
from keras.losses import mse, binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import tensorflow as tf

import anndata
from anndata import read_h5ad
import copy


# Default parameters for the model

def scrnaseq_vae_default_params ():
    params = {
        'architecture' : 'vae',
        'model' : 'gaussian',
        'variational' : True,
        'log_preprocess' : True,
        'act' : 'relu',
        'init' : 'glorot_uniform',
        'input_dropout' : 0.2,
        'dropout' : 0.2,
        'decode_dropout' : True,
        'last_decode_dropout' : False,
        'batchnorm_momentum' : 0.01,
        'batchnorm_epsilon' : 0.001,
        'l1' : 0.0,
        'l2' : 0.0,
        'l1_enc' : 0.0,
        'l2_enc' : 0.0,
        'ridge' : 0.0,
        'reconstruction_loss' : 'mse',
        'optimizer' : Adam(0.001),
        'optimizer_discriminator' : Adam(0.001),
        'optimizer_generator' : Adam(0.001),
        'n_epochs' : 400,
        'n_epochs_vae' : 5,
        'n_epochs_discriminator' : 5,
        'n_epochs_generator' : 5,
        'batch_size' : 128,
        'latent_dim' : 10,
        'hidden_structure' : [128,64],
        'layer_structure' : 'augmented',
        'beta' : 1.0,
        'return_latent_activations' : False,
        'discriminator_pdf' : 'gaussian',
        'discriminator_prior_mean': 0.0,
        'discriminator_prior_sigma': 1.0,
        'discriminator_prior_kappa': 1.0,
        'discriminator_wasserstein': False,
        'discriminator_hidden_structure': [512,216],
        'wasserstein_lambda': 1.0,
        'size_factors' : 'static',
        'validation_split' : 0.1,
        'training_method' : 'fit',
        'logdir': './tensorboard_logs',
        'patience_reduce_lr': 0,
        'factor_reduce_lr': 0.0
    }
    return params




def scrnaseq_vae (adata, raw_adata = None, size_factors = None, params = None, verbose=False,
          debug=False, tf_debug=False, tensorboard=False):

    if params is None:
        params = scrnaseq_vae_default_params ()
    else:
        for key, value in params.items ():
            assert key in scrnaseq_vae_default_params (), key + ' not a valid key for params'
        for key, value in scrnaseq_vae_default_params().items ():
            assert key in params, 'params does not have ' + key + ' key'

    assert params ['architecture'] in ['vae', 'aae']
    assert params ['model'] in ['gaussian', 'zinb']
    assert params ['discriminator_pdf'] in ['gaussian', 'vonmises']
    assert params ['training_method'] in ['fit', 'fit_batch', 'fit_generator']
    if params ['model'] == 'gaussian':
        assert params ['reconstruction_loss'] in ['mse', 'binary_crossentropy']
    if params ['size_factors'] is not None:
        assert params ['size_factors'] in ['static', 'dynamic', 'user']

    if debug:
        print ('scrnaseq_vae params:')
        print (params)
        if isinstance (params['optimizer'], Adam):
            print ('optimizer:', params['optimizer'].get_config())
        if isinstance (params['optimizer_discriminator'], Adam):
            print ('optimizer_discriminator:', params['optimizer_discriminator'].get_config())
        if isinstance (params['optimizer_generator'], Adam):
            print ('optimizer_generator:', params['optimizer_generator'].get_config())

    print ('Tensorflow version:', tf.__version__)


    # Implementation of different schemes for applying size factors
    # The static and dynamic schemes are inspired from dca and scvi respectively

    class SizeFactors():
        def __init__(self,
                adata,
                sf_user,
                sf_mode,
                debug):
            self.adata = adata
            self.sf_user = sf_user
            self.sf_mode = sf_mode
            self.debug = debug
            self.sf_means = None
            self.sf_vars = None
            self.sf_log_vars = None
            self.sf_values = None
            self.model_inputs = None

            self.calcInitialValues ()
            self.createInputs ()

        def calcInitialValues (self):
            if self.sf_mode is not None:
                assert self.sf_mode in ['static', 'dynamic', 'user']
                if self.sf_user is not None and self.sf_mode == 'user':
                    assert self.sf_user.shape[0] == self.adata.X.shape[0] and self.sf_user.shape[1] == 1
                    sf_values = self.sf_user
                else:
                    if self.sf_mode == 'static':
                        self.sf_values = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
                        if self.debug:
                            print ('adata.obs.n_counts:')
                            print (self.adata.obs.n_counts)
                    elif self.sf_mode == 'dynamic':
                        # By using scanpy-calculated counts, further pre-processing can be performed and
                        # will not spoil the calculation of the reference mean and standard deviation
                        log_counts = np.log(self.adata.obs.n_counts)
                        self.sf_means = (np.mean(log_counts) * np.ones((self.adata.X.shape[0], 1))).astype(np.float32)
                        self.sf_vars = (np.var(log_counts) * np.ones((self.adata.X.shape[0], 1))).astype(np.float32)
                        self.sf_log_vars = np.log (self.sf_vars)
                        if self.debug:
                            print ('DYNAMIC SIZE FACTOR CALCS')
                            print (log_counts)
                            print (self.sf_means)
                            print (self.sf_vars)
                            print (self.sf_log_vars)
                            print ('SHAPE')
                            print (self.sf_means.shape)
                            print (self.adata.obs.n_counts)
                            print (np.log (self.adata.obs.n_counts))
                            print (np.mean (np.log (self.adata.obs.n_counts)))
            if self.debug:
                if self.sf_values is not None:
                    print ('sf_values:')
                    print (self.sf_values)
                    print (self.sf_values.shape)
                if self.sf_means is not None:
                    print ('sf_means:')
                    print (self.sf_means)
                    print (self.sf_means.shape)

        def getSizeFactors (self, as_arrays=False):
            sf = {}
            if self.sf_mode == 'static':
                if as_arrays:
                    return [self.sf_values]
                sf ['size_factors'] = self.sf_values
            elif self.sf_mode == 'user':
                if as_arrays:
                    return [self.sf_user]
                sf ['size_factors'] = self.sf_user
            elif self.sf_mode == 'dynamic':
                if as_arrays:
                    return [self.sf_means, self.sf_log_vars]
                sf ['size_factors_means'] = self.sf_means
                sf ['size_factors_logvars'] = self.sf_log_vars    # make them log vars
            return sf

        def usingSizeFactors (self):
            return True if self.sf_mode is not None else False

        def usingDynamicSizeFactors (self):
            return True if self.sf_mode == 'dynamic' else False

        def createInputs (self):
            if self.sf_mode == 'static' or self.sf_mode == 'user':
                inp_sf = Input (shape=(1,), name='size_factors')
                self.model_inputs = [inp_sf]
            elif self.sf_mode == 'dynamic':
                inp_means = Input (shape=(1,), name='size_factors_means')
                inp_logvars = Input (shape=(1,), name='size_factors_logvars')
                self.model_inputs = [inp_means, inp_logvars]

        def getInputs (self, getDynamic=True):
            getI = True
            if self.usingDynamicSizeFactors () and not getDynamic:
                getI = False
            return self.model_inputs if self.model_inputs is not None and getI else []


    def show_model (vae, encoder, decoder, discriminator = None, generator = None):
        print ('vae')
        vae.summary ()
        print ('encoder')
        encoder.summary ()
        print ('decoder')
        decoder.summary ()
        if discriminator is not None:
            print ('discriminator')
            discriminator.summary ()
        if generator is not None:
            print ('generator')
            generator.summary ()
        plot_model (vae, to_file='vae.png', show_shapes=True)
        plot_model (encoder, to_file='vae_encoder.png', show_shapes=True)
        plot_model (decoder, to_file='vae_decoder.png', show_shapes=True)
        if discriminator is not None:
            plot_model (discriminator, to_file='vae_discriminator.png', show_shapes=True)
        if generator is not None:
            plot_model (generator, to_file='vae_generator.png', show_shapes=True)

    def build (count_dim, sf, params, mod_args, debug):

        # Model loss calculations

        def gaussian_reconstruction_loss (inputs, outputs, output_dim, params):
            if params ['reconstruction_loss'] == 'mse':
                reconstruction_loss = mse (inputs, outputs)
            elif params ['reconstruction_loss'] == 'binary_crossentropy':
                reconstruction_loss = binary_crossentropy (inputs, outputs)

            if debug:
                reconstruction_loss = K.print_tensor (reconstruction_loss, "\ngaussian reconstruction_loss")

            reconstruction_loss *= output_dim

            if debug:
                reconstruction_loss = K.print_tensor (reconstruction_loss, "\ngaussian reconstruction_loss scaled up")

            return reconstruction_loss

        def zinb_reconstruction_loss (y, mu, theta, pi, output_dim, params, debug):

            eps = 1e-10

            if debug:
                y = K.print_tensor (y, "\ny:")
                mu = K.print_tensor (mu, "\nmu:")
                theta = K.print_tensor (theta, "\ntheta:")
                pi = K.print_tensor (pi, "\npi:")

            t1 = tf.lgamma(theta+eps) + tf.lgamma(y+1.0) - tf.lgamma(y+theta+eps)
            t2 = (theta+y) * tf.log(1.0 + (mu/(theta+eps))) + (y * (tf.log(theta+eps) - tf.log(mu+eps)))

            if debug:
                t1 = K.print_tensor (t1, "\nt1:")
                t2 = K.print_tensor (t2, "\nt2:")

            nb_case = t1 + t2 - tf.log(1.0-pi+eps)

            if debug:
                nb_case = K.print_tensor (nb_case, "\nnb_case:")

            zero_nb = tf.pow(theta/(theta+mu+eps), theta)

            if debug:
                zero_nb = K.print_tensor (zero_nb, "\nzero_nb:")

            zero_case = -tf.log(pi + ((1.0-pi)*zero_nb) + eps)

            if debug:
                zero_case = K.print_tensor (zero_case, "\nzero_case:")

            result = tf.where(tf.less(y, 1e-8), zero_case, nb_case)
 
            ridge = params ['ridge']*tf.square(pi)
            end_result = result + ridge

            if debug:
                result = K.print_tensor (result, "\nresult:")
                ridge = K.print_tensor (ridge, "\nridge:")
                end_result = K.print_tensor (end_result, "\nend_result:")

            # https://github.com/keras-team/keras/blob/master/keras/losses.py
            #reconstruction_loss = K.mean(end_result, axis=-1)
            reconstruction_loss = K.sum(end_result, axis=-1)

            if debug:
                reconstruction_loss = K.print_tensor (reconstruction_loss, "\nzinb reconstruction_loss:")

            return reconstruction_loss

        def gaussian_kl_loss (z_mean, z_log_var, params, debug):
            kl_loss = 1 + z_log_var - K.square (z_mean) - K.exp (z_log_var)
            kl_loss = K.sum (kl_loss, axis=-1)
            kl_loss *= -0.5

            if debug:
                z_mean = K.print_tensor (z_mean, "\nz_mean")
                z_log_var = K.print_tensor (z_log_var, "\nz_log_var")
                kl_loss = K.print_tensor (kl_loss, "\nkl_loss")

            return kl_loss

        # https://stackoverflow.com/questions/41863814/kl-divergence-in-tensorflow
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/kullback_leibler.py
        def gaussian_kl_divergence (m, v, params, debug):
            ds = tf.contrib.distributions
            p = ds.Normal(loc=m[0], scale=v[0])
            q = ds.Normal(loc=m[1], scale=v[1])
            kl = ds.kl_divergence(p, q)
            if debug:
                kl = K.print_tensor (kl, "\nkl_divergence")
            return kl

        def gaussian_loss (inputs, outputs, z_mean, z_log_var, sf,
                          z_mean_sf, z_log_var_sf, sf_means, sf_log_vars,
                          mod_args,
                          output_dim, params, debug):
            if debug:
                kl_beta_weight = K.print_tensor (mod_args['kl_beta_weight'], "\nkl_beta_weight")
            else:
                kl_beta_weight = mod_args['kl_beta_weight']

            reconstruction_loss = gaussian_reconstruction_loss (inputs, outputs,
                           output_dim, params)

            if params ['variational']:
                kl_loss = gaussian_kl_loss (z_mean, z_log_var, params, debug)
                total_loss = reconstruction_loss + kl_beta_weight * kl_loss
            else:
                total_loss = reconstruction_loss
            if debug:
                total_loss = K.print_tensor (total_loss, "\ngaussian total vae_loss")

            vae_loss = K.mean (total_loss)
            recon_loss = K.mean (reconstruction_loss)

            if debug:
                vae_loss = K.print_tensor (vae_loss, "\ngaussian vae_loss")
                recon_loss = K.print_tensor (recon_loss, "\ngaussian recon_loss")

            kl_sf = None
            if sf.usingDynamicSizeFactors ():
                m = [sf_means, z_mean_sf]
                v = [K.sqrt(K.exp(sf_log_vars)), K.sqrt(K.exp(z_log_var_sf))]
                kl_sf = gaussian_kl_divergence (m, v, params, debug)

                if debug:
                    kl_sf = K.print_tensor (kl_sf, "\ngaussian kl_sf")
                kl_sf = K.mean (kl_sf)
                if debug:
                    kl_sf = K.print_tensor (kl_sf, "\ngaussian mean kl_sf")

            return vae_loss, recon_loss, kl_sf

        def zinb_loss (inputs, mu, dispersion, pi, z_mean, z_log_var, sf,
                             z_mean_sf, z_log_var_sf, sf_means, sf_log_vars,
                             mod_args,
                             output_dim, params, debug):
            if debug:
                kl_beta_weight = K.print_tensor (mod_args['kl_beta_weight'], "\nkl_beta_weight")
            else:
                kl_beta_weight = mod_args['kl_beta_weight']

            reconstruction_loss = zinb_reconstruction_loss (inputs, mu, dispersion, pi,
                           output_dim, params, debug)

            if params ['variational']:
                kl_loss = gaussian_kl_loss (z_mean, z_log_var, params, debug)
                total_loss = reconstruction_loss + kl_beta_weight * kl_loss
            else:
                total_loss = reconstruction_loss
            if debug:
                total_loss = K.print_tensor (total_loss, "\nzinb total loss")

            vae_loss = K.mean (total_loss)
            recon_loss = K.mean (reconstruction_loss)

            if debug:
                vae_loss = K.print_tensor (vae_loss, "\nzinb vae_loss")
                recon_loss = K.print_tensor (recon_loss, "\nzinb recon_loss")

            kl_sf = None
            if sf.usingDynamicSizeFactors ():
                m = [sf_means, z_mean_sf]
                v = [K.sqrt(K.exp(sf_log_vars)), K.sqrt(K.exp(z_log_var_sf))]
                kl_sf = gaussian_kl_divergence (m, v, params, debug)

                if debug:
                    kl_sf = K.print_tensor (kl_sf, "\nzinb kl_sf")
                kl_sf = K.mean (kl_sf)
                if debug:
                    kl_sf = K.print_tensor (kl_sf, "\nzinb mean kl_sf")

            return vae_loss, recon_loss, kl_sf


        # Model construction

        def layers (x, params, dims, nm, encoding=True):
            
            assert params ['layer_structure'] in ['simple', 'augmented']

            if encoding and params ['input_dropout'] > 1e-10:
                x = Dropout (params ['input_dropout'],
                          name=nm+'_input_dropout')(x)
            for i, d in enumerate(dims):
                if params ['layer_structure'] == 'simple':
                    x = Dense (d, activation = params ['act'],
                        kernel_initializer = params ['init'],
                        name = nm + '_' + str (i))(x)
                elif params ['layer_structure'] == 'augmented':
                    # kernel_regularizer ??
                    if encoding:
                        r = l1_l2 (params ['l1_enc'], params ['l2_enc'])
                    else:
                        r = l1_l2 (params ['l1'], params ['l2'])
                    x = Dense (d, activation = None,
                        kernel_initializer = params ['init'],
                        kernel_regularizer = r,
                        name = nm + '_' + str (i))(x)
                    x = BatchNormalization (center=True, scale=False,
                              momentum = params ['batchnorm_momentum'],
                              epsilon = params ['batchnorm_epsilon'],
                              name=nm + '_norm_' + str (i))(x)
                    x = Activation (params ['act'],
                              name=nm + '_act_' + str (i))(x)
                    avoid_last_drop=False
                    if i == len(dims)-1 and not params ['last_decode_dropout']:
                        avoid_last_drop=True
                    #if encoding or (params ['decode_dropout'] and not avoid_last_drop):
                    if not avoid_last_drop:
                        x = Dropout (params ['dropout'],
                              name=nm + '_drop_' + str (i))(x)
            return x

        # This implements the reparameterization trick required in VAEs
        def sampling (args, operation):
            z_mean, z_log_var = args
            batch = K.shape (z_mean)[0]
            dim = K.int_shape (z_mean)[1]
            epsilon = K.random_normal (shape=(batch,dim), mean=0.0, stddev=1.0)
            if debug:
                z_mean = K.print_tensor (z_mean, "sampling" + operation + " z_mean")
                z_log_var = K.print_tensor (z_log_var, "sampling" + operation + " z_log_var")
                epsilon = K.print_tensor (epsilon, "sampling" + operation + " epsilon")
            latent_space = z_mean + K.exp(0.5 * z_log_var) * epsilon
            if debug:
                latent_space = K.print_tensor (latent_space, "sampling" + operation + " latent_space")
            return latent_space

        def build_encoder (original_dim, sf, params):
            z_mean, z_log_var, z_mean_sf, z_log_var_sf, z_sf = [None, None, None, None, None]
            inputs = Input (shape=(original_dim,), name='count_input')
            x = layers (inputs, params, params ['hidden_structure'], 'encoder',
                    encoding=True)
            if sf.usingDynamicSizeFactors ():
                x_sf = layers (inputs, params, params ['hidden_structure'], 'encoder_sf',
                    encoding=False)
                z_mean_sf = Dense (1, name = 'z_mean_sf')(x_sf)
                z_log_var_sf = Dense (1, name = 'z_log_var_sf')(x_sf)
                z_sf = Lambda (sampling, output_shape=(1,),
                              arguments = {'operation': '_sf'}, name = 'z_sf')([z_mean_sf, z_log_var_sf])
            if params ['variational']:
                z_mean = Dense (params ['latent_dim'], name = 'z_mean')(x)
                z_log_var = Dense (params ['latent_dim'], name = 'z_log_var')(x)
                z = Lambda (sampling, output_shape=(params['latent_dim'],),
                          arguments = {'operation':''}, name = 'z')([z_mean, z_log_var])
                if sf.usingDynamicSizeFactors ():
                    encoder = Model (inputs, [z, z_mean, z_log_var, z_sf, z_mean_sf, z_log_var_sf],
                              name = 'encoder')
                else:
                    encoder = Model (inputs, [z, z_mean, z_log_var], name = 'encoder')
            else:
                z = Dense (params ['latent_dim'], name = 'z')(x)
                if sf.usingDynamicSizeFactors ():
                    encoder = Model (inputs, [z, z_sf, z_mean_sf, z_log_var_sf], name = 'encoder')
                else:
                    # 2 outputs to make the rest of the code cleaner for indexing
                    encoder = Model (inputs, [z, z], name = 'encoder')

            return encoder, inputs, z_mean, z_log_var, z_mean_sf, z_log_var_sf, z_sf


        def build_decoder (original_dim, sf, z_sf, count_input, params, debug):
            latent_inputs = Input (shape =(params['latent_dim'],),
                      name = 'z_decoder_input')
            x = layers (latent_inputs, params, params ['hidden_structure'][::-1], 'decoder', encoding=False)
            sf_l = []
            if params ['model'] == 'gaussian':
                # multiply each data point by size factors ?
                outputs = Dense (original_dim, activation = 'sigmoid',
                            kernel_initializer = params ['init'], name='decoder_output')(x)
                sf_l = sf.getInputs ()
            elif params ['model'] == 'zinb':
                MeanAct = lambda a: tf.clip_by_value (K.exp(a), 1e-5, 1e6)
                DispAct = lambda a: tf.clip_by_value (tf.nn.softplus(a), 1e-4, 1e4)
                ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)), name = 'colwisemult')
                ExpZsf = Lambda(lambda l: K.exp(l), name = 'expzsf')
                PrintMuLayer = Lambda(lambda l: K.print_tensor (l, 'mu'))
                PrintMuOutLayer = Lambda(lambda l: K.print_tensor (l, 'mu_out'))
                PrintSfLayer = Lambda(lambda l: K.print_tensor (l, 'sf'))
                PrintZsfLayer = Lambda(lambda l: K.print_tensor (l, 'zsf'))
                #kernel_regularizer?
                mu = Dense (original_dim, activation = MeanAct,
                        kernel_initializer = params ['init'], name='mu')(x)
                dispersion = Dense (original_dim, activation = DispAct,
                        kernel_initializer = params ['init'], name='dispersion')(x)
                pi = Dense (original_dim, activation = 'sigmoid',
                        kernel_initializer = params ['init'], name='pi')(x)
                if debug:
                    mu = PrintMuLayer (mu)
                if sf.usingSizeFactors ():
                    if sf.usingDynamicSizeFactors ():
                        sf_l = sf.getInputs ()
                        if debug:
                            z_sf = PrintZsfLayer (z_sf)
                        z_sf = ExpZsf (z_sf)
                        mu_out = ColwiseMultLayer ([mu, z_sf])
                    else:
                        sf_l = sf.getInputs ()

                        sf_lmult = [None] * len (sf_l)
                        if debug:
                            sf_lmult [0] = PrintSfLayer (sf_l [0])
                        else:
                            sf_lmult [0] = sf_l [0]
                        mu_out = ColwiseMultLayer ([mu, sf_lmult[0]])
                else:
                    mu_out = mu
                if debug:
                    mu_out = PrintMuOutLayer (mu_out)
                outputs = [mu_out, dispersion, pi]
            decoder = Model ([latent_inputs] + sf_l + [count_input], outputs, name = 'decoder')

            if sf.usingDynamicSizeFactors ():
                return decoder, sf_l [0], sf_l [1]
            else:
                return decoder, None, None

        def build_discriminator (params):
            model = Sequential()

            for i, siz in enumerate (params['discriminator_hidden_structure']):
                model.add(Dense(siz, input_dim=params['latent_dim'] if i == 0 else None))
                model.add(LeakyReLU(alpha=0.2))
                if i != len (params['discriminator_hidden_structure'])-1:
                    model.add(Dropout(0.2))
            model.add(Dense(1, activation="sigmoid" if not params['discriminator_wasserstein'] else None))

            return model

        def settrainable(model, onoff):
            for layer in model.layers:
                layer.trainable = onoff
            model.trainable = onoff


        if debug:
            print ('count_dim:', count_dim)

        if debug and tf_debug:
            from tensorflow.python import debug as tf_dbg
            sess = K.get_session()
            sess = tf_dbg.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_dbg.has_inf_or_nan)
            K.set_session(sess)

        sf_inputs = sf.getInputs ()
        raw_count_input = Input (shape=(count_dim,), name='raw_count_input')
        encoder, count_input, z_mean, z_log_var, z_mean_sf, z_log_var_sf, z_sf = build_encoder (count_dim,
                 sf, params)
        decoder, sf_means, sf_log_vars = build_decoder (count_dim, sf, z_sf, count_input, params, debug)
        encoded_repr = encoder(count_input)[0]
        outputs = decoder ([encoded_repr] + sf_inputs + [count_input])
        vae = Model ([count_input, raw_count_input]+sf_inputs, outputs, name = 'vae')
        if params ['model'] == 'gaussian':
            vae_loss, reconstruction_loss, kl_sf = gaussian_loss (raw_count_input,
                         outputs,
                         z_mean, z_log_var,
                         sf,
                         z_mean_sf, z_log_var_sf,
                         sf_means, sf_log_vars,
                         mod_args,
                         count_dim, params, debug)
        elif params ['model'] == 'zinb':
            vae_loss, reconstruction_loss, kl_sf = zinb_loss (raw_count_input,
                         #mu, dispersion, pi,
                         outputs [0], outputs [1], outputs [2],
                         z_mean, z_log_var,
                         sf,
                         z_mean_sf, z_log_var_sf,
                         sf_means, sf_log_vars,
                         mod_args,
                         count_dim, params, debug)
        l = vae_loss if params ['architecture'] == 'vae' else reconstruction_loss
        if sf.usingDynamicSizeFactors () and kl_sf is not None:
            l += kl_sf
        vae.add_loss (l)
        vae.compile (optimizer = params ['optimizer'], loss=None)

        discriminator = None
        generator = None
        if params ['architecture'] == 'aae':
            discriminator = build_discriminator (params)
            def wasserstein_loss (apply_lambda=False, debug=False):
                def loss (y_true, y_pred):
                    if debug:
                        print ('WASSERSTEIN LOSS:', apply_lambda)
                    loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
                    if apply_lambda:
                        loss_tensor *= params ['wasserstein_lambda']
                    return loss_tensor
                return loss
            discriminator.compile(loss='binary_crossentropy' \
                      if not params['discriminator_wasserstein'] else wasserstein_loss(apply_lambda=False, debug=debug),
                optimizer=params['optimizer_discriminator'],
                metrics=['accuracy'])
            if debug:
                print ('Trainable Discriminator model')
                discriminator.summary ()
            settrainable (discriminator, False)
            #this doesn't work?
            if sf.usingDynamicSizeFactors ():
                def list_layers (model):
                    print ('list layers')
                    for layer in model.layers:
                        print (layer.name)
                list_layers (encoder)
                z_sf.trainable = False
                z_mean_sf.trainable = False
                z_log_var_sf.trainable = False
                for layer in encoder.layers:
                    rex="encoder_sf"
                    if layer.name [0 : len(rex)] == rex:
                        print ('untrainable:', layer.name)
                        layer.trainable = False
            generator = Model (count_input, discriminator (encoder(count_input)[0]))
            if sf.usingDynamicSizeFactors ():
                generator.add_loss (K.mean (0*z_sf))
            generator.compile(loss='binary_crossentropy'  \
                      if not params['discriminator_wasserstein'] else wasserstein_loss(apply_lambda=True, debug=debug),
                    optimizer=params['optimizer_generator'])
            if debug:
                print ('Non-Trainable Discriminator model')
                discriminator.summary ()

        return vae, encoder, decoder, discriminator, generator

    # Used to debug whether weights are updated in the discriminator/generator stacks

    def print_weights(model, msg):
        print ('Printing weights', msg)
        for layer in model.layers:
            w = layer.get_weights ()
            print ('Layer:', layer.name)
            print (w)

    # Training the network

    def train (X, rawX, sf, vae, encoder, decoder, discriminator, generator, params, mod_args, verbose, debug, tensorboard):
        def adjust_beta (epoch):
            if params['beta'] is None:
                if debug:
                    print ('Setting kl_beta_weight:', float(epoch)/params['n_epochs'])
                K.set_value (mod_args ['kl_beta_weight'], float(epoch)/params['n_epochs'])
                if debug:
                    bval = K.get_value (mod_args['kl_beta_weight'])
                    print ('Changed kl_beta_weight:', bval)

        class BetaChanger(Callback):
            def on_epoch_begin(self, epoch, logs={}):
                adjust_beta (epoch)

        class LRChanger(Callback):
            def on_epoch_begin(self, epoch, logs={}):
                lr = float (K.get_value(self.model.optimizer.lr))
                print ('VAE lr:', lr)
                print ('VAE adam:', self.model.optimizer.get_config())

        class ModelDebugger(Callback):
            def on_epoch_begin(self, epoch, logs={}):
                print ('EPOCH begin:', epoch)
                print ('\tkl_beta_weight=', K.get_value (mod_args ['kl_beta_weight']))
                print ('\tlr=', float (K.get_value(self.model.optimizer.lr)))
            def on_epoch_end(self, epoch, logs={}):
                logs = logs or {}
                loss = logs.get ('loss')
                valloss = logs.get ('val_loss')
                print ('EPOCH end:', epoch)
                print ('\tLOSS=', loss)
                print ('\tVALLOSS=', valloss)

        class ModelInfo(Callback):
            def on_epoch_begin(self, epoch, logs={}):
                print ('EPOCH:', epoch)
                print ('\tkl_beta_weight=', K.get_value (mod_args ['kl_beta_weight']))
                print ('\tlr=', float (K.get_value(self.model.optimizer.lr)))

        beta_changer = BetaChanger ()
        lr_changer = LRChanger ()
        model_debugger = ModelDebugger ()
        model_info = ModelInfo ()

        if tensorboard:
            from . tb_callback import MyTensorBoard
            # https://github.com/keras-team/keras/issues/11433
            #tboard = TensorBoard (log_dir=params ['logdir'], histogram_freq=1, write_grads=True)
            tboard = MyTensorBoard (log_dir=params ['logdir'], histogram_freq=1, write_grads=True)


        if params ['architecture'] == 'vae':
            inputs = {'count_input': X, 'raw_count_input': rawX}
            inputs.update (sf.getSizeFactors ())
            print ('Inputs')
            print (inputs)
            for i in inputs:
                print (len (inputs [i]))
            output = []
            callbacks = [beta_changer, lr_changer]
            if verbose:
                callbacks.append (model_info)
            if debug:
                callbacks.append (model_debugger)
            if tensorboard:
                callbacks.append (tboard)
            if params ['patience_reduce_lr'] > 0:
                lr_cb = ReduceLROnPlateau (monitor='val_loss', patience=params['patience_reduce_lr'],
                          factor=params['factor_reduce_lr'], verbose=verbose)
                callbacks.append (lr_cb)
            loss = vae.fit (inputs, output,
                    callbacks=callbacks,
                    epochs = params ['n_epochs'],
                    batch_size = params ['batch_size'],
                    shuffle=True,
                    validation_split = params ['validation_split'])
            vae.save_weights ('vae.h5')
            lh_vae = loss.history
            lh_d = {}
            lh_g = {}
        elif params ['architecture'] == 'aae':
            lh_vae = {}
            lh_d = {}
            lh_g = {}
            for i in [lh_vae, lh_d, lh_g]:
                i ['loss'] = []
                i ['val_loss'] = []

            sfs = sf.getSizeFactors (as_arrays=True)

            lr_cb = None
            if params ['patience_reduce_lr'] > 0:
                lr_cb = ReduceLROnPlateau (monitor='val_loss', patience=params['patience_reduce_lr'],
                          factor=params['factor_reduce_lr'], verbose=verbose)
                lr_cb.set_model (vae)
                lr_cb.on_train_begin ()
            aae_logs = {}

            if params ['training_method'] == 'train_fit_batch':
                rand_x = np.random.RandomState (42)
                n_batches = int (len(X)/params['batch_size'])
                if (n_batches*params['batch_size']) < len(X):
                    n_batches += 1

            def create_discriminator_data (datapoints, encoder, params, debug):
                if params ['discriminator_pdf'] == 'gaussian':
                    real_latent = np.random.normal(params ['discriminator_prior_mean'],
                                 params ['discriminator_prior_sigma'],
                                 size=(len (datapoints), params ['latent_dim']))
                elif params ['discriminator_pdf'] == 'vonmises':
                    real_latent = np.random.vonmises(params ['discriminator_prior_mean'],
                                 params ['discriminator_prior_kappa'],
                                 size=(len (datapoints), params ['latent_dim']))
                # do we need unshuffled data for this?
                fake_latent = encoder.predict (datapoints)[0]
                if debug:
                    print ('fake_latent')
                    print (fake_latent)
                    for i in fake_latent:
                        print ('\t', np.min (i), np.max (i))
                    print ('real_latent')
                    print (real_latent)
                    for i in real_latent:
                        print ('\t', np.min (i), np.max (i))
                valid = np.ones((len (datapoints), 1))
                fake = np.zeros((len (datapoints), 1))
                discriminator_input = np.concatenate ([fake_latent, real_latent])
                discriminator_labels = np.concatenate ([fake, valid])

                return discriminator_input, discriminator_labels

            # This is no longer fully-tested
            # It will be required for very large datasets
            def train_fit_batch ():
                if debug:
                    print ('TRAIN_FIT_BATCH')
                from sklearn.utils import shuffle
                if sf.usingSizeFactors ():
                    if len(sfs) == 1:
                        X, sfs [0] = shuffle(X, sfs [0], random_state=42)
                    else:
                        X, sfs [0], sfs [1] = shuffle(X, sfs [0], sfs [1], random_state=42)
                else:
                    rand_x.shuffle (X)
                for i in np.arange (n_batches):
                    start = i * params ['batch_size']
                    end = start + params ['batch_size']
                    samples = X [start:end]
                    sample_sfs = []
                    if sf.usingSizeFactors ():
                        for i in sfs:
                            sample_sfs.append (i [start:end])

                    if debug:
                        print ('BATCH:', i, start, end)
                        print (samples)
                        print (samples.shape)
                    output = []
                    inputs = [samples] + sample_sfs
                    vae_loss = vae.fit (inputs, output, epochs = n_fit_vae_epochs,
                            batch_size = len(samples), validation_split = 0.0,
                            shuffle = False)
                    discriminator_input, discriminator_labels = create_discriminator_data (samples,
                              encoder, params, debug)
                    if debug:
                        print ('DISCRIMINATOR')
                    d_loss = discriminator.fit (x=discriminator_input, y=discriminator_labels,
                            epochs=n_fit_d_epochs, batch_size = len(samples), validation_split=0.0,
                            shuffle=False)
                    if debug:
                        print ('GENERATOR')
                    valid = np.ones((len (samples), 1))
                    g_loss = generator.fit (x=samples, y=valid,
                            epochs=n_fit_g_epochs, batch_size = len(samples), validation_split=0.0,
                            shuffle=False)

                    return vae_loss, d_loss, g_loss

            def train_fit ():
                if debug:
                    print ('TRAIN_FIT')
                vae_callbacks = [model_info]
                inputs = {'count_input': X, 'raw_count_input' : rawX}
                inputs.update (sf.getSizeFactors ())
                output = []
                vae_loss = vae.fit (inputs, output,
                        epochs = params ['n_epochs_vae'],
                        callbacks = vae_callbacks,
                        batch_size = params ['batch_size'],
                        shuffle=True,
                        #validation_split=0.0)
                        validation_split = params ['validation_split'])

                discriminator_input, discriminator_labels = create_discriminator_data (X,
                              encoder, params, debug)
                if debug:
                    print ('DISCRIMINATOR')
                d_loss = discriminator.fit (x=discriminator_input, y=discriminator_labels,
                        epochs = params ['n_epochs_discriminator'],
                        batch_size = params ['batch_size'],
                        shuffle=True,
                        #validation_split=0.0)
                        validation_split=params['validation_split'])
                if debug:
                    print_weights (discriminator, 'discriminator after discriminator training')
                    print_weights (vae, 'VAE before generator')
                if debug:
                    print ('GENERATOR')
                valid = np.ones((len (X), 1))
                inputs = {'count_input': X}
                g_loss = generator.fit (x=inputs, y=valid,
                        epochs = params ['n_epochs_generator'],
                        batch_size = params ['batch_size'],
                        shuffle=True,
                        #validation_split=0.0)
                        validation_split=params['validation_split'])
                if debug:
                    print_weights (discriminator, 'discriminator after generator training')
                    print_weights (vae, 'VAE after generator')

                return vae_loss, d_loss, g_loss


            for epoch in range(params ['n_epochs']):
                print ('EPOCH (aae)', epoch)
                adjust_beta (epoch)
                if params ['training_method'] == 'fit_batch':
                    vae_loss, d_loss, g_loss = train_fit_batch ()
                elif params ['training_method'] == 'fit':
                    vae_loss, d_loss, g_loss = train_fit ()

                aae_logs ['loss'] = vae_loss.history['loss'][-1]
                aae_logs ['val_loss'] = vae_loss.history['val_loss'][-1]
                if lr_cb is not None:
                    lr_cb.on_epoch_end (epoch, aae_logs)

                if debug:
                    print ('vae_loss=', vae_loss.history ['loss'])
                    print ('vae_val_loss=', vae_loss.history ['val_loss'])
                    print ('d_loss=', d_loss.history ['loss'])
                    print ('d_val_loss=', d_loss.history ['val_loss'])
                    print ('g_loss=', g_loss.history ['loss'])
                    print ('g_val_loss=', g_loss.history ['val_loss'])

                for h, lh in zip ([vae_loss, d_loss, g_loss], [lh_vae, lh_d, lh_g]):
                    losses = []
                    v_losses = []
                    for i in h.history ['loss']:
                        losses.append (float (i))
                    if params ['training_method'] == 'fit':
                        for i in h.history ['val_loss']:
                            v_losses.append (float (i))
                    lh ['loss'].append (np.min (losses))
                    lh ['val_loss'].append (np.min (v_losses))
                if debug:
                    print ('LOSS')
                    print ('Epoch', epoch, ':', lh_vae ['loss'][-1])
                    print ('VALLOSS')
                    print ('Epoch', epoch, ':', lh_vae ['val_loss'][-1])

        return lh_vae, lh_d, lh_g


    module_args = {'kl_beta_weight': K.variable (params['beta'] if params['beta'] is not None else 1.0)}

    sf = SizeFactors (adata = adata, sf_user = size_factors,
               sf_mode = params ['size_factors'], debug = debug)

    vae, encoder, decoder, discriminator, generator = build (adata.shape [1],
              sf, params, module_args, debug)
    show_model (vae, encoder, decoder, discriminator, generator)

    if raw_adata is None:
        adata.raw = adata.copy ()
    else:
        adata.raw = raw_adata

    if params ['log_preprocess']:
        adata.X = np.log (1 + adata.X)


    c = adata.copy ()
    loss_history, d_loss_history, g_loss_history = train (adata.X, adata.raw.X, sf,
               vae, encoder, decoder, discriminator, generator, params, module_args, verbose, debug, tensorboard)
    adata = c

    inputs = {'count_input': adata.X, 'raw_count_input':adata.raw.X}
    inputs.update (sf.getSizeFactors ())
    if params ['model'] == 'gaussian':
        x = vae.predict (inputs)
    elif params ['model'] == 'zinb':
        x = vae.predict (inputs)[0]

    latent_outputs = encoder.predict (adata.X)
    latent_space = latent_outputs [0]
    latent_activations = {}
    if params ['variational'] and params ['return_latent_activations']:
            latent_activations ['z_mean'] = latent_outputs [1]
            latent_activations ['z_log_var'] = latent_outputs [2]
    if sf.usingDynamicSizeFactors ():
        latent_activations ['z_sf'] = latent_outputs [3]

    adata.X = x
    # so that the scanpy functions will work
    adata.obsm['X_dca'] = latent_space
    adata.uns['vae_loss_history'] = loss_history
    if params ['architecture'] == 'aae':
        adata.uns['d_loss_history'] = d_loss_history
        adata.uns['g_loss_history'] = g_loss_history

    return adata, latent_activations


