
#  Candidate: 8228T

import sys
import os
import argparse
import copy
from argparse import Namespace
import datetime
from pprint import pprint, pformat
import hashlib
import traceback
import time


print ('Running:', sys.argv[0])


default_hidden_size='64,32,64'
default_dca_batch_size=32
default_scvi_batch_size=128
default_my_vae_batch_size=128

default_base_directory = "/local/scratch/cnl29/project"
plot_resolution = 100
#plot_format = "eps"
plot_format = "png"
prog_mnemonic = 'foetal_human'

# plotting colors
myColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
            '#ffffff', '#000000']


def arg_to_filename(i):
    switcher={
        1:'study5317b-star-genecounts.tsv',
        2:'study5317-star-genecounts.tsv',
        3:'study5317-tic158-star-genecounts.tsv',
        4:'study5317-tic161-star-genecounts.tsv',
        5:'study5317-tic172-star-genecounts.tsv',
        6:'immune_control_expression_matrix.txt',
        200:'MergedSamplesBeforeQC.h5ad',
        201:'MergedSamplesAfterQC.h5ad',
        210:'SampleMerged_gaussian_0.1_1.0.h5ad',
        211:'SampleMerged_gaussian_0.2_1.0.h5ad',
        212:'SampleMerged_gaussian_0.5_1.0.h5ad',
        213:'SampleMerged_gaussian_0.1_2.0.h5ad',
        214:'SampleMerged_gaussian_1.0_1.0.h5ad',
        215:'SampleMerged_gaussian_1.0_5.0.h5ad',
        216:'SampleMerged_gaussian_1.0_10.0.h5ad',
        217:'SampleMerged_gaussian_1.0_20.0.h5ad',
        220:'SampleMerged_pink_1.0_.h5ad',
        221:'SampleMerged_pink_2.0_.h5ad',
        222:'SampleMerged_pink_5.0_.h5ad',
        223:'SampleMerged_pink_10.0_.h5ad',
        224:'SampleMerged_pink_20.0_.h5ad',
        300:'MouseCortex.h5ad'
      }
    return switcher.get(i,"Invalid file number")


dca_args = Namespace (log1p = True)


def write_args (args, d_args):
    args_str = pformat (args)
    dargs_str = pformat (d_args)
    print ("Prog args:" + args_str)
    print ("DCA args:" + dargs_str)

    f = open (results_dir + '/sample' + args.sample + '/args.txt', "w")
    f.write ("Program args:\n")
    f.write (args_str)
    f.write ("\n")
    f.write ("DCA args:\n")
    f.write (dargs_str)
    f.close ()



def parse_args():
    parser = argparse.ArgumentParser(description='Gene Analyzer')

    parser.add_argument('--sample', type=str, default='1', dest = "sample",
            help="Sample number (default: 1, alternatives: 2,3,4,5)")
    parser.add_argument('--verbose', dest='verbose',
            action='store_true', help='Verbose (default: False)')
    parser.add_argument('--debug', dest='debug', default = False,
            action='store_true', help='Debug (default: False)')
    parser.add_argument('--tf_debug', dest='tf_debug', default = False,
            action='store_true', help='Debug tensorflow (default: False)')
    parser.add_argument('--tensorboard', dest='tensorboard', default = False,
            action='store_true', help='produce tensorboard logs (default: False)')
    parser.add_argument('--show', dest='show', default = False,
            action='store_true', help='Show plots (default: False)')
    parser.add_argument('--noplot', dest='noplot', default = False,
            action='store_true', help='Do not do plots (default: False)')
    parser.add_argument('--nomgviolinandspace', dest='nomgviolinandspace', default = False,
            action='store_true', help='Do not do marker gene violin and space plots (default: False)')
    parser.add_argument('--nogpu', dest='nogpu', default = False,
            action='store_true', help='Do not use gpu (default: False)')
    parser.add_argument('--thrashgpus', dest='thrashgpus', default = False,
            action='store_true', help='Do not nicely select gpu (default: False)')
    parser.add_argument('--nosave', dest='nosave',
            action='store_true', help='Do not save plots (default: False)')
    parser.add_argument('--noplottitles', dest='noplottitles',
            action='store_true', help='Do not put titles on plots (default: False)')
    parser.add_argument('--nodenoise', dest='nodenoise',
            action='store_true', help='Do not do denoise (default: False)')
    parser.add_argument('--nofiltermaxcount', dest='nofiltermaxcount',
            action='store_true', help='Do not filter max count (default: False)')
    parser.add_argument('--nofoetal', dest='nofoetal',
            action='store_true', help='Not a foetal sample (default: False)')
    parser.add_argument('--regress', dest='regress', default = False,
            action='store_true', help='Do regress_out (default: False)')
    parser.add_argument('--nopreanalysis', dest='nopreanalysis', default = False,
            action='store_true', help='Do not do preanalysis (default: False)')
    parser.add_argument('--extractHVGs', dest='extractHVGs', default = False,
            action='store_true', help='Extract HVGs before denoising (default: False)')
    parser.add_argument('--intermediate_load', dest='intermediate_load', default = False,
            action='store_true', help='Load from intermediate saved h5ad (default: False)')
    parser.add_argument('--intermediate_write', dest='intermediate_write', default = False,
            action='store_true', help='Write intermediate data (default: False)')
    parser.add_argument('--shadowDenoise', dest='shadowDenoise', default = False,
            action='store_true', help='Shadow the denoising (default: False)')
    parser.add_argument('--load_upstream', type=str, default=None, dest = "load_upstream",
            help="Upstream h5ad file to load (default: None; alternatives: after_metadata, after_cropping)")
    parser.add_argument('--background', dest='background', default = False,
            action='store_true', help='Assume running in the background (default: False)')
    parser.add_argument('--use_my_dca', dest='use_my_dca', default = False,
            action='store_true', help='Use My version of DCA (default: False)')
    parser.add_argument('--use_my_scvi', dest='use_my_scvi', default = False,
            action='store_true', help='Use My version of SCVI (default: False)')
    parser.add_argument('--onlyscvifork', dest='onlyscvifork', default = False,
            action='store_true', help='Only fork into a separate scvi pipeline (default: False)')
    parser.add_argument('--noscvifork', dest='noscvifork', default = False,
            action='store_true', help='Do not fork into a separate scvi pipeline (default: False)')
    parser.add_argument('--denoiser', type=str, default='my_vae', dest = "denoiser",
            help="Denoiser to use (default: dca) Alternatives(scvi,my_vae,my_aae")
    parser.add_argument('--denoise_moment', type=str, default='denoise_after_cropping', dest = "denoise_moment",
            help="Moment when to run denoise (default: denoise_after_cropping)")
    parser.add_argument('--denoise_mode', type=str, default='denoise', dest = "denoise_mode",
            help="Mode in which to run denoise (default: denoise, alternatives: latent, full)")
    parser.add_argument('--hidden_size', type=str, default=default_hidden_size,
            dest = 'hidden_size',
            help="Size of hidden layers (default: " + default_hidden_size + ")")
    parser.add_argument('--batch_size', type=int, default=None,
            dest = 'batch_size',
            help="Size of batch layers (default: " + str (default_dca_batch_size) + ")")
    parser.add_argument('--dca_evaluate', type=str, default=None, dest = "dca_evaluate",
            help="Mode for evaluating dca (default: None)")
    parser.add_argument('--emulate_dca_preprocess', default=False, dest = "emulate_dca_preprocess",
            action='store_true', help="scvi to emulate dca preprocessing (default: False)")
    parser.add_argument('--n_latent', default=10, type=int, dest = "n_latent",
            help="n_latent for scvi (default: 10)")
    parser.add_argument('--n_layers', default=1, type=int, dest = "n_layers",
            help="n_layers for scvi (default: 1)")
    parser.add_argument('--n_hidden', default=128, type=int, dest = "n_hidden",
            help="n_hidden for scvi (default: 128)")
    parser.add_argument('--n_epochs', default='1000', type=str, dest = "n_epochs",
            help="n_epochs (default: 1000)")
    parser.add_argument('--optimizer', default='Adam', type=str, dest = "optimizer",
            help="optimizer (default: Adam)")
    parser.add_argument('--validation_split', default=0.1, type=float, dest = "validation_split",
            help="Validation percentage (default: 0.1)")
    parser.add_argument('--lr', default='0.001', type=str, dest = "lr",
            help="lr for scvi (default: 0.001)")
    parser.add_argument('--kl', default=None, type=float, dest = "kl",
            help="kl for scvi (default: None; alternative: float 0<kl<1+z)")
    parser.add_argument('--dropout_rate', default=0.1, type=float, dest = "dropout_rate",
            help="dropout_rate for scvi (default: 0.1)")
    parser.add_argument('--input_dropout_rate', default=0.0, type=float, dest = "input_dropout_rate",
            help="input dropout_rate for dca (default: 0.0)")
    parser.add_argument('--ridge', default=None, type=float, dest = "ridge",
            help="ridge for dca (default: 0.0)")
    parser.add_argument('--l1', default=0.0, type=float, dest = "l1",
            help="l1 coefficient (default: 0.0)")
    parser.add_argument('--l2', default=0.0, type=float, dest = "l2",
            help="l2 coefficient (default: 0.0)")
    parser.add_argument('--l1_enc', default=0.0, type=float, dest = "l1_enc",
            help="l1 coefficient for hidden end/dec layers(default: 0.0)")
    parser.add_argument('--l2_enc', default=0.0, type=float, dest = "l2_enc",
            help="l2 coefficient for hidden end/dec layers(default: 0.0)")
    parser.add_argument('--nobatchnorm', dest='nobatchnorm', default = False,
            action='store_true', help='Do not do batch normalisation(default: False)')
    parser.add_argument('--patience', default=15, type=int, dest = "patience",
            help="patience for scvi early stopping(default: 15)")
    parser.add_argument('--patience_reduce_lr', default=0, type=int, dest = "patience_reduce_lr",
            help="patience for reducing lr(default: 0)")
    parser.add_argument('--factor_reduce_lr', default=0, type=float, dest = "factor_reduce_lr",
            help="factor for reducing lr(default: 0)")
    parser.add_argument('--nosizefactors', default=False, dest = "nosizefactors",
            action='store_true', help='Do not use size factors(default: False)')
    parser.add_argument('--novariational', default=False, dest = "novariational",
            action='store_true', help='Do not use variational latent space with my_vae (default: False)')
    parser.add_argument('--size_factor_algorithm', type=str, default='static', dest = "size_factor_algorithm",
            help="Method of applying size factors (default: static)(alternatives:dynamic,user)")
    parser.add_argument('--clustering_method', type=str, default='louvain', dest = "clustering_method",
            help="Clustering method to use (default: louvain, Alternatives: leiden, full)")
    parser.add_argument('--clustering_quality', type=str, default=None, dest = "clustering_quality",
            help="Clustering quality metrics to compute (If specified: [{silhouette{,calinski_harabaz{,davies_bouldin}}}])")
    parser.add_argument('--clustering_resolution', type=float, default=1.3, dest = "clustering_resolution",
            help="Clustering resolution to use in clustering algorithms (default: 1.3)")
    parser.add_argument('--reconstruction_loss', type=str, default='zinb', dest = "reconstruction_loss",
            help="Reconstruction loss to use with scvi (default: zinb, Alternatives: nb)")
    parser.add_argument('--evaluation_bundle', type=str, default=None, dest = "evaluation_bundle",
            help="Evaluation bundle for metrics (default: None)")
    parser.add_argument('--evaluate_bundle', dest='evaluate_bundle', default = False,
            action='store_true', help='Evaluate metric bundle default: False)')
    parser.add_argument('--pdf', dest='pdf', type=str, default = 'zinb',
            help='PDF my_vae default: zinb)')
    parser.add_argument('--discriminator_pdf', dest='discriminator_pdf', type=str, default = 'gaussian',
            help='Discriminator PDF my_vae default: gaussian; alternative: vonmises)')
    parser.add_argument('--discriminator_wasserstein', dest='discriminator_wasserstein', default = False,
            action='store_true', help='Use Wasserstein loss for my_aae(default: False)')
    parser.add_argument('--discriminator_prior_sigma', dest='discriminator_prior_sigma', type=float, default = 1.0,
            help='Discriminator prior sigma my_aae default: 1.0')
    parser.add_argument('--discriminator_hidden_size', type=str, default='512,216',
            dest = 'discriminator_hidden_size',
            help="Size of discriminator hidden layers (default: " + '512,216' + ")")
    parser.add_argument('--wasserstein_lambda', dest='wasserstein_lambda', type=float, default = 1.0,
            help='Wasserstein lambda for my_aae default: 1.0')
    parser.add_argument('--skip_to', type=str, default=None, dest = "skip_to",
            help="Functionality to skip to (default: None)")
    parser.add_argument('--batch_id', type=str, default=None, dest = "batch_id",
            help="Batch ID for saving results and plots (default: None)")
    parser.add_argument('--base_directory', type=str, default=None,
            dest = "base_directory",
            help="Base directory for files (default: /local/scratch/cnl29/Project)")

    #return parser.parse_args()
    args = parser.parse_args()
    def parse_lr (lr_spec):
        return [float(x) for x in lr_spec.split(',')]
    def parse_epochs (epoch_spec):
        return [int(x) for x in epoch_spec.split(',')]

    args.lr = parse_lr (args.lr)
    if len (args.lr) == 1:
        args.lr = args.lr[0]
    args.n_epochs = parse_epochs (args.n_epochs)
    if len (args.n_epochs) == 1:
        args.n_epochs = args.n_epochs[0]

    return args

def get_optimizer (spec):
    i=spec.find ('{')
    if i != -1:
        optimizer=spec[0:i]
        opt_args=spec[i:]
    else:
        optimizer=spec
        opt_args=None
    return optimizer, opt_args

def dump_prog_args (args):
  print ('Program args:')
  print ('sample:' + args.sample)
  print ('show:' + str (args.show))
  print ('nosave:' + str (args.nosave))
  print ('noplottitles:' + str (args.noplottitles))
  print ('nodenoise:' + str (args.nodenoise))
  print ('nofiltermaxcount:' + str (args.nofiltermaxcount))
  print ('nofoetal:' + str (args.nofoetal))
  print ('noplot:' + str (args.noplot))
  print ('nomgviolinandspace:' + str (args.nomgviolinandspace))
  print ('nogpu:' + str (args.nogpu))
  print ('thrashgpus:' + str (args.thrashgpus))
  print ('regress:' + str (args.regress))
  print ('nopreanalysis:' + str (args.nopreanalysis))
  print ('extractHVGs:' + str (args.extractHVGs))
  print ('intermediate_load:' + str (args.intermediate_load))
  print ('intermediate_write:' + str (args.intermediate_write))
  print ('shadowDenoise:' + str (args.shadowDenoise))
  print ('load_upstream:' + str (args.load_upstream))
  print ('background:' + str (args.background))
  print ('use_my_dca:' + str (args.use_my_dca))
  print ('use_my_scvi:' + str (args.use_my_scvi))
  print ('onlyscvifork:' + str (args.onlyscvifork))
  print ('noscvifork:' + str (args.noscvifork))
  print ('denoiser:' + args.denoiser)
  print ('denoise_moment:' + args.denoise_moment)
  print ('denoise_mode:' + args.denoise_mode)
  print ('hidden_size:' + str (args.hidden_size))
  print ('batch_size:' + str (args.batch_size))
  print ('dca_evaluate:' + str (args.dca_evaluate))
  print ('emulate_dca_preprocess:' + str (args.emulate_dca_preprocess))
  print ('reconstruction_loss:' + str (args.reconstruction_loss))
  print ('n_latent:' + str (args.n_latent))
  print ('n_layers:' + str (args.n_layers))
  print ('n_hidden:' + str (args.n_hidden))
  print ('n_epochs:' + str (args.n_epochs))
  print ('optimizer:' + str (args.optimizer))
  print ('validation_split:' + str (args.validation_split))
  print ('lr:' + str (args.lr))
  print ('kl:' + str (args.kl))
  print ('dropout_rate:' + str (args.dropout_rate))
  print ('input_dropout_rate:' + str (args.input_dropout_rate))
  print ('ridge:' + str (args.ridge))
  print ('l1:' + str (args.l1))
  print ('l2:' + str (args.l2))
  print ('l1_enc:' + str (args.l1_enc))
  print ('l2_enc:' + str (args.l2_enc))
  print ('nobatchnorm:' + str (args.nobatchnorm))
  print ('nosizefactors:' + str (args.nosizefactors))
  print ('novariational:' + str (args.novariational))
  print ('size_factor_algorithm:' + str (args.size_factor_algorithm))
  print ('patience:' + str (args.patience))
  print ('patience_reduce_lr:' + str (args.patience_reduce_lr))
  print ('factor_reduce_lr:' + str (args.factor_reduce_lr))
  print ('skip_to:' + str (args.skip_to))
  print ('clustering_method:' + str (args.clustering_method))
  print ('clustering_resolution:' + str (args.clustering_resolution))
  print ('clustering_quality:' + str (args.clustering_quality))
  print ('verbose:' + str (args.verbose))
  print ('debug:' + str (args.debug))
  print ('tf_debug:' + str (args.tf_debug))
  print ('tensorboard:' + str (args.tensorboard))
  print ('base_directory:' + str (args.base_directory))
  print ('evaluation_bundle:' + str (args.evaluation_bundle))
  print ('evaluate_bundle:' + str (args.evaluate_bundle))
  print ('pdf:' + str (args.pdf))
  print ('discriminator_pdf:' + str (args.discriminator_pdf))
  print ('discriminator_prior_sigma:' + str (args.discriminator_prior_sigma))
  print ('discriminator_hidden_size:' + str (args.discriminator_hidden_size))
  print ('discriminator_wasserstein:' + str (args.discriminator_wasserstein))
  print ('wasserstein_lambda:' + str (args.wasserstein_lambda))
  print ('batch_id:' + str (args.batch_id))


def determine_if_beforePreAnalysis (moment):
    before_PreAnalysis = True

    if moment == 'denoise_after_preanalysis':
        before_PreAnalysis = False ;
    elif moment == 'denoise_after_integratedanalysis':
        before_PreAnalysis = False ;

    return before_PreAnalysis

def determine_if_beforeIntegratedAnalysis (moment):
    before_IntegratedAnalysis = True

    if moment == 'denoise_after_integratedanalysis':
        before_IntegratedAnalysis = False ;

    return before_IntegratedAnalysis



prog_args = parse_args ()


try:
    s = int (prog_args.sample)
    if s >= 300:
        prog_args.nofoetal = True
        print ('Setting nofoetal for sample', prog_args.sample)
except ValueError:
    print ("Bad sample number: ", prog_args.sample)
    exit(1)

if prog_args.denoiser not in ['dca', 'scvi', 'my_vae', 'my_aae']:
    print ("Bad denoiser: ", prog_args.denoiser)
    exit(1)

if prog_args.denoise_moment not in ['denoise_after_metadata', 'denoise_after_cropping', 'denoise_after_preanalysis', 'denoise_after_integratedanalysis']:
    print ("Bad denoise_moment: ", prog_args.denoise_moment)
    exit(1)

if prog_args.clustering_method not in ['louvain', 'leiden', 'full']:
    print ("Bad clustering_method: ", prog_args.clustering_method)
    exit(1)

if prog_args.reconstruction_loss not in ['zinb', 'nb']:
    print ("Bad reconstruction_loss: ", prog_args.reconstruction_loss)
    exit(1)

if prog_args.load_upstream is not None and prog_args.load_upstream not in ['after_metadata', 'after_cropping']:
    print ('Bad load_upstream specification: ', prog_args.load_upstream)
    exit (1)

if prog_args.load_upstream is not None and prog_args.intermediate_load:
    print ('Cannot specify both intermediate_load and load_upstream')
    exit (1)

if prog_args.denoise_mode not in ['denoise', 'latent', 'full']:
    print ("Bad denoise_mode: ", prog_args.denoise_mode)
    exit(1)

if prog_args.denoise_mode == 'full' and prog_args.denoiser == 'dca' and not prog_args.use_my_dca:
    print ("denoise_mode full requires use_my_dca: ")
    exit(1)

optimizer, _ =get_optimizer (prog_args.optimizer)
if optimizer not in ['Adam', 'RMSprop']:
    print ('Optimizer is not Adam or RMSprop', prog_args.optimizer)
    exit (1)

def parse_evaluation_bundle (bundle_spec):
    if bundle_spec is None:
        return []
    return [str(x) for x in bundle_spec.split(',')]

if not prog_args.evaluate_bundle and prog_args.evaluation_bundle is not None:
    bundle_names = parse_evaluation_bundle (prog_args.evaluation_bundle)
    if len(bundle_names) > 1:
        print ("Only one evaluation_bundle name is allowed for collecting metrics")
        exit (1)

if prog_args.evaluate_bundle and prog_args.evaluation_bundle is None:
    print ("No evaluation_bundle specified with evaluate_bundle")
    exit (1)

def parse_clustering_quality (quality_spec):
    if quality_spec is None:
        return ['silhouette', 'calinski_harabaz', 'davies_bouldin']
    return [str(x) for x in quality_spec.split(',')]

if prog_args.clustering_quality is not None:
    qual_names = parse_clustering_quality (prog_args.clustering_quality)
    print (qual_names)
    for i in qual_names:
        if i not in ['silhouette', 'calinski_harabaz', 'davies_bouldin']:
            print ("Unrecognised clustering quality metric specified:", i)
            exit (1)


if prog_args.denoiser in ['my_vae', 'my_aae']:
    if prog_args.denoiser == 'my_vae':
        vae_architecture = 'vae'
    elif prog_args.denoiser == 'my_aae':
        vae_architecture = 'aae'
    if prog_args.pdf not in ['gaussian', 'zinb']:
        print ('Non valid pdf for', prog_args.denoiser, ':', prog_args.pdf)
        exit (1)
    if prog_args.discriminator_pdf not in ['gaussian', 'vonmises']:
        print ('Non valid disciminator pdf for', prog_args.denoiser, ':', prog_args.pdf)
        exit (1)
    if not prog_args.nosizefactors:
        if prog_args.size_factor_algorithm not in ['static', 'dynamic', 'user']:
            print ('Non valid size_factor algorithm for my_vae:', proga_args.size_factor_algorithm)
            exit (1)

if prog_args.denoiser != 'my_aae':
    if isinstance (prog_args.lr, list):
        print ('Multiple learning rates specified')
        exit (1)
    if isinstance (prog_args.n_epochs, list):
        print ('Multiple epochs specified')
        exit (1)

if prog_args.discriminator_pdf != 'gaussian' and vae_architecture == 'aae':
    prog_args.novariational = True


'''
if prog_args.denoise_mode == 'full' and prog_args.denoiser != 'dca':
    print ("denoise_mode full requires dca denoiser: ")
    exit(1)
'''

if prog_args.dca_evaluate is not None:
    if prog_args.denoise_mode == 'full':
        print ("denoise_mode full is not valid for dca_evaluate: ")
        exit(1)
    if prog_args.dca_evaluate not in ['hidden_size']:
        print ("Bad dca_evaluate: ", prog_args.dca_evaluate)
        exit(1)
    if prog_args.denoiser != 'dca':
        print ("dca_evaluate is for dca denoiser, not", prog_args.denoiser)
        exit(1)

'''
if prog_args.denoiser == 'scvi' and prog_args.denoise_mode != 'latent':
    print ("scvi only runs in latent mode")
    exit(1)
'''
 

if prog_args.skip_to is not None and prog_args.skip_to not in ['cell_group_analysis']:
    print ("Bad skip_to: ", prog_args.skip_to)
    exit(1)

#if prog_args.nodenoise:
# will not actually be used
   #prog_args.denoise_mode = 'denoise'


if prog_args.background:
    from matplotlib import use
    use ('Agg')

if prog_args.dca_evaluate is not None:
    prog_args.noplot = True

dump_prog_args (prog_args)


if prog_args.base_directory is not None:
    base_directory = prog_args.base_directory
else:
    sp = os.environ.get ('FOETAL_SCRATCH_PATH')
    if sp is not None:
        base_directory = sp
    else:
        base_directory = default_base_directory


if prog_args.batch_size is None:
    if prog_args.denoiser == 'dca':
        prog_args.batch_size = default_dca_batch_size
    elif prog_args.denoiser == 'scvi':
        prog_args.batch_size = default_scvi_batch_size
    elif prog_args.denoiser == 'my_vae':
        prog_args.batch_size = default_my_vae_batch_size
    elif prog_args.denoiser == 'my_aae':
        prog_args.batch_size = default_my_vae_batch_size

print ('BATCH SIZE is:', prog_args.batch_size)

clustering_methods = ['louvain']
if prog_args.clustering_method == 'louvain':
    clustering_methods = ['louvain']
elif prog_args.clustering_method == 'leiden':
    clustering_methods = ['leiden']
elif prog_args.clustering_method == 'full':
    clustering_methods = ['louvain', 'leiden']



data_dir = base_directory + '/data/RnaSeqGeneMatrix'
support_dir = data_dir + '/support'
raw_dir = data_dir + '/raw'
processed_dir = data_dir + '/processed'
results_dir = base_directory + '/results/dca/RnaSeqGeneMatrix/processed'
plots_dir = base_directory + '/trame/RnaSeqGeneMatrix/pipeline'

base_results_dir = results_dir

if prog_args.batch_id is not None:
    results_dir = results_dir + '/' + prog_args.batch_id
    plots_dir = plots_dir + '/' + prog_args.batch_id

markerListPath  = support_dir + "/Marker_gene_list_foetal.xlsx"




'''
debug_show = False
show_for_plots = prog_args.show if not debug_show else None
'''



import numpy as np
import pandas as pd

from matplotlib import pylab as plt
import seaborn as sns
from matplotlib import cm
import json

import traceback

import scanpy as sc
import anndata
from sklearn import metrics

sc.settings.verbosity = 0
if prog_args.verbose:
    sc.settings.verbosity = 1
sc.set_figure_params(scanpy=True, dpi=70)
sc.settings.figdir = plots_dir + '/sample' + prog_args.sample



def get_metrics_pathname(args, use_batch_id=False):
    metrics_path = (results_dir if use_batch_id else base_results_dir) + '/sample' + args.sample
    return metrics_path

def get_metrics_filename(args, dn_mode):
    metrics_file = get_metrics_pathname (args, use_batch_id=True) + '/basic_metrics'
    metrics_file = metrics_file + '.' + get_plot_label (args, denoise_id = dn_mode)
    metrics_file = metrics_file + '.txt'

    return metrics_file




for i in [results_dir + '/sample' + prog_args.sample,
          plots_dir + '/sample' + prog_args.sample,
          processed_dir + '/sample' + prog_args.sample,
          support_dir + '/processed',
          get_metrics_pathname (prog_args)]:
    os.makedirs (i, exist_ok=True)



def get_plot_label (args, denoise_id = None):

    if args.nodenoise:
        plot_series = 'no_denoise'
    else:
        plot_series = args.denoiser
        plot_series = plot_series + '.' + args.denoise_moment
        if denoise_id is None:
            plot_series = plot_series + '.' + args.denoise_mode
        else:
            plot_series = plot_series + '.' + denoise_id

    if args.regress:
        plot_series = plot_series + '.' + 'regress'

    if args.hidden_size != default_hidden_size:
        ha = args.hidden_size
        ha = ha.replace(',', '_')
        plot_series = plot_series + '.' + ha
    if args.denoiser == 'dca':
        if args.batch_size != default_dca_batch_size:
            plot_series = plot_series + '.' 'batch_size_' + str (args.batch_size)
    elif args.denoiser == 'scvi':
        if args.batch_size != default_scvi_batch_size:
            plot_series = plot_series + '.' 'batch_size_' + str (args.batch_size)
    elif args.denoiser == 'my_vae' or args.denoiser == 'my_aae':
        if args.batch_size != default_my_vae_batch_size:
            plot_series = plot_series + '.' 'batch_size_' + str (args.batch_size)

    return plot_series



def get_slideshow_filename(args, file_id = None):
    slideshow_file = plots_dir + '/sample' + args.sample + '/slideshow'
    if file_id is not None:
        slideshow_file = slideshow_file + '.' + file_id
    else:
        slideshow_file = slideshow_file + '.' + get_plot_label (args, denoise_id = args.denoise_mode)
    slideshow_file = slideshow_file + '.txt'

    return slideshow_file

def add_to_slideshow(fname, args, file_id = None):
    with open (get_slideshow_filename (args, file_id), 'a') as f:
        print (fname, file = f)

def save_plot (filename, args, file_id = None, f=None, plt=None):
    # bbox_inches='tight' ?
    if f is not None:
        f.savefig(filename, format=plot_format, dpi=plot_resolution, bbox_inches='tight')
    if plt is not None:
        plt.savefig(filename, format=plot_format, dpi=plot_resolution, bbox_inches='tight')
    fname = os.path.basename (filename)
    add_to_slideshow (fname, args, file_id)


if not prog_args.noplot:
    f = open (get_slideshow_filename (prog_args,
           'evaluation' if prog_args.evaluate_bundle else None), 'w')
    f.close ()


if prog_args.evaluate_bundle:
    import glob
    def get_denoiser_color (dn):
        for d, c in zip (['dca', 'scvi', 'my_vae', 'my_aae', 'nodenoise'], ['blue', 'red', 'yellow', 'green', 'black']):
            if dn == d:
                return ('xkcd:' + c)
    def get_data (full_data, metric):
        dataset = {}
        prms = []
        data = []
        n = 0
        for i in full_data:
            prms.append (i['params'])
            ptr = i
            try:
                for j in metric:
                    ptr = ptr [j]
                data.append (ptr)
            except KeyError as e:
                print ('Bad data: key not in data', e)
                continue
            if 'hash' in i:
                h = i['hash']
            else:
                h = str (n)
                n += 1
            if not h in dataset:
                dataset [h] = [i['params'],[ptr]]
            else:
                dataset [h][1].append (ptr)
        return prms, data, dataset
    def get_constant_param_list (prms):
        c_list = {}
        v_list = []
        print ('prms:', prms)
        for i in ['extractHVGs', 'filter_genes', 'reconstruction_loss',
                  'n_latent', 'n_layers', 'n_hidden', 'n_epochs', 'lr',
                  'kl', 'dropout_rate', 'input_dropout_rate', 'ridge',
                  'l1', 'l2', 'l1_enc', 'l2_enc',
                  'batchnorm', 'cl_res', 'patience', 'hidden_size',
                  'wasserstein_lambda'
                 ]:
            p_l = []
            for p in prms:
                if i in p:
                    p_l.append (p[i])
            p_s = set (p_l)
            if len (p_s) == 1:
                c_list[i]=p_l[0]
            else:
                v_list.append (i)
        return c_list, v_list
        
    def plot_metric_data (prms, data, dataset, m_spec):
        import matplotlib

        use_std = True

        fig, ax = plt.subplots(1, 1, figsize = (28,16), dpi=100)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        matplotlib.rcParams.update({'errorbar.capsize': 16})

        #https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
        #https://stackoverflow.com/questions/39500265/manually-add-legend-items-python-matplotlib


        c_list, v_list = get_constant_param_list (prms)
        print ('c_list:', c_list)
        print ('v_list:', v_list)
        #y_pos = np.arange(len(data))
        y_pos = np.arange(len(dataset))
        unique_colors = []
        bar_titles = []
        for i, (key, pkt) in enumerate (dataset.items ()):
        #for i, (p, d) in enumerate (zip (prms, data)):
            p = pkt [0]
            _d = pkt [1]
            if len (_d) == 0 or sum (x is None for x in _d) == len (_d):
            #if d is None:
                continue
            #sm = np.sum (_d)
            mx = np.max (_d)
            mn = np.min (_d)
            '''
            d = sm / len (_d)
            '''
            d = np.mean (_d)
            std = (np.std (_d) / np.sqrt (len (_d)))
            clr = get_denoiser_color (p ['denoiser'])
            lbl = None
            if clr not in unique_colors:
                lbl = p ['denoiser']
                unique_colors.append (clr)
            err_plus = std if use_std else (d-mn)
            err_minus = std if use_std else (mx-d)
            #ax.bar(y_pos [i], d, align='center', color = clr, label = lbl, yerr = [[d-mn], [mx-d]])
            ax.bar(y_pos [i], d, align='center', color = clr, label = lbl, yerr = [[err_plus], [err_minus]])
            p_list = []
            for v in v_list:
                if v in p:
                    p_list.append (v + '=' + str (p[v]))
            x_title = ''
            for i, t in enumerate (p_list):
                if i != 0:
                    x_title = x_title + '\n'
                x_title = x_title + t
            #bar_titles.append (p_list)
            bar_titles.append (x_title)

        #print ('bar_titles:', bar_titles)
        ax.set_xticklabels(bar_titles, rotation='vertical')
        ax.set_xticks(y_pos)
        ax.legend ()
        #ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.grid (b=False, axis='x')
        ax.set_ylabel(m_spec)
        ax.set_title(c_list)

        plt.tight_layout ()
    
        m_list = ''
        for i, m in enumerate (m_spec):
            if i != 0:
                m_list = m_list + '_'
            m_list = m_list + m
        fname = plots_dir + '/sample' + prog_args.sample + '/evaluation.' + m_list + '.' + plot_format
        save_plot (fname, prog_args, 'evaluation', f=fig)
        if prog_args.show:
            plt.show ()
        plt.close()


    bundle_names = parse_evaluation_bundle (prog_args.evaluation_bundle)
    json_data = []
    for nm in bundle_names:
        jsonfile_metrics = get_metrics_pathname (prog_args) + '/metrics'
        if prog_args.evaluation_bundle is not None:
            jsonfile_metrics = jsonfile_metrics + '.' + nm
        json_list = glob.glob (jsonfile_metrics + '.*.json')
        for f in json_list:
            json_data.append (json.load (open(f)))
    if prog_args.debug:
        print (json_data)
    for m_spec in [['loss', 'train', 'final'],
                   ['loss', 'test', 'final'],
                   ['cluster_confidence', 'imputed', 'louvain', 'FractionMeanScore'],
                   ['cluster_confidence', 'latent', 'louvain', 'FractionMeanScore'],
                   ['cluster_confidence', 'imputed', 'leiden', 'FractionMeanScore'],
                   ['cluster_confidence', 'latent', 'leiden', 'FractionMeanScore'],
                   ['cluster_confidence', 'imputed', 'louvain', 'FractionCount'],
                   ['cluster_confidence', 'latent', 'louvain', 'FractionCount'],
                   ['cluster_confidence', 'imputed', 'leiden', 'FractionCount'],
                   ['cluster_confidence', 'latent', 'leiden', 'FractionCount'],
                   ['cluster_quality', 'imputed', 'louvain', 'silhouette'],
                   ['cluster_quality', 'imputed', 'leiden', 'silhouette'],
                   ['cluster_quality', 'imputed', 'leiden-scvi', 'silhouette'],
                   ['cluster_quality', 'latent', 'louvain', 'silhouette'],
                   ['cluster_quality', 'latent', 'leiden', 'silhouette'],
                   ['cluster_quality', 'latent', 'leiden-scvi', 'silhouette'],
                   ['cluster_quality', 'imputed', 'louvain', 'calinski_harabaz'],
                   ['cluster_quality', 'latent', 'leiden', 'calinski_harabaz'],
                   ['cluster_quality', 'imputed', 'louvain', 'davies_bouldin'],
                   ['cluster_quality', 'latent', 'leiden', 'davies_bouldin']
                   ]:
        prms, data, dataset = get_data (json_data, m_spec)
        if prog_args.debug:
            print ('dataset')
            print (dataset)
            for p, d in zip (prms, data):
                print (p, d)
        n_none = sum (x is None for x in data)
        if n_none < len (data):
            plot_metric_data (prms, data, dataset, m_spec)
    exit (0)















write_args (prog_args, dca_args)


if not prog_args.nodenoise:
    if prog_args.denoiser == 'dca':
        if determine_if_beforePreAnalysis (prog_args.denoise_moment):
            if not prog_args.use_my_dca:
                from dca.api import dca
            else:
                print ('Using my_dca')
                from my_dca.api import dca
        else:
            if prog_args.denoiser == 'dca':
                from my_dca.api import dca
    elif prog_args.denoiser == 'scvi':
        if not prog_args.use_my_scvi:
            print ('Importing scvi modules')
            from scvi.dataset import AnnDataset
            from scvi.models import *
            from scvi.inference import UnsupervisedTrainer
        else:
            print ('Importing my_scvi modules')
            from my_scvi.dataset import AnnDataset
            from my_scvi.models import *
            from my_scvi.inference import UnsupervisedTrainer
        #from sklearn.manifold import TSNE
        import anndata
    elif prog_args.denoiser in ['my_vae', 'my_aae']:
        from my_vae.my_vae import scrnaseq_vae, scrnaseq_vae_default_params
 





def get_h5ad_filename (args, dca_id, id, denoise_id = None):
    if denoise_id is not None:
        if denoise_id == 'latent':
            return
    dca_id = dca_id.replace('.', '_')
    h5ad_file = processed_dir + '/sample' + args.sample + '/Sample' + args.sample + '_' + dca_id + '_' + id + '.h5ad'
    return h5ad_file

def save_h5ad (adata, args, dca_id, id, denoise_id = None, force_write=False):
    if not force_write and not args.intermediate_write:
        return
    if denoise_id is not None:
        if denoise_id == 'latent':
            return
    dca_id = dca_id.replace('.', '_')
    h5ad_filename = get_h5ad_filename (args, dca_id, id, denoise_id)
    adata.write(h5ad_filename, compression='gzip')

def load_h5ad (args, dca_id, id, denoise_id = None):
    if denoise_id is not None:
        if denoise_id == 'latent':
            return
    dca_id = dca_id.replace('.', '_')
    h5ad_filename = get_h5ad_filename (args, dca_id, id, denoise_id)
    adata = sc.read (h5ad_filename, first_column_names = True)
    return adata

def get_gpu (args):
    import subprocess
    from subprocess import PIPE

    if args.thrashgpus:
        return None

    if args.nogpu:
        print ("get_gpu: nogpu is set...Not using GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = str ("-1")
        print ('CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])
        return None

    f = open("./gpu_id.out", "w")

    subprocess.run(["sh", "../util/gpu_select.sh"], stdout = f)

    f.close ()
    f = open("./gpu_id.out", "r")
    gpu_id = f.read()
    f.close ()

    print ('get_gpu:', gpu_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str (gpu_id)
    print ('CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

    return gpu_id

def examine_adata (adata, id):
    print (id, ': number of zeros in imputed data:', len (np.ravel (adata.X)) - np.count_nonzero (adata.X))
    print (id, ': max value in imputed data:', np.amax (adata.X))
    print (id, ': min value in imputed data:', np.amin (adata.X))
    print (id, ': number of negative values:', np.sum (np.ravel (adata.X) < 0))
    print (id, ': number of nan values:', np.sum (np.isnan (np.ravel (adata.X))))
    print (id, ': number of inf values:', np.sum (np.isinf (np.ravel (adata.X))))
    '''
    try:
        print (id, ': none HVGs:', np.sum (adata.var['highly_variable'].index is None))
        print (id, ': not none HVGs:', np.sum (adata.var['highly_variable'].index is not None))
    except KeyError:
        print ('highly_variable not set')
    np.set_printoptions(threshold=np.inf)
    arr = np.array (adata.var.index)
    print (id, ': index len:', len (arr))
    print (id, ': index:', arr)
    np.set_printoptions(threshold=1000)
    print (id, ': number of negative values:', np.sum (np.ravel (adata.X) < 0))
    print ('Group Names:')
    try:
        group_names = (adata.uns['rank_genes_groups']['names'].dtype.names)
        for i in group_names:
            print (adata.uns['rank_genes_groups']['names'][i])
    except :
        print ('group names not set')
    '''
    print (adata)
    print (adata.X.shape)
    print (adata.X)
    print ('var index:')
    print (adata.var.index)







def plot_evaluated_losses (args, vallosses, losses, hid_sizes, dn_mode):
    if args.noplot:
        return

    ncols = 3
    nrows = (int) (len (vallosses) / ncols)
    if len (vallosses) % ncols != 0:
        nrows = nrows + 1

    f, axs = plt.subplots (nrows, ncols, figsize=(24,14)) 

    idx=0
    for i in range(0, nrows):
        for j in range(0, ncols):
            if idx >= len (vallosses):
                continue
            vl = vallosses [idx]
            l = losses [idx]
            if nrows > 1:
                axs[i][j].plot(range (len (l)), l, range (len (vl)), vl)
                axs[i][j].semilogy()
                axs[i][j].set_xlabel('N. epochs')
                axs[i][j].set_ylabel('Error')
                axs[i][j].set_title(str(hid_sizes [idx]))
            else:
                axs[idx].plot(range (len (l)), l, range (len (vl)), vl)
                axs[idx].semilogy()
                axs[idx].set_xlabel('N. epochs')
                axs[idx].set_ylabel('Error')
                axs[idx].set_title(str(hid_sizes [idx]))
            idx = idx + 1

    plt.legend(['Training set', 'Hold-out set (validation set)'])
    
    fname = plots_dir + '/sample' + args.sample + '/loss.evaluate.' + get_plot_label (args, denoise_id = dn_mode) + '.' + 'hidden_size' + '.' + plot_format
    save_plot (fname, args, f=f)
    if args.show:
        plt.show()
    plt.close()

def plot_dca_loss (adata, p_metrics, args, dn_mode, id = None):
    valloss = adata.uns['dca_loss_history']['val_loss']
    loss = adata.uns['dca_loss_history']['loss']

    print ("DCA plot loss:")
    print ('val_loss:', valloss)
    print ('loss:', loss)

    with open (get_metrics_filename (args, dn_mode), 'a') as f:
        print ('Loss (train):', valloss [-1], file = f)
        print ('Loss (test):', loss [-1], file = f)

    p_metrics ['loss']['train']['full'] = valloss
    p_metrics ['loss']['train']['final'] = valloss [-1]
    p_metrics ['loss']['test']['full'] = loss
    p_metrics ['loss']['test']['final'] = loss [-1]

    fname = results_dir + '/sample' + args.sample + '/loss.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + 'txt'
    with open (fname, 'w') as f:
        print ('val_loss:', file = f)
        print (valloss, file = f)
        print ('loss:', file = f)
        print (loss, file = f)


    if args.noplot:
        return

    i = range(len(loss))

    f, ax = plt.subplots (1, 1, figsize=(4,4)) 
    ax.plot(i, loss, i, valloss)
    ax.semilogy()
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Error')
    plt.legend(['Training set', 'Hold-out set (validation set)'])
    ax.set_title('dca')
    
    fname = plots_dir + '/sample' + args.sample + '/loss.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, args, f=f)
    if args.show:
        plt.show()
    plt.close()




def plot_scvi_loss (adata, trainer, args, dn_mode, p_metrics, id = None):
    ll_train_set = trainer.history["ll_train_set"]
    ll_test_set = trainer.history["ll_test_set"]

    print ("SCVI plot loss:")
    print ('ll_train_set:', ll_train_set)
    print ('len ll_train_set:', len (ll_train_set))
    print ('ll_test_set:', ll_test_set)
    print ('len ll_test_set:', len (ll_test_set))

    with open (get_metrics_filename (args, dn_mode), 'a') as f:
        print ('Loss (train):', ll_train_set [-1], file = f)
        print ('Loss (test):', ll_test_set [-1], file = f)

    p_metrics ['loss']['train']['full'] = ll_train_set
    p_metrics ['loss']['train']['final'] = ll_train_set [-1]
    p_metrics ['loss']['test']['full'] = ll_test_set
    p_metrics ['loss']['test']['final'] = ll_test_set [-1]

    fname = results_dir + '/sample' + args.sample + '/loss.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + 'txt'
    with open (fname, 'w') as f:
        print ('train_set:', file = f)
        print (ll_train_set, file = f)
        print ('test_set:', file = f)
        print (ll_test_set, file = f)

    if args.noplot:
        return

    f, ax = plt.subplots (1, 1, figsize=(18,10)) 

    x = np.linspace(0,args.n_epochs,(len(ll_train_set)))
    ax.plot(x, ll_train_set, x, ll_test_set)

    ax.set_xlabel('N. epochs')
    ax.set_ylabel('Error')
    ax.set_title('scvi')
    
    fname = plots_dir + '/sample' + args.sample + '/loss.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, args, f=f)
    if args.show:
        plt.show()
    plt.close()



def plot_my_loss (adata, args, dn_mode, p_metrics, model_element, losses, id = None):

    print ('loss for', model_element, ':', losses [0])
    print ('val_loss for', model_element, ':', losses [1])

    with open (get_metrics_filename (args, dn_mode), 'a') as f:
        print ('Loss ('+model_element+')(train):', losses[0][-1], file = f)
        print ('Loss ('+model_element+')(test):', losses [1][-1], file = f)

    if model_element == 'vae':
        p_metrics ['loss']['train']['full'] = losses [0]
        p_metrics ['loss']['train']['final'] = losses [0][-1]
        p_metrics ['loss']['test']['full'] = losses [1]
        p_metrics ['loss']['test']['final'] = losses [1][-1]
    else:
        p_metrics ['loss'][model_element]['train']['full'] = losses [0]
        p_metrics ['loss'][model_element]['train']['final'] = losses [0][-1]
        p_metrics ['loss'][model_element]['test']['full'] = losses [1]
        p_metrics ['loss'][model_element]['test']['final'] = losses [1][-1]

    fname = results_dir + '/sample' + args.sample + '/loss.' + model_element + '.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + 'txt'
    with open (fname, 'w') as f:
        print ('loss:', file = f)
        print (losses [0], file = f)
        print ('val_loss:', file = f)
        print (losses [1], file = f)

    if args.noplot:
        return

    i = range(len(losses [0]))

    f, ax = plt.subplots (1, 1, figsize=(4,4)) 
    ax.plot(i, losses [0], i, losses [1])
    ax.semilogy()
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Error')
    plt.legend(['Training set', 'Hold-out set (validation set)'])
    title = args.denoiser + '(' + model_element + ')'
    ax.set_title(title)
    
    fname = plots_dir + '/sample' + args.sample + '/loss.' + model_element + '.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, args, f=f)
    if args.show:
        plt.show()
    plt.close()

def plot_my_vae_loss (adata, args, dn_mode, p_metrics, id = None):

    print ('MY_' + 'VAE' if args.denoiser == 'my_vae' else 'AAE', 'plot loss:')

    plot_my_loss (adata, args, dn_mode, p_metrics, 'vae',
        [adata.uns['vae_loss_history']['loss'], adata.uns['vae_loss_history']['val_loss']],
        id = id)
    if args.denoiser == 'my_aae':
        plot_my_loss (adata, args, dn_mode, p_metrics, 'discriminator',
            [adata.uns['d_loss_history']['loss'], adata.uns['d_loss_history']['val_loss']],
            id = id)
        plot_my_loss (adata, args, dn_mode, p_metrics, 'generator',
            [adata.uns['g_loss_history']['loss'], adata.uns['g_loss_history']['val_loss']],
            id = id)



def plot_latent_distributions (adata, args, dn_mode, mean_vals, var_vals, lg = True, id = None):
    if args.noplot:
        return

    from random import randint
    import scipy.stats as stats
    import math

    means = []
    variances = []
    for i in range (10):
        j = randint (0, adata.X.shape [0] - 1)
        means.append (mean_vals [j,:])
        variances.append (var_vals [j,:])

    for i, (m_l, v_l) in enumerate (zip (means, variances)):
        print ('Random means and variances:', i)
        f, axs = plt.subplots (4, 3, figsize=(12,8), sharey=True) 
        for idx, (mu, lv, ax) in enumerate (zip (m_l, v_l, axs.ravel())):
            v = lv
            if lg:
                v = np.exp (lv)
            sigma = math.sqrt (v)
            #x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            #x = np.linspace(mu-5, mu+5, 100)
            x = np.linspace(-5, 5, 100)
            print ('pdf: mu', mu, ' sigma', sigma)
            ax.plot(x, stats.norm.pdf(x, mu, sigma))
            #ax.set_xlabel('x')
    
        fname = plots_dir + '/sample' + args.sample + '/latent_distributions.' + str (i) + '.' + get_plot_label (args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
        save_plot (fname, args, f=f)
        if args.show:
            plt.show()
        plt.close()


def run_dca_evaluate (adata, args, moment, mode, normalize_per_cell = True, log1p = True, scale = True, bypass_norm = False):

    hid_sizes = [[64,32,64], [64,16,64], [64, 8, 64], [128, 64, 128], [128, 32, 128], [128,16,128], [128, 8, 128]]

    if args.dca_evaluate == 'hidden_size':
        vallosses = []
        losses = []
        for i in hid_sizes:
            adata_evaluate = adata.copy ()
            gpu_id = get_gpu (args)
            print ("DCA EVALUATE for hidden size:", i)
            if determine_if_beforePreAnalysis (args.denoise_moment):
                 adata_evaluate = dca (adata_evaluate, mode = mode, copy = True,
                     normalize_per_cell = normalize_per_cell, log1p = log1p, scale = scale,
                     hidden_size = i,
                     verbose = True, return_info = True)
            else:
                 adata_evaluate = dca (adata_evaluate, mode = mode, copy = True,
                     normalize_per_cell = normalize_per_cell, log1p = log1p, scale = scale,
                     hidden_size = i,
                     verbose = True, return_info = True,
                     bypass_normalization_check = bypass_norm)
            vl = adata_evaluate.uns['dca_loss_history']['val_loss']
            l = adata_evaluate.uns['dca_loss_history']['loss']
            vallosses.append (vl)
            losses.append (l)
        print ('Hidden size evaluation:')
        print (vallosses)
        print (losses)
        plot_evaluated_losses (args, vallosses, losses, hid_sizes, mode)


#copied from dca.io.py
def write_text_matrix(matrix, args, filename, rownames=None, colnames=None, transpose=False):
    if not args.intermediate_write:
        return
    if isinstance (matrix, anndata.AnnData):
        X = matrix.X
        colnames = matrix.var_names.values
        rownames = matrix.obs_names.values
    else:
        X = matrix
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(X, index=rownames, columns=colnames).to_csv(filename,
                 sep='\t',
                 index=(rownames is not None),
                 header=(colnames is not None),
                 float_format='%.6f')




def run_dca (adata, pmetrics, args, moment, mode, normalize_per_cell = True, log1p = True, scale = True, bypass_norm = False):



    save_h5ad (adata, prog_args, dca_id = get_plot_label (args, denoise_id = mode), id = 'BeforeDCA')

    dca_hidden_size = [int(x) for x in prog_args.hidden_size.split(',')]

    network_kwds = {}

    if args.ridge != None:
        network_kwds ['ridge'] = args.ridge
    network_kwds ['input_dropout'] = args.input_dropout_rate
    network_kwds ['l1_coef'] = args.l1
    network_kwds ['l2_coef'] = args.l2
    network_kwds ['l1_enc_coef'] = args.l1_enc
    network_kwds ['l2_enc_coef'] = args.l2_enc

    print ('hidden_size :', dca_hidden_size)

    print ('Running DCA____________________________________', mode)
    print (adata)
    print (adata.X)
    print ('adata shape:', adata.X.shape)
    print ('moment: ', moment)
    print ('mode: ', mode)
    print ('normalize_per_cell: ', normalize_per_cell)
    print ('log1p: ', log1p)
    print ('hidden_size: ', dca_hidden_size)
    print ('batch_size: ', args.batch_size)
    print ('ridge: ', args.ridge)
    print ('l1: ', args.l1)
    print ('l2: ', args.l2)
    print ('l1_enc: ', args.l1_enc)
    print ('l2_enc: ', args.l2_enc)
    print ('hidden_dropout: ', args.dropout_rate)
    print ('input_dropout: ', args.input_dropout_rate)
    print ('batchnorm: ', not args.nobatchnorm)
    print (network_kwds)
    print (datetime.datetime.now ())
    start_time = time.time ()

    gpu_id = get_gpu (prog_args)

    if args.debug:
        cmp_ad = adata.copy ()

    examine_adata (adata, "DCA before")

    if determine_if_beforePreAnalysis (args.denoise_moment) and not args.use_my_dca:
        adata = dca (adata, mode = mode, copy = True,
            normalize_per_cell = normalize_per_cell, log1p = log1p, scale = scale,
            hidden_size = dca_hidden_size,
            batch_size = args.batch_size,
            hidden_dropout=args.dropout_rate,
            batchnorm=not args.nobatchnorm,
            network_kwds = network_kwds,
            verbose = True, return_info = True)
    else:
        #mode=full is allowed here
        adata = dca (adata, mode = args.denoise_mode, copy = True,
            normalize_per_cell = normalize_per_cell, log1p = log1p, scale = scale,
            hidden_size = dca_hidden_size,
            batch_size = args.batch_size,
            hidden_dropout=args.dropout_rate,
            batchnorm=not args.nobatchnorm,
            network_kwds = network_kwds,
            verbose = True, return_info = True,
            bypass_normalization_check = bypass_norm)

    if args.debug:
        print ('adata_change =', adata.X == cmp_ad.X)

    examine_adata (adata, "DCA after")

    plot_dca_loss (adata, pmetrics, args, mode, id = None)

    print ('Finished running DCA_________________________________________________')
    end_time = time.time ()
    print (datetime.datetime.now ())
    print (adata)
    print (adata.X)
    print ('adata shape:', adata.X.shape)
    if mode == 'latent':
        print ('adata latent shape:', adata.obsm ['X_dca'].shape)


    print ('Start writing DCA______________________________________________')

    save_h5ad (adata, prog_args, dca_id = get_plot_label (args, denoise_id = mode), id = 'AfterDCA')

    pmetrics ['execution_time'] = end_time - start_time

    print ('Finished writing DCA_______________________________________________')

    return adata


def run_scvi (adata, args, moment, mode, ctrl, p_metrics):

    # there appears not to be a mechanism for creating an AnnDataset in-memeory
    save_h5ad (adata, args, dca_id = get_plot_label (args, denoise_id = mode), force_write=True, id = 'BeforeSCVI')
    fname = get_h5ad_filename (args, dca_id = get_plot_label (args, denoise_id = mode), id= 'BeforeSCVI')

    # if the file is in an absolute path, this is all that is needed
    gene_dataset = AnnDataset(fname)

    print ('Running SCVI', mode)
    if args.debug:
        print (adata)
        print (adata.X)
        print ('adata shape:', adata.X.shape)
        print ('moment: ', moment)
        print ('mode: ', mode)
        print ('reconstruction_loss: ', args.reconstruction_loss)
        print (datetime.datetime.now ())
    start_time = time.time ()

    gpu_id = get_gpu (prog_args)

    use_batches=False

    if args.debug:
        cmp_ad = adata.copy ()

    examine_adata (adata, "SCVI before")

    print ('scvi n_latent =', args.n_latent)
    print ('scvi n_layers =', args.n_layers)
    print ('scvi n_hidden =', args.n_hidden)
    print ('scvi n_epochs =', args.n_epochs)
    print ('scvi use_batches =', use_batches)
    print ('scvi n_batch =', args.batch_size)
    print ('scvi lr =', args.lr)
    print ('scvi kl =', args.kl)
    print ('scvi dropout_rate =', args.dropout_rate)
    print ('scvi patience =', args.patience)

    # emulate dca pre-processing

    if args.emulate_dca_preprocess:
        sc.pp.filter_genes (adata.X, min_counts=1)
        sc.pp.log1p(adata)

        print ('filter_genes')
        '''
        sc.pp.normalize_per_cell(adata)
        sc.pp.scale(adata)
        '''


    examine_adata (adata, "SCVI before/after preprocessing")


    dispersion = 'gene'
    #dispersion = 'gene-cell'


    # threshold ??
    early_stopping_args = dict (patience=args.patience)



    vae = VAE(gene_dataset.nb_genes, dispersion = dispersion,
             n_latent = args.n_latent, n_layers = args.n_layers, n_hidden = args.n_hidden,
             reconstruction_loss = args.reconstruction_loss,
             log_variational = not args.emulate_dca_preprocess,
             dropout_rate = args.dropout_rate,
             n_batch=gene_dataset.n_batches * use_batches)

    #https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch
    #https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
    print ('SCVI MODEL:')
    print (vae)
    print ('children:')
    '''
    print (vae.children)
    for child in vae.children():
        print ('child:')
        print (child)
        for param in child.parameters():
            print ('params:')
            print (param)
    '''
    for name, module in vae.named_children():
        print (name)
        if name == 'z_encoder':
            print (module)
            for c_name, c_module in module.named_children():
                print ('\t', c_name)
    print ('by variable:')
    print (vae.z_encoder)
    print (vae.z_encoder.mean_encoder)
    #https://seba-1511.github.io/tutorials/beginner/former_torchies/nn_tutorial.html

    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(input))
        print('input[0]: ', type(input[0]))
        print('output: ', type(output))
        print('')
        print('input size:', input[0].size())
        print('output size:', output.data.size())
        print('output norm:', output.data.norm())
        print('output data:', output.data)

    #vae.z_encoder.mean_encoder.register_forward_hook(printnorm)


    if not args.use_my_scvi:
        trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.75,
                              use_cuda=not args.nogpu,
                              frequency=5,
                              kl=args.kl,
                              verbose = args.verbose,
                              early_stopping_kwargs=early_stopping_args)
    else:
        trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.75,
                              use_cuda=not args.nogpu,
                              frequency=5,
                              kl=args.kl,
                              batch_size = args.batch_size,
                              verbose = args.verbose,
                              early_stopping_kwargs=early_stopping_args)
    trainer.train(n_epochs=args.n_epochs, lr=args.lr)

    print ('TRAINING FINISHED')

    full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
    latent, batch_indices, labels = full.sequential().get_latent()
    batch_indices = batch_indices.ravel()

    means, variances, l = full.sequential().get_z_x()
    np.set_printoptions(threshold=np.inf)
    print ('means:')
    print (means.shape)
    print (means)
    print ('variances:')
    print (variances.shape)
    print (variances)
    print ('latent:')
    print (l.shape)
    print (l)
    print ('scvi_latent:')
    print (latent.shape)
    print (latent)
    np.set_printoptions(threshold=1000)

    variances = np.power (variances, 2)


    if args.debug:
        print ('latent space after scvi')
        print (len (latent))
        print (latent)
        np_latent = np.array (latent)
        print (np_latent.shape)

    latent = l

    imputed_values = full.sequential ().imputation ()

    plot_scvi_loss (adata, trainer, args, mode, p_metrics, id = None)

    '''
    from random import randint
    m_l = []
    v_l = []
    for i in range (10):
        j = randint (0, adata.X.shape [0] - 1)
        m_l.append (means [j,:])
        v_l.append (variances [j,:])
    '''
    plot_latent_distributions (adata, args, mode, means, variances, lg = False, id = None)

    #adata = anndata.AnnData (latent)

    adata.obsm ['X_dca'] = latent
    if mode == 'denoise':
        adata.X = imputed_values

    examine_adata (adata, "SCVI after")

    adata = adata.copy ()

    if args.debug:
        print ('adata_change =', adata.X == cmp_ad.X)

    if args.debug:
        print ('Finished running SCVI')
        print (datetime.datetime.now ())
        print (adata)
        print ('adata.X')
        print (adata.X)
        print ('adata shape:', adata.X.shape)
        print ('adata latent shape:', adata.obsm ['X_dca'].shape)
    end_time = time.time ()

    p_metrics ['execution_time'] = end_time - start_time

    print ('Start writing SCVI______________________________________________')

    save_h5ad (adata, prog_args, dca_id = get_plot_label (args, denoise_id = mode), id = 'AfterSCVI')

    ctrl ['dataset'] = gene_dataset
    ctrl ['latent'] = latent
    ctrl ['imputed_values'] = imputed_values
    ctrl ['trainer_posterior'] = full
    ctrl ['batch_indices'] = batch_indices
    ctrl ['trainer'] = trainer

    return adata



def run_my_vae (adata, p_metrics, args, moment, mode):

    if args.shadowDenoise:
        adata = load_h5ad (args, dca_id = get_plot_label (args, denoise_id = mode), id = 'AfterMY_VAE')
        return adata

    save_h5ad (adata, prog_args, dca_id = get_plot_label (args, denoise_id = mode), id = 'BeforeMY_VAE')

    print ('Running MY_VAE____________________________________', mode)
    print (adata)
    print (adata.X)
    print ('adata shape:', adata.X.shape)
    print ('batch_size: ', args.batch_size)
    print ('n_epochs: ', args.n_epochs)
    print (datetime.datetime.now ())
    start_time = time.time ()

    gpu_id = get_gpu (prog_args)

    if args.debug:
        cmp_ad = adata.copy ()

    examine_adata (adata, "MY_VAE before")

    raw_adata = adata.copy ()

    sc.pp.filter_genes (adata.X, min_counts=1)
    if args.emulate_dca_preprocess:
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)


    prms = scrnaseq_vae_default_params ()
    prms ['architecture'] = vae_architecture
    prms ['model'] = args.pdf
    prms ['log_preprocess'] = not args.emulate_dca_preprocess
    prms ['latent_dim'] = args.n_latent
    prms ['batch_size'] = args.batch_size
    prms ['validation_split'] = args.validation_split
    prms ['patience_reduce_lr'] = args.patience_reduce_lr
    prms ['factor_reduce_lr'] = args.factor_reduce_lr
    prms ['beta'] = args.kl
    if args.hidden_size != default_hidden_size:
        prms ['hidden_structure'] = [int(x) for x in args.hidden_size.split(',')]
    prms ['return_latent_activations'] = True
    prms ['dropout'] = args.dropout_rate
    prms ['input_dropout'] = args.input_dropout_rate
    prms ['l1'] = args.l1
    prms ['l2'] = args.l2
    prms ['l1_enc'] = args.l1_enc
    prms ['l2_enc'] = args.l2_enc
    if args.ridge is not None:
        prms ['ridge'] = args.ridge
    if args.nosizefactors:
        prms ['size_factors'] = None
    else:
        prms ['size_factors'] = args.size_factor_algorithm
    prms ['variational'] = not args.novariational
    prms ['discriminator_pdf'] = args.discriminator_pdf
    prms ['discriminator_prior_sigma'] = args.discriminator_prior_sigma
    prms ['discriminator_wasserstein'] = args.discriminator_wasserstein
    prms ['discriminator_hidden_structure'] = [int(x) for x in args.discriminator_hidden_size.split(',')]
    prms ['wasserstein_lambda'] = args.wasserstein_lambda
    from keras.optimizers import Adam, RMSprop
    optimizer, optimizer_args = get_optimizer (args.optimizer)
    if optimizer_args is not None:
        optimizer_args = json.loads (optimizer_args)
    else:
        optimizer_args = {}
    # https://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
    if not isinstance (args.lr, list):
        #prms ['optimizer'] = Adam (args.lr)
        #optimizer_args = {'lr': 0.004, 'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'decay': 0.0, 'epsilon': 1e-07, 'amsgrad': False}
        if optimizer_args is None:
            optimizer_args ['lr'] = args.lr
        if optimizer == 'Adam':
            optimizer = Adam (**optimizer_args)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop (**optimizer_args)
        prms ['optimizer'] = optimizer
    else:
        #prms ['optimizer'] = Adam (args.lr[0])
        if optimizer_args is None:
            optimizer_args ['lr'] = args.lr[0]
        if optimizer == 'Adam':
            optimizer = Adam (**optimizer_args)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop (**optimizer_args)
        prms ['optimizer'] = optimizer
        if len (args.lr) > 1:
            prms ['optimizer_discriminator'] = Adam (args.lr[1])
        if len (args.lr) > 2:
            prms ['optimizer_generator'] = Adam (args.lr[2])
    if not isinstance (args.n_epochs, list):
        prms ['n_epochs'] = args.n_epochs
    else:
        prms ['n_epochs'] = args.n_epochs[0]
        if len (args.n_epochs) > 1:
            prms ['n_epochs_vae'] = args.n_epochs [1]
        if len (args.n_epochs) > 2:
            prms ['n_epochs_discriminator'] = args.n_epochs [2]
        if len (args.n_epochs) > 3:
            prms ['n_epochs_generator'] = args.n_epochs [3]

    if prms ['discriminator_pdf'] != 'gaussian' and vae_architecture == 'aae':
        prms ['variational'] = False

    for k, v in prms.items():
        print (k, v)
        print (type (v))
        if isinstance (v, Adam):
            print (v.get_config ())
        elif isinstance (v, RMSprop):
            print (v.get_config ())

    adata, latent_activations = scrnaseq_vae (adata, raw_adata = raw_adata, params = prms,
            verbose = args.verbose, debug = args.debug,
            tf_debug = args.tf_debug, tensorboard = args.tensorboard)

    if not args.novariational:
        print ('latent_means:')
        print (latent_activations ['z_mean'])
        print ('latent_logvars:')
        print (latent_activations ['z_log_var'])
    print ('latent_space:')
    print (adata.obsm['X_dca'])
    if args.size_factor_algorithm == 'dynamic':
        print ('dynamic size_factors:')
        print (latent_activations ['z_sf'])

    if not args.novariational:
        plot_latent_distributions (adata, args, mode,
               latent_activations ['z_mean'], latent_activations ['z_log_var'],
               lg = True, id = None)


    if args.debug:
        print ('adata_change =', adata.X == cmp_ad.X)

    examine_adata (adata, "MY_VAE after")

    plot_my_vae_loss (adata, args, mode, p_metrics, id = None)

    print ('Finished running MY_VAE_________________________________________________')
    end_time = time.time ()
    print (datetime.datetime.now ())
    print (adata)
    print (adata.X)
    print ('adata shape:', adata.X.shape)
    if mode == 'latent':
        print ('adata latent shape:', adata.obsm ['X_dca'].shape)

    p_metrics ['execution_time'] = end_time - start_time


    print ('Start writing MY_VAE______________________________________________')

    save_h5ad (adata, prog_args, dca_id = get_plot_label (args, denoise_id = mode), id = 'AfterMY_VAE',
          force_write=True)

    return adata




def denoise_decide (adata, pc, pmetrics, args, dn_mode, moment = 'denoise_after_metadata'):

    if args.nodenoise:
        return adata

    if adata is None:
        return adata

    ran_this_moment = False

    print ('adata denoise_decide BEFORE:')
    print ('adata shape = ', adata.shape)
    print (adata)
    print ('-------------------------')


    if args.denoise_moment == moment :
        before_PreAnalysis = determine_if_beforePreAnalysis (moment)
        if args.dca_evaluate is not None:
            adata = run_dca_evaluate (adata, args, moment, dn_mode,
                  normalize_per_cell = before_PreAnalysis,
                  log1p = before_PreAnalysis,
                  scale = before_PreAnalysis,
                  bypass_norm = not before_PreAnalysis)
            exit (0)
        else:
            if not pc ['denoising_completed']:
                pc ['denoising_completed'] = True
                ran_this_moment = True
                dn_done = False
                for i in range (20):
                    if dn_done:
                        break
                    if i > 0:
                        time.sleep (20)
                    try:
                        if args.denoiser == 'dca':
                            tmp_adata = run_dca (adata, pmetrics, args, moment, dn_mode,
                                  normalize_per_cell = before_PreAnalysis,
                                  log1p = before_PreAnalysis,
                                  scale = before_PreAnalysis,
                                  bypass_norm = not before_PreAnalysis)
                        elif args.denoiser == 'scvi':
                            tmp_adata = run_scvi (adata, args, moment, dn_mode, pc ['scvi_control'], pmetrics)
                        elif args.denoiser in ['my_vae', 'my_aae']:
                            tmp_adata = run_my_vae (adata, pmetrics, args, moment, dn_mode)
                        dn_done = True
                    except RuntimeError as e:
                        # assumption that runtime errors are with the GPUs
                        print (e)
                        traceback.print_exc()
                if dn_done:
                    adata = tmp_adata
                else:
                    raise RuntimeError ('Unable to perform denoising')

    print ('adata in denoise_decide AFTER:')
    print ('adata shape = ', adata.shape)
    print (adata)
    print ('-------------------------')
    print (adata.X)
    print ('-------------------------')

    if args.denoise_mode == 'full':
        if dn_mode == 'denoise':
            if ran_this_moment:
                pc['created_copy'] = adata.copy ()
                pc[moment] = pc['created_copy']
            else:
                if pc['created_copy'] is not None:
                    pc[moment] = pc['created_copy']
                else:
                    pc[moment] = pc['initial_copy']
        else:
            adata = pc[moment]

    return adata



def get_hvg_constants ():

    min_mean=0.1
    max_mean=10
    min_disp=0.25

    return min_mean, max_mean, min_disp



def getCellGroupColors (cell_groups):

    #https://matplotlib.org/tutorials/colors/colors.html
    # https://blog.xkcd.com/2010/05/03/color-survey-results/
    #https://xkcd.com/color/rgb/

    grp_list = []
    for key, grp in cell_groups.items ():
        grp_list.append (str (grp))
    grp_list = set (grp_list)

    colors = ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange', 'yellow',
            'periwinkle', 'tan', 'ochre', 'grey', 'magenta', 'salmon', 'beige', 'forest green',
            'light blue', 'light green', 'light purple', 'peach', 'mustard',
            'steel', 'ocean']

    print ('Number of groups for coloring=', len (grp_list))
    print ('Number of colors=', len (colors))

    color_table = {}
    color_idx = 0
 
    for key, grp in cell_groups.items ():
        if str (grp) not in color_table:
            if color_idx >= len (colors):
                color_table [str (grp)] = 'xkcd:' + colors [-1]
            else:
                color_table [str (grp)] = 'xkcd:' + colors [color_idx]
                color_idx += 1
 
    print (color_table)

    return color_table







#def calcHRGFreqTable (adata_with_gene_rank_observations, cell_groups, save_file = None, debug = False, verbose = False, attrib_split_groups = True, return_both = False):
def calcHRGFreqTable (group_names, all_gene_names, all_scores, cell_groups, save_file = None, debug = False, verbose = False, attrib_split_groups = True, return_both = False):

    #group_names = (adata_with_gene_rank_observations.uns['rank_genes_groups']['names'].dtype.names)
    
    all_genes_matrix = []
    marker_genes_matrix = []
    undefined_list = []
    defined_list = []
    cnt = 0

    for count, group_name in enumerate(group_names):
        if verbose:
            print ('Cluster:', count, '.........', group_name)
        #gene_names = adata_with_gene_rank_observations.uns['rank_genes_groups']['names'][group_name]
        #scores = adata_with_gene_rank_observations.uns['rank_genes_groups']['scores'][group_name]
        gene_names = all_gene_names[group_name]
        scores = all_scores[group_name]
    
        if verbose: 
            print ('Gene names')
            print (gene_names)
            print (len (gene_names))
            print (set(gene_names))
            print (len (set(gene_names)))
            print ('Scores')
            print (scores)
            print (len (scores))
            for (g, s) in zip (gene_names, scores):
                print ('Gene:', g, ' Score:', s)
    
        cluster_genes = []
        cluster_marker_genes = []
        undefined_score = 0
    
        def get_gene_group_name (gene):
            ggrp = None
            if gene in cell_groups:
                nm = cell_groups [gene]
                #ggrp = nm [0]
                ggrp = nm
            return ggrp

        if verbose: 
            print ('Assembling marker gene list')
        for gene_index, gene in enumerate (gene_names):
            ggrp = None
            ggrp = get_gene_group_name (gene)
            if ggrp != None :
                cnt += 1
                defined_list.append (gene)
            else:
                undefined_list.append (gene)
            x = [ggrp, gene, scores [gene_index], gene_index]
            cluster_genes.append (x)
            if ggrp is not None:
                cluster_marker_genes.append (x)
            else:
                undefined_score = undefined_score + scores [gene_index]
        cluster_marker_genes.append ([['UNDEFINED'], None, undefined_score, -1])
        y = [group_name, cluster_genes]
        y2 = [group_name, cluster_marker_genes]
        all_genes_matrix.append (y)
        marker_genes_matrix.append (y2)
    
    print ('Marker genes matrix:')
    print (marker_genes_matrix)
    print ('All genes matrix:')
    print (all_genes_matrix)
    if verbose:
        print ('Count of marker genes is = ', cnt)
        print ("Len undefined:", len (undefined_list))
        print ("Len defined:", len (defined_list))
        print ("Len undefined unique:", len (set (undefined_list)))
        print ("Len defined unique:", len (set (defined_list)))
    
    if debug:
        s = set (undefined_list)
        print ('Undefined:')
        for i in s:
            print (i)
        print ('Defined:')
        for i in set (defined_list):
            print (i)
    

    totals = []
    
    for i in marker_genes_matrix:
        n = []
        for j in i[1]:
            if j[0] not in n:
                n.append (j[0])
        #for j in i[1]:
            #n.append (j[0])
        #n_s = set (n)
        #print (n_s)
        if debug:
            print (i, n)
            print (n)
        m = []
        #for j in n_s:
        for j in n:
            m.append ([j, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        totals.append ([i[0], m])
    
    if verbose:
        print ('Totals before counting:')
        for i in totals:
            print ('Cluster:', i[0])
            print (i)
            for j in i[1]:
                print (j)
    



    # do with sum/count !!!
    for cl_idx, i in enumerate (marker_genes_matrix):
        def get_totals_array (index, name):
            for data in totals [index][1]:
                if data[0] == name:
                    return data
        for j in i[1]:
            a = get_totals_array (cl_idx, j[0])
            a [1] = a [1] + 1.0
            a [2] = a [2] + j [2]


    if verbose:
        print ('Totals after counting:')
        for i in totals:
            print ('Cluster:', i[0])
            print (i)
            for j in i[1]:
                print (j)






    def attribute_split_groups (tots):
        if verbose:
            print ('Attribution of multiple cell group totals')
        new_totals = []
        for i in tots:
            print ('Cluster:', i[0])
            print (i[1])
            tots_for_grp = []
            for j in i[1]:
                n_in_grp = len (j[0])
                cnt = j [1]
                score = j [2]
                print (n_in_grp, cnt, score)
                print ('attributing', n_in_grp)
                cnt = cnt / n_in_grp
                score = score / n_in_grp
                seen = False
                for g in j [0]:
                   print ('attribute', g)
                   seen = False
                   for k in range (len (tots_for_grp)):
                       print ('Trying to find', k, tots_for_grp [k][0])
                       g_list = [g]
                       if g_list == tots_for_grp[k][0]:
                           print ('Adding', [g])
                           tots_for_grp [k][1] = tots_for_grp [k][1] + cnt
                           tots_for_grp [k][2] = tots_for_grp [k][2] + score
                           print (tots_for_grp)
                           seen = True
                           break
                   if not seen:
                       print ('Creating', [g])
                       new_j = copy.deepcopy (j)
                       new_j[0] = [g]
                       new_j[1] = cnt
                       new_j[2] = score
                       tots_for_grp.append (new_j)
                       print (tots_for_grp)
            new_tots_for_cluster = [i[0], tots_for_grp]
            new_totals.append (new_tots_for_cluster)
        return new_totals

    if attrib_split_groups:
        totals = attribute_split_groups (totals)

    if verbose:
        print ('Totals after attribution:')
        for i in totals:
            print ('Cluster:', i[0])
            print (i)
            for j in i[1]:
                print (j)



    
    def make_summed_totals (tots) :
    
        for i in range (len (tots)):
            cnt = 0
            score = [0.0, 0.0]
            for j in range (len (tots[i][1])):
                k = tots [i][1][j]
                cnt = cnt + k [1]
                score [0] = score [0] + k [2]
                if ( k [0] != ['UNDEFINED'] ):
                    score [1] = score [1] + k [2]
    
            for j in range (len (tots[i][1])):
                k = tots [i][1][j]
                k [3] = cnt - 1
                k [4] = score [1]
                k [5] = cnt
                k [6] = score [0]
                if k[0] != ['UNDEFINED']:
                    k [7] = k [1] / k [3]
                    k [8] = k [2] / k [4]
                else:
                    k [7] = 0.0
                    k [8] = 0.0
                k [9] = k [1] / k [5]
                k [10] = k [2] / k [6]
    
    
        def print_totals (totali, f = None):
            for i in totali:
                print ('Cluster:', i[0], file = f)
                for j in i[1]:
                    print (j, file = f)
    
    
        if verbose:
            print ('Before Sorting')
            print_totals (tots)
    
        sorted_tots = []
    
        for i in range (len (tots)):
            def takeCnt (elem):
                return elem [7]
            s = sorted (tots [i][1], key=takeCnt, reverse=True)
            a = [tots [i][0], s]
            sorted_tots.append (a)
    
        if verbose:
            print ('Sorted')
            print (sorted_tots)
            print_totals (sorted_tots)

        if save_file is not None:
            with open (save_file, 'w') as f:
                print_totals (sorted_tots, f = f)
                print ('-------------------------------------------', file=f)
                print (sorted_tots, file=f)
        return sorted_tots

    sorted_totals = make_summed_totals (totals)
    
    return sorted_totals, marker_genes_matrix
    

def plotHRGFreqTable (freq_table, cell_groups, plot_type, args, dn_mode, verbose = False, save_file = '.plot',
         plot_fmt = 'png', dpi = 100):

    if args.noplot:
        return
    print ('PLOT: plotHRGFreqTable:', len (freq_table))

 
    color_table = getCellGroupColors (cell_groups)
    print (color_table)

    nrows = (int)(len(freq_table) / 3)
    if ( len(freq_table) % 3 != 0 ):
        nrows += 1

    for (idx, y_title) in zip ([1, 2, 7, 8, -1, -2], ['Count', 'Score', 'FractionCount', 'FractionScore', 'MeanScore', 'FractionMeanScore']):
        print ([row[0] for row in freq_table[0][1] if row[0] != ['UNDEFINED']])
        print ([row[1] for row in freq_table[0][1] if row[0] != ['UNDEFINED']])

        fig, axs = plt.subplots(nrows, 3, sharey=True,
             figsize = (24,14), dpi=100)
        fig.tight_layout (rect=[0, 0.03, 1, 0.95])

        print (idx, '...', y_title)
        for (cluster_data, ax) in zip (freq_table, axs.ravel()):
            cell_groups =[row[0] for row in cluster_data[1] if row [0] != ['UNDEFINED']]
            y_pos = np.arange(len(cell_groups))
  
            if idx > 0:
                y_data =[row[idx] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
            else:
                y_cnt =[row[1] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
                y_score =[row[2] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
                y_mean_score = [s/c for s,c in zip (y_score, y_cnt)]
                total_mean_score = np.sum (y_mean_score)
                if idx == -1:
                    y_data = y_mean_score
                elif idx == -2:
                    total_mean_score = [total_mean_score]
                    total_mean_score *= len (y_mean_score)
                    y_data = [s/t for t,s in zip (total_mean_score, y_mean_score)]
  
            print (len (cell_groups))
            print ('y_pos:', y_pos)
            print ('y_data:', y_data)
            print (len (y_data))
            clr_list = []
            for i in cell_groups:
                clr_list.append (color_table [str (i)])
            for cn, cg in enumerate (cell_groups):
                print ('cn:', cn, 'color:', clr_list [cn], 'y_data:', y_data [cn])
                ax.bar(y_pos [cn], y_data [cn], align='center', color = clr_list [cn], label = cg)
            #ax.set_xticks(y_pos)
            ax.set_xticks([])
            #ax.set_xticklabels(cell_groups)
            ax.legend ()
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.grid (b=False, axis='x')
            ax.set_ylabel(y_title)
            ax.set_title('Cluster ' + str (cluster_data [0]))

        if not args.noplottitles:
            plt.suptitle ('Cell gene group frequency by ' + y_title + ' (' + plot_type + ')', fontsize = 14)
    
        if save_file is not None:
            fname = save_file + '.' + y_title + '.'  + plot_fmt
            save_plot (fname, args, f=fig)
        if args.show:
            plt.show()
        plt.close()
 

 
def correlationAnalysis (adata, args, dn_mode, mg_mtx, corr_method, pipeline_id = None):
        from scipy.stats.stats import pearsonr
        import itertools

        def get_marker_genes (cluster):
            markers = []
            data = cluster [1][:]
            for d in data:
                if d[1] is not None:
                    markers.append (d[1])
            return markers
        def get_expr_data (adata, gene):
            gene_mask = adata.var_names == gene
            print (gene_mask)
            n_cols = list (gene_mask).count (True)
            print (n_cols)
            if n_cols == 0:
                print ('Marker gene ', gene, 'is not in the matrix!')
                return None
            if n_cols > 1:
                print ('Marker gene ', gene, 'appears more than once in the matrix!')
                return None
            expr_data = adata.X[:, gene_mask]
            print (expr_data.sum ())
            print (len (expr_data))
            return expr_data
        def corr_analysis (data):

            corr, p_value = pearsonr (data [0], data [1])
            print ("Pearson = ( corr: ", corr, ') ( p_value: ', p_value, ')')

        def corr_plots (cluster, plot_data, corr_method, args, dn_mode):
            if prog_args.noplot:
                return
            f = plt.figure(figsize = (25,15))
            f.suptitle("Cluster " + cluster)
            n = len (plot_data)
            n_cols = 4
            if n < n_cols:
                n_rows = 1
                n_cols = n
            else:
                n_rows = (int)(n / n_cols)
                if n % n_cols != 0:
                    n_rows += 1
            for i, pd in enumerate (plot_data):
                ax = f.add_subplot(n_rows, n_cols, i + 1)
                ax.scatter(pd [1][0], pd [1][1])
                ax.set(xlabel=pd[0][0], ylabel=pd[0][1])

            sns.despine(offset=10, trim=False)
            plt.tight_layout()
            nm = '/correlation.'
            if pipeline_id is not None:
                nm = nm + pipeline_id + '.'
            fname = plots_dir + '/sample' + args.sample + nm + get_plot_label (args, denoise_id = dn_mode) + '.' + corr_method + '.' + 'cluster_' + cluster + '.' + plot_format
            save_plot (fname, args, f=f)
            if args.show:
                plt.show()
            plt.close(f)



        print ('Marker genes matrix')
        print (mg_mtx)
        for i in mg_mtx:
            print ('Cluster:', i [0])
            markers = get_marker_genes (i)
            print (markers)
            plot_data = []
            for L in range(0, len(markers)+1):
                for subset in itertools.combinations(markers, L):
                    if len (subset) == 2:
                        print('Calculating correlation for:', subset)
                        expr_data = [get_expr_data (adata, subset [0]),
                            get_expr_data (adata, subset [1])]
                        corr_analysis (expr_data)
                        plot_data.append ([subset, expr_data])
            if len (plot_data) > 0:
                corr_plots (i [0], plot_data, corr_method, args, dn_mode)


def latent_correlationAnalysis (adata, args, dn_mode):

    from scipy.stats.stats import pearsonr
    import itertools

    def plot_latent_heatmap (cfs, dn_mode):
        if args.noplot:
            return
        f = plt.figure(figsize = (25,15))
        ax = sns.heatmap(cfs, linewidth=0.5)
        f.suptitle("Latent correlation heatmap")
        nm = '/latent_correlation.'
        fname = plots_dir + '/sample' + args.sample + nm + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
        save_plot (fname, args, f=f)
        if args.show:
            plt.show()
        plt.close(f)

    def latent_correlations (adata, dn_mode):
        n_genes = adata.obsm['X_dca'].shape[1]
        idxs = []
        cfs = np.zeros ((n_genes, n_genes))
        ps = np.zeros ((n_genes, n_genes))
        # ? +1
        for i in range(0, n_genes):
            idxs.append (i)
            cfs [i, i] = 1.0
            ps [i, i] = 1.0
        for pair in itertools.combinations(idxs, 2):
            d1 = adata.obsm['X_dca'][:][pair [0]]
            d2 = adata.obsm['X_dca'][:][pair [1]]
            corr, p_value = pearsonr (d1, d2)
            print ("Pearson for latent = col: ", pair [0], 'col:', pair [1], '=', corr)
            for (i, j) in zip ([0,1],[1,0]):
                cfs [idxs[pair[i]], idxs [pair[j]]] = corr
                ps [idxs[pair[i]], idxs [pair[j]]] = p_value
        print ('Latent coefficients:')
        print (cfs)
        plot_latent_heatmap (cfs, dn_mode)


    if dn_mode != 'latent':
        return


    print ('X_dca shape:', adata.obsm['X_dca'].shape)
    n_genes = adata.obsm['X_dca'].shape[1]
    print ('latent n_genes=', n_genes)
    if n_genes <= 32:
        latent_correlations (adata, dn_mode)
    else:
        print ('Too many genes for latent correlation analysis')


def metricsHRGFreqTable (freq_table, clustering_method, plot_type, plot_type_id, args, dn_mode,
            pmetrics, verbose = False, save_file = None, metrics_id = None):
    print ('metricsHRGFreqTable:', clustering_method)

    for (idx, y_title) in zip ([7, 8, -2], ['FractionCount', 'FractionScore', 'FractionMeanScore']):
        print ([row[0] for row in freq_table[0][1] if row[0] != ['UNDEFINED']])
        print ([row[1] for row in freq_table[0][1] if row[0] != ['UNDEFINED']])

        print (idx, '...', y_title)
        total = 0.0
        for cluster_data in freq_table:
            print ('Cluster:', cluster_data [0])
            if idx > 0:
                data =[row[idx] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
            else:
                y_cnt =[row[1] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
                y_score =[row[2] for row in cluster_data[1] if row[0] != ['UNDEFINED']]
                y_mean_score = [s/c for s,c in zip (y_score, y_cnt)]
                total_mean_score = np.sum (y_mean_score)
                total_mean_score = [total_mean_score]
                total_mean_score *= len (y_mean_score)
                data = [s/t for t,s in zip (total_mean_score, y_mean_score)]
            if ( len (data) > 0 ):
                mx = np.max (data)
                total += mx
                print ('max:', mx, 'total=', total)
        print ('total=', total, 'count=', len(freq_table))
        metric = total / len(freq_table)

        if plot_type_id is None:
            pmetrics ['cluster_confidence']['imputed' if dn_mode == 'denoise' else 'latent'][clustering_method][y_title] = metric

        print ('METRIC for', y_title, '(', clustering_method, '-', plot_type, ')', ':', metric)
        if save_file is not None:
            info_txt = y_title
            info_txt = info_txt + ' (' + clustering_method + ')'
            info_txt = info_txt + ' (' + dn_mode + ')'
            if metrics_id is not None:
                info_txt = info_txt + ' (' + metrics_id + ')'
            with open (save_file, 'a') as f:
                print (plot_type, '-', info_txt, ':', metric, file = f)

 


def cellGroupBins (group_names, gene_names, scores, cell_groups, clustering_method, args, dn_mode, pmetrics,
               pipeline_id = None, plot_id = None):

    for (splt, id, attrib_type) in zip ([True, False], [None, 'noattrib'], ['With attribution', 'Without attribution']):
        save_file = results_dir + '/sample' + args.sample + '/cell_group_analysis'
        if id is not None:
            save_file = save_file + '.' + id
        if pipeline_id is not None:
            save_file = save_file + '.' + pipeline_id
        save_file = save_file + '.' + get_plot_label (args, denoise_id = dn_mode)
        save_file = save_file + '.' + clustering_method
        save_file = save_file + '.txt'

        metrics_file = get_metrics_filename (args, dn_mode)

        plot_file = plots_dir + '/sample' + prog_args.sample + '/'
        if plot_id is not None:
            plot_file = plot_file + plot_id + '.'
        plot_file = plot_file + 'cell_group_analysis'
        if id is not None:
            plot_file = plot_file + '.' + id
        plot_file = plot_file + '.' + get_plot_label (args, denoise_id = dn_mode)
        plot_file = plot_file + '.' + clustering_method

        freq_table, mg_matrix = calcHRGFreqTable (group_names, gene_names, scores, cell_groups,
                verbose = args.verbose, debug = args.debug, save_file =save_file, attrib_split_groups = splt)
        plotHRGFreqTable (freq_table, cell_groups, attrib_type, args, dn_mode, verbose = args.verbose, save_file = plot_file)
        metricsHRGFreqTable (freq_table, clustering_method, attrib_type, id, args, dn_mode, pmetrics,
                  verbose = args.verbose, save_file = metrics_file, metrics_id = pipeline_id)
    return mg_matrix

def cellGroupAnalysis (adata, cell_groups, clustering_method, args, dn_mode, pmetrics):

    group_names = (adata.uns['rank_genes_groups']['names'].dtype.names)
    gene_names = adata.uns['rank_genes_groups']['names']
    scores = adata.uns['rank_genes_groups']['scores']

    print ('Group names:')
    print (group_names)
    print (len (group_names))

    mg_matrix = cellGroupBins (group_names, gene_names, scores, cell_groups, clustering_method, args, dn_mode, pmetrics)
    correlationAnalysis (adata, args, dn_mode, mg_matrix, clustering_method)












"""# Helping functions"""

def readAnnotationMatrix(path, delimiter='\t', verbose=True):
    
    annotationMatrix = pd.read_csv(path, delimiter=delimiter)
    
    if verbose:
        print(" * Reading Annotation Matrix, path=%s\n"%path)
    
    return annotationMatrix

# This function reads the given marker gene list and searches these genes in our sample
"""# Expressed genes - xls"""

def readMarkerList(path, smartSeq, annotationMatrix, verbose=True):

    
    if verbose:
        print(" * Reading the given marker list")
    
    markerList = pd.read_excel(path)
    
    if verbose:
        print(" * Cleaning the marker list ...")
    
    markerClean = markerList[pd.isna(markerList).sum(axis=1) != len(markerList.columns)]

    print ('Here is markerClean ---------------------------------------')
    print (markerClean)
    
    if verbose:
        print(" * Searching expressed genes ...")
        
    expressedGenes = smartSeq.var.index.tolist()
    markerCleanEprs = []

    for gene in markerClean["Unnamed: 3"]:
                
        A = annotationMatrix.symbol[annotationMatrix["id"] == gene]
        if len(A.values) > 0:
            for i in range(0, len(A.values)):
                if A.values[i] in expressedGenes:
                    markerCleanEprs.append(A.values[i])
                    
        else:
            if gene in expressedGenes:
                markerCleanEprs.append(gene)
    
    indices = np.unique(markerCleanEprs, return_index=True)[1]
    markerCleanEprs = [markerCleanEprs[index] for index in sorted(indices)]
        
    if verbose:
        print(" * There are %d genes expressed in the dataset"%len(markerCleanEprs))
    
    print ('Here is markerCleanEprs ---------------------------------------')
    print (markerCleanEprs)

    return markerCleanEprs

# This function allows for finding the list of Mitochondrial Genes according to the given Annotation Matrix
# This list is used to calculate the percentage of mitochondrial genes in the samples.

def findMitochondrialGenes(annotationMatrix, verbose=True):
    
    genes = annotationMatrix.id[annotationMatrix.symbol.str.match('MT-')]
    
    if verbose:
        for mt in genes:
            print(mt, end=" ")
        print("\n")

        for mt in annotationMatrix.symbol[annotationMatrix.symbol.str.match('MT-')]:
            print(mt, end=" ")
        print("\n")
        
        print("\n")
    
    return genes

# Basic function to load the gene expression matrix as well as calculate both the the mitochondrial percentage
# and the ERCC content (ERCCs are control genes)

def loadSmartSeq(path, delimiter='\t', mit=True, ercc=True, verbose=True, min_genes=10, min_cells=10,
                 annotationMatrix=None, title=""):
        
    if verbose:
        print(" * Loading sample, path=%s"%path)
    
    rawData = pd.read_csv(path, delimiter=delimiter)
    rawData = rawData.iloc[0:-4:,:]
    indeces = rawData.iloc[:,0]
    names   = {}
    for i,idx in enumerate(indeces):
        names[i] = idx

    cols = {}
    for i,col in enumerate(rawData.columns):
        cols[col] = 'X' + col

    rawData.rename(index=names, inplace=True)
    rawData       = rawData.drop(columns=['Unnamed: 0'])
    rawData.rename(columns=cols, inplace=True)
    countsRawData = rawData[rawData.index.str.match('ENSG0')]

    smartSeq = sc.AnnData(countsRawData.T)

    # These are needed to set the 'n_genes' observation
    # Are they really needed for anything else at this stage?
    sc.pp.filter_cells(smartSeq, min_genes=min_genes)
    sc.pp.filter_genes(smartSeq, min_cells=min_cells)
    
    smartSeq.obs['study'] = title
    smartSeq.obs['study'] = smartSeq.obs['study'].astype('category')
    
            
    if mit:
        try:
            mitGenes = findMitochondrialGenes(annotationMatrix, verbose=False)
            smartSeq.obs['percent_mito'] = np.sum(rawData.loc[mitGenes].T, axis=1) / np.sum(rawData.T, axis=1)
            if verbose:
                print(" * Calculating the mitochondrial percentage")
        except:
            print(" * Warning: the Annotation Matrix must be provided to calculate the mitochondrial percentage!")
        
    if ercc:
        erccGenes = rawData.index[rawData.index.str.match('ERCC-')]
        smartSeq.obs['ercc_content'] = np.sum(rawData.loc[erccGenes].T, axis=1) / np.sum(rawData.T, axis=1)
        if verbose:
            print(" * Calculating the ercc content")

    smartSeq.obs['n_counts'] = smartSeq.X.sum(axis=1)
    
    if verbose:
        print(" * Initial SmartSeq2 Object: %d genes across %d single cells"%(smartSeq.n_vars, smartSeq.n_obs))
        print("\n")
    
    return smartSeq

# This function loads the metadata containing several information (e.g., the origin, gate, ...)

def loadMetadata(path, smartSeq, delimiter=",", verbose=True, numPrint=10):
        
    if verbose:
        print(" * Loading metadata, path=%s"%path)
        
    metadata = pd.read_csv(path, delimiter=delimiter) 
    metadata.rename(index=metadata.CELL_NAME, inplace=True)
    metadata = metadata.drop(columns=['Unnamed: 0'])
    
    smartSeqLoc = smartSeq.copy()
    smartSeqLoc.obs = pd.concat([smartSeq.obs, metadata.loc[smartSeq.obs.index.tolist()]], axis=1)
    smartSeqLoc.obs.sort_index(axis=1, inplace=True)
    
    if verbose:
        display(smartSeqLoc.obs.head(numPrint))
    
    return smartSeqLoc

# This function assigns the origin of the cells

def assignOrigin(smartSeq, namesIn=[], namesOut=[], verbose=True):
    
    smartSeqLoc = smartSeq.copy()
    smartSeqLoc.obs['origin'] = smartSeqLoc.obs.COMMON_NAME
        
    for idx,name in enumerate(namesIn):
        names = smartSeqLoc.obs.origin[smartSeqLoc.obs.origin.str.contains(name)]
        smartSeqLoc.obs['origin'].replace(names, namesOut[idx], inplace=True)
    smartSeqLoc.obs['origin'] = smartSeqLoc.obs.origin.astype('category')
    
    if verbose:
        print(smartSeqLoc.obs.origin.cat.categories.tolist())
    
    return smartSeqLoc

# This function assigns the gate of the cells

def assignGate(smartSeq, namesIn=[], namesOut=[], verbose=True):
    
    smartSeqLoc = smartSeq.copy()
    smartSeqLoc.obs['gate'] = smartSeqLoc.obs.COMMON_NAME
        
    for idx,name in enumerate(namesIn):
        names = smartSeqLoc.obs.gate[smartSeqLoc.obs.gate.str.contains(name)]
        smartSeqLoc.obs['gate'].replace(names, namesOut[idx], inplace=True)
    smartSeqLoc.obs['gate'] = smartSeqLoc.obs.gate.astype('category')
    
    if verbose:
        print(smartSeqLoc.obs.gate.cat.categories.tolist())
    
    return smartSeqLoc

# This function assigns the type of the cells

def assignCellType(smartSeq, namesIn=[], namesOut=[], verbose=True):
    
    smartSeqLoc = smartSeq.copy()
    smartSeqLoc.obs['cell_type'] = smartSeqLoc.obs.COMMON_NAME
        
    for idx,name in enumerate(namesIn):
        names = smartSeqLoc.obs.cell_type[smartSeqLoc.obs.cell_type.str.contains(name)]
        smartSeqLoc.obs['cell_type'].replace(names, namesOut[idx], inplace=True)
    smartSeqLoc.obs['cell_type'] = smartSeqLoc.obs.cell_type.astype('category')
    
    if verbose:
        print(smartSeqLoc.obs.cell_type.cat.categories.tolist())
    
    return smartSeqLoc

# This function is used to show the violin plots for the Quality Control step

def plotViolinQuality(smartSeq, dn_mode, listVariables=[], pointSize=4, height=8, id = None):

    if prog_args.noplot:
        return

    if prog_args.nofoetal:
        if len (smartSeq.obs.dtypes.index) > 0:
            #listVariables = smartSeq.obs.dtypes.index [:4]
            listVariables = smartSeq.obs.dtypes.index [:2]
        else:
            return

    plot_id = 'violin_quality'
    if id is not None:
        plot_id = plot_id + '_' + id
    print ('PLOT:', plot_id)
    
    cols = len(listVariables)
    width  = height*cols

    print ('listVariables:', listVariables)
    print ('cols:', cols)

    f, axs = plt.subplots(1,cols,figsize=(width,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    examine_adata (smartSeq, "BEFORE violin_plot")
    
    for idx,var in enumerate(listVariables):
        print ('Plotting:', var)
        sc.pl.violin(smartSeq, var, jitter=0.4, size=pointSize,
                     #multi_panel=False, ax=axs[idx], show=False)
                     multi_panel=False, ax=axs[idx], show=False)
    
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/' + plot_id + '.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function is used to show the scatter plots for the Quality Control step

def plotScatterPlotQuality(smartSeq, dn_mode, colors=[], pointSize=4, height=8, cat='type'):

    if prog_args.noplot:
        return

    print ('PLOT: quality')
    
    cols = 3
    width  = height*cols

    f, axs = plt.subplots(1,cols,figsize=(width,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    sc.pl.scatter(smartSeq, x="n_genes", y="percent_mito", color=cat,
                  palette=colors, ax=axs[0], size=pointSize, show=False)

    sc.pl.scatter(smartSeq, x="n_genes", y="ercc_content", color=cat,
                  palette=colors, ax=axs[1], size=pointSize, show=False)

    sc.pl.scatter(smartSeq, x="percent_mito", y="ercc_content", color=cat,
                  palette=colors, ax=axs[2], size=pointSize, show=False)
    
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/quality.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function shows the histrograms related to the number of genes, percentage of mitochondrial, and
# ERCC content. It is useful to the thresholds used to filter out the unwanted cells

def plotHistogramQuality(smartSeq, dn_mode, thresNumGen=150, thresPercMit=0.25, thresErccCont=0.6, bins=50):
    if prog_args.noplot:
        return

    cols   = 3

    if prog_args.nofoetal:
        if np.sum (smartSeq.obs.dtypes.index == 'n_genes') > 0:
            cols = 1
        else:
            return

    print ('PLOT: histogram')

    height = 6

    f, axs = plt.subplots(1,cols,figsize=(height*cols,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    sns.distplot(smartSeq.obs["n_genes"],
                 bins=bins,
                 color='black',
                 hist=True,
                 kde=False,
                 #ax=axs[0])
                 ax=axs[0] if cols > 1 else axs)

    ax=axs[0] if cols > 1 else axs
    ax.axvline(thresNumGen, color='red')


    if cols > 1:
        sns.distplot(smartSeq.obs["percent_mito"],
                 bins=bins,
                 color='black',
                 hist=True,
                 kde=False,
                 ax=axs[1])

        axs[1].axvline(thresPercMit, color='red')
    
    
        sns.distplot(smartSeq.obs["ercc_content"],
                 bins=bins,
                 color='black',
                 hist=True,
                 kde=False,
                 ax=axs[2])

        axs[2].axvline(thresErccCont, color='red')

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/histogram.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function is used to show the scatter plots of the type, origin, and gate of the cells

def plotScatterPlotDifferentGroup(smartSeq, dn_mode, pointSize=150, height=8, palette=sns.color_palette("deep")):

    if prog_args.noplot:
        return

    if prog_args.nofoetal:
        return

    print ('PLOT: scatter')

    cols = 2
    width  = height*cols

    f, axs = plt.subplots(1,cols,figsize=(width,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    sc.pl.scatter(smartSeq, x="n_genes", y="percent_mito", color="origin",
                  palette=palette, ax=axs[0], size=pointSize, show=False)

    sc.pl.scatter(smartSeq, x="n_genes", y="percent_mito", color="gate",
                  palette=palette, ax=axs[1], size=pointSize, show=False)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/scatter.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function maps the Ensembl notation to the common name of the genes, based on the given Annotation Matrix

def mappingEnsemblToAnnotated(smartSeq, annotationMatrix, verbose=True):
    
    if verbose:
        print(" * Applying the mapping based on the provided Annotation Matrix ...")
    
    oldIndices = smartSeq.var.index.tolist()
    names = {}
    for index in oldIndices:
        A = annotationMatrix.symbol[annotationMatrix["id"] == index]
        if len(A.values) > 0:
            for i in range(0, len(A.values)):
                names[index] = A.values[i]
        else:
            names[index] = index
            
    smartSeq.var.rename(index=names, inplace=True)
    
    if verbose:
        print(" * Gene names renamed from ensembl annotated gene name")
        smartSeq.var.head(5)

# This function calculates the scores and assigns a cell cycle phase (G1, S or G2M)

def CellCycleScoring(path, smartSeq, annotationMatrix, verbose=True):
    
    cell_cycle_genes = [x.strip() for x in open(path)]

    s_genes   = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    
    
    if verbose:
        print(" * Searching s genes")

    s_genesIDs = []
    for gene in s_genes:
        A = annotationMatrix.id[annotationMatrix["symbol"] == gene]
        if len(A.values) > 0:
            for i in range(0, len(A.values)):
                s_genesIDs.append(A.values[i])
    
    if verbose:
        print(" * Searching g2m genes")

    g2m_genesIDs = []
    for gene in g2m_genes:
        A = annotationMatrix.id[annotationMatrix["symbol"] == gene]
        if len(A.values) > 0:
            for i in range(0, len(A.values)):
                g2m_genesIDs.append(A.values[i])  

    sc.tl.score_genes_cell_cycle(smartSeq, s_genes=s_genesIDs, g2m_genes=g2m_genesIDs)
    
    
    if verbose:
        print(" * Cell cycle phase calculated")

# Function showing the PCA space

def plotPCA(smartSeq, dn_mode, listVariables=[], pointSize=150, width=8, height=8, cols=2, palette=sns.color_palette("deep"), plot_id = None):

    if prog_args.noplot:
        return

    print ('PLOT: pca')
    
    if len(listVariables) > 1:
        rows = int(len(listVariables)/cols)

        if rows*cols < len(listVariables):
            rows += 1
            
    else:
        rows = 1
        cols = 1
    
    f, axs = plt.subplots(rows,cols,figsize=(width*cols,height*rows))
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    idx = 0
    for r in range(0, rows):
        for c in range(0, cols):
            
            if idx > len(listVariables):
                break
            
            var = listVariables[idx]
            
            if cols == 1 and rows == 1:
                if var == '':
                    sc.pl.pca(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                else:
                    sc.pl.pca(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                    
            elif cols == 1 or rows == 1:
                
                if var == '':
                    sc.pl.pca(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
                else:
                    sc.pl.pca(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
            else:
                    if var == '':
                        sc.pl.pca(smartSeq,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
                    else:
                        sc.pl.pca(smartSeq,
                                  color=var,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
            idx += 1
            
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/pca.' + get_plot_label (prog_args, denoise_id = dn_mode) + (plot_id if plot_id is not None else '') + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function plots the Highly Variable Genes

def plotHVGs(smartSeq, dn_mode, width=8, height=8, plot_id=None):

    if prog_args.noplot:
        return
    print ('PLOT: filter_genes_dispersion')
    
    #f, axs = plt.subplots()
    
    sns.set(rc={'figure.figsize':(width,height)})
    sns.set(font_scale=1.5)
    sns.set_style("white")

    fname_add = '.' + plot_id + '.' + prog_mnemonic + '.' + plot_format

    sc.pl.highly_variable_genes(smartSeq,
        show=prog_args.show,
        save = None if prog_args.nosave else fname_add)
    add_to_slideshow ('filter_genes_dispersion' + fname_add, prog_args)
    

# This function plots variance ratio of the PCA components

def PCA_ElbowPlot(smartSeq, dn_mode, n_pcs=50, log=True, width=16, height=8):

    if prog_args.noplot:
        return

    print ('PLOT: pca_variance')
    print (smartSeq)
    print (smartSeq.X)

    try:
        sns.set(rc={'figure.figsize':(width,height)})
        sns.set(font_scale=1.5)
        sns.set_style("white")

        fname_add = '.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + prog_mnemonic + '.' + plot_format

        sc.pl.pca_variance_ratio(smartSeq, n_pcs=n_pcs, log=log,
            show=prog_args.show,
            save = None if prog_args.nosave else fname_add)

        sns.despine(offset=10, trim=False)
        add_to_slideshow ('pca_variance_ratio' + fname_add, prog_args)
    except:
        # This is here because of the numerical stability bug in scvi
        # which I appeared to solve in my_scvi by adding eps to lgamma
        e = sys.exc_info ()[2]
        t = traceback.format_exc ()
        print ('FOETAL EXCEPTION:', e)
        print (t)

# This function performs a pre analysis, namely:
#    * Normalisation of the data
#    * Highly Variable Genes
#    * Regression out unwanted sources of variation
#    * Scaling the data
#    * PCA
    

def preAnalysis(smartSeq, dn_mode, args, min_mean=0.1, max_mean=10, min_disp=0.25, listRegress=[]):

    print(" * Normalising the data and log transformation")
    examine_adata (smartSeq, "BEFORE NORMALIZE")
    sc.pp.normalize_per_cell(smartSeq, counts_per_cell_after=1e4)
    #sc.pp.normalize_total(smartSeq)
    examine_adata (smartSeq, "BEFORE LOG1P")
    sc.pp.log1p(smartSeq)
    examine_adata (smartSeq, "AFTER LOG1P")

    #save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'Pre_AfterLog', denoise_id = dn_mode)

    print(" * Highly Variable Genes")
    
    min_mean, max_mean, min_disp = get_hvg_constants ()
    sc.pp.highly_variable_genes(smartSeq, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
    examine_adata (smartSeq, "AFTER highly_variable_genes")

    print(" * Highly Variable Genes completed")

    save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'Pre_AfterHVGs', denoise_id = dn_mode)

    plotHVGs(smartSeq, dn_mode, plot_id=(get_plot_label (prog_args, denoise_id = dn_mode) + '.' + 'preAnalysis'))
        
    print(" \t* Number of control HVGs: %d\n"%(smartSeq[:, smartSeq.var['highly_variable']].n_vars))
    print(smartSeq[:, smartSeq.var['highly_variable']])

    if prog_args.regress:
        if len(listRegress):
            print(" * Regression out on", listRegress, "to remove unwanted sources of variation")
            sc.pp.regress_out(smartSeq, listRegress)
            examine_adata (smartSeq, "AFTER regress_out")
            save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'Pre_AfterRegress', denoise_id = dn_mode)
    
    print(" * Scaling the data")
    sc.pp.scale(smartSeq, max_value=10)
    examine_adata (smartSeq, "AFTER scale")
    save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'Pre_AfterScale', denoise_id = dn_mode)

    
    print(" * Performing PCA on the detected HVGs")
    sc.settings.verbosity = 3
    sc.tl.pca(smartSeq, svd_solver='arpack', use_highly_variable=True)
    #sc.tl.pca(smartSeq, svd_solver='arpack')
    sc.settings.verbosity = 0
    
    examine_adata (smartSeq, "BEFORE PCA_ElbowPlot")
    PCA_ElbowPlot(smartSeq, dn_mode, width=12, height=9)
    #plotPCA(smartSeq, dn_mode, listVariables=['study'], plot_id= '.preAnalysis')
    plotPCA(smartSeq, dn_mode, listVariables=['n_counts'], plot_id= '.preAnalysis')

    save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'AfterPreAnalysis', denoise_id = dn_mode)


    
# Function showing the tSNE space

def plotTSNE(smartSeq, dn_mode, listVariables=[], pointSize=150, width=8, height=8, cols=2, palette=cm.plasma, id=None, denoise_id = None):

    if prog_args.noplot:
        return

    plot_id = 'tsne'
    if id is not None:
        plot_id = plot_id + ' (' + id + ')'
    print ('PLOT:', plot_id)

    if len(listVariables) > 1:
        rows = int(len(listVariables)/cols)

        if rows*cols < len(listVariables):
            rows += 1
            
    else:
        rows = 1
        cols = 1
    
    f, axs = plt.subplots(rows,cols,figsize=(width*cols,height*rows))
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    idx = 0
    for r in range(0, rows):
        for c in range(0, cols):
            
            if idx > len(listVariables):
                break
            
            var = None if len (listVariables) == 0 else listVariables[idx]
            
            if cols == 1 and rows == 1:
                if var == '':
                    sc.pl.tsne(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                else:
                    sc.pl.tsne(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                    
            elif cols == 1 or rows == 1:
                
                if var == '':
                    sc.pl.tsne(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
                else:
                    sc.pl.tsne(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
            else:
                    if var == '':
                        sc.pl.tsne(smartSeq,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
                    else:
                        sc.pl.tsne(smartSeq,
                                  color=var,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
            idx += 1
            
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/tsne.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)



# Function showing the UMAP space

def plotUMAP(smartSeq, dn_mode, listVariables=[], pointSize=150, width=8, height=8, cols=2, palette=sns.color_palette("deep"), id = None, denoise_id = None):
    if prog_args.noplot:
        return

    plot_id = 'umap'
    if id is not None:
        plot_id = plot_id + ' (' + id + ')'
    print ('PLOT:', plot_id)

    if len(listVariables) > 1:
        rows = int(len(listVariables)/cols)

        if rows*cols < len(listVariables):
            rows += 1
            
    else:
        rows = 1
        cols = 1
    
    f, axs = plt.subplots(rows,cols,figsize=(width*cols,height*rows))
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    idx = 0
    for r in range(0, rows):
        for c in range(0, cols):
            
            if idx > len(listVariables):
                break
            
            var = None if len (listVariables) == 0 else listVariables[idx]
            
            if cols == 1 and rows == 1:
                if var == '':
                    sc.pl.umap(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                else:
                    sc.pl.umap(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs,
                              show=False)
                    
            elif cols == 1 or rows == 1:
                
                if var == '':
                    sc.pl.umap(smartSeq,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
                else:
                    sc.pl.umap(smartSeq,
                              color=var,
                              size=pointSize,
                              palette=palette,
                              ax=axs[idx],
                              show=False)
            else:
                    if var == '':
                        sc.pl.umap(smartSeq,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
                    else:
                        sc.pl.umap(smartSeq,
                                  color=var,
                                  size=pointSize,
                                  palette=palette,
                                  ax=axs[r,c],
                                  show=False)
            idx += 1
            
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/umap.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)


"""# Integrated Analysis using tSNE and UMAP"""
# This function calculates the tSNE and UMAP spaces.
# As a first step, the neighborhood graph of observations is calculated.

def integratedAnalysis(smartSeq, dn_mode, n_pcs=50, metric='correlation'):
    sc.settings.verbosity = 3

    save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'BeforeIntegratedAnalysis')
    
    print(" * Computing the neighborhood graph of observations using %s"%metric)

    if dn_mode == 'denoise':
        use_rep = None
        if prog_args.nofoetal:
            if len (smartSeq.obs.dtypes.index) > 0:
                #listVars = smartSeq.obs.dtypes.index [:4]
                listVars = smartSeq.obs.dtypes.index [:2]
            else:
                listVars = []
        else:
            listVars = ['n_genes', 'n_counts', 'percent_mito', 'ercc_content']
 
    elif dn_mode == 'latent':
        use_rep = 'X_dca'
        listVars = []


    sc.pp.neighbors(smartSeq,
                n_pcs=n_pcs,
                use_rep=use_rep,
                random_state=10,
                metric = metric,
                method='umap')


    print("\n * Computing tSNE for", dn_mode)
    sc.tl.tsne(smartSeq, use_rep = use_rep, random_state=10, n_pcs=n_pcs)
    
    print("\n * Computing UMAP for", dn_mode)
    sc.tl.umap(smartSeq, random_state=10, n_components=3, min_dist=0.3)
    
    plotTSNE(smartSeq, dn_mode, listVariables = listVars, id = 'integratedAnalysis', denoise_id = dn_mode)
    plotUMAP(smartSeq, dn_mode, listVariables = listVars, id = 'integratedAnalysis', denoise_id = dn_mode)

# from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def plotSilhouette(X, s_values, silhouette_avg, dn_mode, labels, method, palette = cm.plasma, id=None, denoise_id = None):

    if prog_args.noplot:
        return

    plot_id = 'silhouette'
    if id is not None:
        plot_id = plot_id + ' (' + id + ')'
    print ('PLOT:', plot_id)

    n_clusters = len (set(labels))
    print ('LABELS:', labels, n_clusters)
    print (set(labels))


    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(18, 7)

    #ax1.set_xlim([-1, 1])
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            s_values[labels == str(i)]
        print ('ith:', ith_cluster_silhouette_values)

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    '''
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    '''

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for " + method + " clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')


    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/silhouette.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + ((id + '.') if id is not None else '') + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)

# This function clusters the neighborhood graph using either the Louvain or Leiden algorithm

def clustering(smartSeq, args, dn_mode, pmetrics,
              method='bad_value', spaces=["tsne"], resolution=0.8, palette=sns.color_palette("deep"), when_id=None):
    
    if method not in ['louvain', 'leiden']:
        print(" * Supported clustering algorithms: louvain, leiden")
        return
    
    print("\n * Performing the clustering (%s algorithm)"%method)
    
    sc.settings.verbosity = 3
    if method == 'louvain':
        sc.tl.louvain(smartSeq, resolution=resolution)
    if method == 'leiden':
        examine_adata (smartSeq, "Before leiden")
        # for debugging
        write_text_matrix (smartSeq, args,
                results_dir + '/sample' + args.sample + '/sample' + args.sample + '.BeforeLeiden.tsv')
        sc.tl.leiden(smartSeq, resolution=resolution)
   
    sc.settings.verbosity = 0

    if dn_mode == 'latent':
        id = 'latent_clusters.' + method
    else:
        id = 'clustering.' + method
    if when_id is not None:
        id = id + ('.' + when_id)

    X = smartSeq.X if dn_mode == 'denoise' else smartSeq.obsm ['X_dca']
    labels = smartSeq.obs [method]
    s_m = None
    s_ch = None
    s_db = None
    if 'silhouette' in parse_clustering_quality (args.clustering_quality):
        s_m = metrics.silhouette_score (X, labels, metric = 'euclidean')
        s_m_samples = metrics.silhouette_samples (X, labels, metric = 'euclidean')
        plotSilhouette(X, s_m_samples, s_m, dn_mode, labels, method, palette = palette, id=id)
        print ('SILHOUETTE')
        print (s_m_samples)
    if 'calinski_harabaz' in parse_clustering_quality (args.clustering_quality):
        s_ch = metrics.calinski_harabaz_score (X, labels)
    if 'davies_bouldin' in parse_clustering_quality (args.clustering_quality):
        s_db = metrics.davies_bouldin_score (X, labels)
    with open (get_metrics_filename (args, dn_mode), 'a') as f:
        print (method, '-', id, ':', s_m, file = f)
    pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent'][method]['silhouette'] = float (s_m) if s_m is not None else None
    pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent'][method]['calinski_harabaz'] = float (s_ch) if s_ch is not None else None
    pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent'][method]['davies_bouldin'] = float (s_db) if s_db is not None else None
    
    for space in spaces:
        if space == 'tsne':
            plotTSNE(smartSeq, dn_mode, listVariables=[method], palette=palette, id = id)
        elif space == 'umap':
            plotUMAP(smartSeq, dn_mode, listVariables=[method], palette=palette, id = id)
        else:
            print(" * Warning, the allowed spaces are: tsne and umap")

    
    keys = smartSeq.obs[method].unique().tolist()
    keys = sorted(map(int, keys))

    for key in keys:

        cells = smartSeq[smartSeq.obs[method] == str(key)]
        print(" \t* %4d cells in cluster %2d"%(cells.n_obs, key))


    if args.denoise_mode == 'full' and dn_mode == 'latent':
        id = 'clustering.' + method
        if when_id is not None:
            id = id + ('.' + when_id)
        print("\n * Computing tSNE using latent space clustering with denoised data for", dn_mode)
        sc.tl.tsne(smartSeq, random_state=10, n_pcs=9)
        for space in spaces:
            if space == 'tsne':
                plotTSNE(smartSeq, dn_mode, listVariables=[method], palette=palette, id = id)
            elif space == 'umap':
                plotUMAP(smartSeq, dn_mode, listVariables=[method], palette=palette, id = id)
            else:
                print(" * Warning, the allowed spaces are: tsne and umap")


def plotRGGBarChart(smartSeq, dn_mode, cell_groups, clustering = 'bad_value'):
    if prog_args.noplot:
        return

    print ('PLOT: rgg_markers')

    color_table = getCellGroupColors (cell_groups)

    group_names = (smartSeq.uns['rank_genes_groups']['names'].dtype.names)

    nrows = (int)(len(group_names) / 3)
    if ( len(group_names) % 3 != 0 ):
        nrows += 1

    fig, axs = plt.subplots(nrows, 3, sharey=True,
             figsize = (28,16), dpi=100)
    fig.tight_layout (rect=[0, 0.03, 1, 0.95])

    for (grp, ax) in zip (group_names, axs.ravel()):
        y_pos = np.arange(30)
        bar_titles = []
        unique_colors = {}
        for i, (g, s) in enumerate (zip (smartSeq.uns['rank_genes_groups']['names'][grp],
                       smartSeq.uns['rank_genes_groups']['scores'][grp])):
            if i >= 30:
                break
            clr = 'xkcd:black'
            lbl = None
            if g in cell_groups:
                lbl = str (cell_groups [g])
                clr = color_table [lbl]
                if lbl in unique_colors:
                    lbl = None
                else:
                    unique_colors [lbl] = clr
            ax.bar(y_pos [i], s, align='center', color = clr, label = lbl)
            bar_titles.append (g)
        ax.set_xticklabels(bar_titles, rotation='vertical')
        ax.set_xticks(y_pos)
        ax.legend ()
        #ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.grid (b=False, axis='x')
        ax.set_ylabel('Score')
        ax.set_title('Group ' + str (grp))

    
    fname = plots_dir + '/sample' + prog_args.sample + '/rgg_markers.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + clustering + '.' + plot_format
    save_plot (fname, prog_args, f=fig)
    if prog_args.show:
        plt.show ()
    plt.close()


# This function shows the founded marker genes

def plotGenesRankGroups(smartSeq, dn_mode, cell_groups, clustering = 'bad_value', n_genes=20, width=6, height=12, ncols=3):

    group_names = (smartSeq.uns['rank_genes_groups']['names'].dtype.names)
    for i in group_names:
        print ('ranked_grp:', i, smartSeq.uns['rank_genes_groups']['names'][i])
        print ('scores:', smartSeq.uns['rank_genes_groups']['scores'][i])

    if prog_args.noplot:
        return

    plotRGGBarChart (smartSeq, dn_mode, cell_groups, clustering)

    plot_id = 'rank_genes_groups_'
    grpby = smartSeq.uns ['rank_genes_groups']['params']['groupby']
    plot_id = plot_id + grpby
    print ('PLOT:', plot_id)

    
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    fname_add = '.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + clustering + '.' + prog_mnemonic + '.' + plot_format

    sc.pl.rank_genes_groups(smartSeq,
                            use_raw = False,
                            n_genes=n_genes,
                            ncols=ncols,
                            show=prog_args.show,
                            save = None if prog_args.nosave else fname_add)

    sns.despine(trim=False)
    add_to_slideshow (plot_id + fname_add, prog_args)



# This function finds the marker genes for each cluster using the given method (e.g., Wilcoxon with Bonferroni correction)

def calculateMarkerGenes(smartSeq, dn_mode, cell_groups, clustering='bad_value', method='wilcoxon', corr_method='bonferroni'):


    print("\n * Calculating the marker genes for each cluster")
    if clustering not in ['louvain', 'leiden']:
        print("clustering algorithm is not one of : louvain, leiden")
        return
    
    examine_adata (smartSeq, "Before rank_genes_groups")
    sc.tl.rank_genes_groups(smartSeq,
                            groupby=clustering,
                            method=method,
                            corr_method=corr_method)
    examine_adata (smartSeq, "After rank_genes_groups")

    save_h5ad (smartSeq, prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'AfterRankGenesGroups')

    plotGenesRankGroups(smartSeq, dn_mode, cell_groups, clustering, n_genes=30)

    if prog_args.noplot:
        return


    print ('PLOT:', 'matrixplot')
    examine_adata (smartSeq, "matrixplot")
    fname_add = '.' + clustering + '.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + clustering + '.' + prog_mnemonic + '.' + plot_format

    sc.pl.rank_genes_groups_matrixplot(smartSeq,
                   n_genes=20,
                   groupby=clustering,
                   use_raw=False,
                   swap_axes=True,
                   figsize=(30,50),
                   dendrogram=False,
                   show=prog_args.show,
                   save = None if prog_args.nosave else fname_add)
    add_to_slideshow ('matrixplot' + fname_add, prog_args)

# This function, given a a gene, is used to show:
#    * the violin plot of the gene expression per cluster
#    * the gene expression in the tSNE/UMAP space
#    * the clusters in the tSNE/UMAP space

def plotViolinAndSpace(smartSeq, gene, dn_mode, space='umap', cluster='bad_value', pointSize=150, width=8, height=8, palette=sns.color_palette("deep")):

    if prog_args.noplot or prog_args.nomgviolinandspace:
        return

    if cluster not in ['louvain', 'leiden']:
        print("clustering algorithm is not one of : louvain, leiden")
        return

    print ('Doing plotViolinAndSpace for gene:' + gene, ' clustering:',  cluster)

    cols = 2
    f, axs = plt.subplots(1,cols,figsize=(width*cols,height))
    
    sns.set(font_scale=1.5)
    sns.set_style("white")
    
    sc.pl.violin(smartSeq,
                 gene,
                 groupby=cluster,
                 use_raw=False,
                 palette=palette,
                 ax=axs[0],
                 show=False)
    
    if space == 'umap':

        sc.pl.umap(smartSeq,
                   color=gene,
                   use_raw=False,
                   cmap='RdGy_r',
                   size = pointSize,
                   ax=axs[1],
                   show=False)

    elif space == 'tsne':
        
        sc.pl.tsne(smartSeq,
                   color=gene,
                   use_raw=False,
                   cmap='RdGy_r',
                   size = pointSize,
                   ax=axs[1],
                   show=False)
    else:
        print(" * Unknown space. The possible values for space are: umap, tsne")
        
    
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/violin_space.' + cluster + '.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + gene + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    if prog_args.show:
        plt.show ()
    plt.close(f)



def plotHighestExprGenes(smartSeq, dn_mode = 'denoise'):
    if prog_args.noplot:
        return

    n_top = 20
    height = (n_top * 0.2) + 1.5
    width = 5
    f, axs = plt.subplots(figsize=(width,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    print ('PLOT: highest_expr_genes')

    sc.pl.highest_expr_genes(smartSeq, n_top=n_top,
        ax = axs,
        show = prog_args.show,
        save = None)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    fname = plots_dir + '/sample' + prog_args.sample + '/highest_expr_genes.' + get_plot_label (prog_args, denoise_id = dn_mode) + '.' + plot_format
    save_plot (fname, prog_args, f=f)
    plt.close(f)





def getGeneCellGroups (args, verbose = False):

    jsonfile_cell_groups = support_dir + '/processed/' + 'cell_groups.json'

    if args is not None and args.intermediate_load:
        d = json.load (open(jsonfile_cell_groups))
        print ('LOADED CELL GROUP DICT')
        print (d)
        print (len (d))
        return d

    markerList = pd.read_excel(markerListPath)

    d={}
    curGrp=None
    for idx, (c1, c2, c3, c4, c5, c1_null, c2_null, c3_null) in enumerate (zip (markerList [markerList.columns[0]],
                markerList[markerList.columns[1]],
                markerList [markerList.columns [2]],
                markerList[markerList.columns[3]],
                markerList [markerList.columns [4]],
                markerList [markerList.columns [0]].isnull (),
                markerList [markerList.columns [1]].isnull (),
                markerList [markerList.columns [2]].isnull ())):
        if verbose:
            print (c1, '...', c2, '...', c3, '...', c1_null, '...', c2_null, '...', c3_null)
        if 0 == idx:
            if not c1_null and not c2_null:
                curGrp = markerList.columns [0]
        if not c1_null and c2_null:
            curGrp = c1
            continue
        #print (curGrp, '....', c1, ':', c2, ':', c3, ':', c4, ':', c5, ':', c1_null)
        if c2_null and (c3_null or c3 == '-'):
            continue
        for (gene, nan) in zip ([c2, c3], [c2_null, c3_null]):
            if nan or gene == '-':
                continue
            if not gene in d:
                print ('Inserting', gene, curGrp)
                d [gene] = [curGrp]
            else:
                if verbose:
                    print ('Checking whether to append', gene, curGrp, d[gene])
                seen = False
                for i in d [gene]:
                    if i == curGrp:
                       seen = True
                    if verbose:
                        print ('\t', i, seen)
                if not seen:
                    if verbose:
                        print ('Appending', gene)
                    d [gene].append (curGrp)
    
    print ('CELL GROUP DICT')
    print (d)
    print (len (d))

    print ('Cell Group ambiguities:')
    for key, value in d.items ():
        if (len (value)) > 1:
            print (key, 'has len,', len (value), ':', value)

    if args is not None and args.intermediate_write:
        json.dump (d, open (jsonfile_cell_groups, 'w'))

    return d









def prepBeforePipeline (pctl):


    print ('prepBeforePipeline................')

    """# Loading Annotation Matrix"""

    annotationMatrix = readAnnotationMatrix(support_dir + "/AnnotationMatrix.txt")

    cell_groups = getGeneCellGroups (prog_args)

    """# Loading the sample"""

    if prog_args.load_upstream is not None and prog_args.load_upstream != 'after_metadata':
        return None, annotationMatrix, cell_groups

    intermediate_h5ad = processed_dir + '/sample' + prog_args.sample + '/AfterMetadata.h5ad'

    if not prog_args.intermediate_load:

        pathSample     = raw_dir + '/' + arg_to_filename(int(prog_args.sample))
        if prog_args.load_upstream is not None and prog_args.load_upstream == 'after_metadata':
            print ('LOAD UPSTREAM after_metadata:', pathSample)
            smartSeqFoetal = sc.read (pathSample, first_column_names = True)
            pctl ['upstream_loaded'] = True
        else:
            if prog_args.debug:
                print ('LOAD NORMAL:', pathSample)

            smartSeqFoetal = loadSmartSeq(pathSample, annotationMatrix=annotationMatrix, title="Sample" + prog_args.sample)
            print ('shape of smart SeqFoetal after raw load: ', smartSeqFoetal.shape)

            """# Loading the metadata"""

            pathMetadata   = support_dir + "/metadataSample" + prog_args.sample + ".csv"
            smartSeqFoetal = loadMetadata(pathMetadata, smartSeqFoetal, verbose=False)

            smartSeqFoetal = assignOrigin(smartSeqFoetal,
                               namesIn  = ['Femur|femur', 'hip|hip1|hip2', 'liver'],
                               namesOut = ['Femur', 'Hip', 'Liver'],
                               verbose  = True)

            smartSeqFoetal = assignGate(smartSeqFoetal,
                             namesIn  = ['38\-', '38\+', 'CMP', 'GMP', 'MEP'],
                             namesOut = ['38-',  '38+',  'CMP', 'GMP', 'MEP'],
                             verbose  = True)

            smartSeqFoetal = assignCellType(smartSeqFoetal,
                                 namesIn  = ['Femur|femur', 'hip|hip1|hip2', 'liver'],
                                 namesOut = ['Single', 'Single', 'Single'],
                                 verbose  = True)

            #display(smartSeqFoetal.obs.head(5))
            print(" * Merged smartSeq object: %d genes across %d single cells"%(smartSeqFoetal.n_vars, smartSeqFoetal.n_obs))

            """# Predict the Cell Cycle state"""


            pathCellCycle = support_dir + "/regev_lab_cell_cycle_genes.txt"
            CellCycleScoring(pathCellCycle, smartSeqFoetal, annotationMatrix)


        """# Mapping Ensembl Ids to Annotated Names"""

        print ('before mappingEnseble')
        print (datetime.datetime.now ())
        mappingEnsemblToAnnotated(smartSeqFoetal, annotationMatrix)
        print (datetime.datetime.now ())
        print ('after mappingEnseble')
        smartSeqFoetal.var_names_make_unique()
        # what on earth is this for other than the very misleading rank_genes_groups mess?
        smartSeqFoetal.raw = smartSeqFoetal


        #smartSeqFoetal.write(processed_dir + '/Sample' + prog_args.sample + 'AfterMetadata.h5ad',
        if prog_args.intermediate_write:
            smartSeqFoetal.write(intermediate_h5ad, compression='gzip')
    else:
        #smartSeqFoetal = sc.read (intermediate_h5ad, first_column_names = True)
        smartSeqFoetal = sc.read (intermediate_h5ad)

    return smartSeqFoetal, annotationMatrix, cell_groups




def pipeline (smartSeqFoetal, pctl, args, dn_mode):

    def denoising_pipeline (smartSeq, pctl, pc, args, dn_mode):

        def scvi_fork (smartSeq, pc_scvi, cell_groups, args, dn_mode, pmetrics):
            from umap import UMAP
            import numpy.random as random
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib
            from scvi.inference.utils import louvain_clusters

            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

            latent = pc_scvi ['latent']
            imputed = pc_scvi ['imputed_values']
            dataset = pc_scvi ['dataset']
            full = pc_scvi ['trainer_posterior']
            batch_indices = pc_scvi ['batch_indices']
            trainer = pc_scvi ['trainer']

            if dn_mode == 'latent':
                latent_u = UMAP(spread=2).fit_transform(latent)

                print ('latent_u:')
                print (latent_u)
            else:
                imputed_u = UMAP(spread=2).fit_transform(imputed)


            if not args.noplot:
                if dn_mode == 'latent':
                    #fname = plots_dir + '/sample' + args.sample + '/scvi.tsne.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                    fname = 'scvi.tsne.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                    trainer.train_set.show_t_sne (n_samples=1000, save_name = fname, color_by = 'scalar')
                    add_to_slideshow (fname, args)


            def plot_umap (values, values_u, dataset, args):
                if args.noplot:
                    return
                cm = LinearSegmentedColormap.from_list('my_cm', ['deepskyblue', 'hotpink'], N=2)
                fig, ax = plt.subplots(figsize=(5, 5))
                ordre = np.arange(values.shape[0])
                random.shuffle(ordre)
                ax.scatter(values_u[ordre, 0], values_u[ordre, 1], 
                c=dataset.batch_indices.ravel()[ordre], 
                cmap=cm, edgecolors='none', s=5)    
                plt.axis("off")
                plt.tight_layout()
                fname = plots_dir + '/sample' + args.sample + '/scvi.umap.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                save_plot (fname, args, f=fig)
                if args.show:
                    plt.show()
                plt.close()

            if dn_mode == 'latent':
                plot_umap (latent, latent_u, dataset, args)
            else:
                plot_umap (imputed, imputed_u, dataset, args)

            if dn_mode == 'latent':
                clusters = louvain_clusters(latent, k=30, rands=0)
            else:
                clusters = louvain_clusters(imputed, k=30, rands=0)

            print ('after louvain_clusters')
            print (clusters)

            X = latent if dn_mode == 'latent' else imputed
            dsc = 'latent' if dn_mode == 'latent' else 'imputed'
            s_m = None
            s_ch = None
            s_db = None
            if 'silhouette' in parse_clustering_quality (args.clustering_quality):
                s_m = metrics.silhouette_score (X, clusters, metric = 'euclidean')
            if 'calinski_harabaz' in parse_clustering_quality (args.clustering_quality):
                s_ch = metrics.calinski_harabaz_score (X, clusters)
            if 'davies_bouldin' in parse_clustering_quality (args.clustering_quality):
                s_db = metrics.davies_bouldin_score (X, clusters)
            with open (get_metrics_filename (args, dn_mode), 'a') as f:
                # The funciton is called louvain_clusters in scvi, but it does leidenalg!
                print ('scvi-leiden', '-', dsc, ':', s_m, file = f)
            pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent']['leiden-scvi']['silhouette'] = float (s_m) if s_m is not None else None
            pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent']['leiden-scvi']['calinski_harabaz'] = float (s_ch) if s_ch is not None else None
            pmetrics ['cluster_quality']['imputed' if dn_mode == 'denoise' else 'latent']['leiden-scvi']['davies_bouldin'] = float (s_db) if s_db is not None else None


            def plot_clusters_in_umap (values_u, clusters, args):
                if args.noplot:
                    return
                '''
                colors = ["#991f1f", "#ff9999", "#ff4400", "#ff8800", "#664014", "#665c52",
                          "#cca300", "#f1ff33", "#b4cca3", "#0e6600", "#33ff4e", "#00ccbe",
                          "#0088ff", "#7aa6cc", "#293966", "#0000ff", "#9352cc", "#cca3c9", "#cc2996"]
                '''
                colors = myColors
                plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
                for i, k in enumerate(np.unique(clusters)):
                    plt.scatter(values_u[clusters == k, 0], values_u[clusters == k, 1], label=k,
                            edgecolors='none', c=colors[k], s=5)
                    plt.legend(borderaxespad=0, fontsize='large', markerscale=5)

                plt.axis('off')
                plt.tight_layout()
                fname = plots_dir + '/sample' + args.sample + '/scvi.clusters_in_umap.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                save_plot (fname, args, plt=plt)
                if args.show:
                    plt.show()
                plt.close()

            if dn_mode == 'latent':
                plot_clusters_in_umap (latent_u, clusters, args)
            else:
                plot_clusters_in_umap (imputed_u, clusters, args)

            if dn_mode != 'latent':
                return

            print ('before one_vs_all_degenes')

            de_res, de_clust = full.one_vs_all_degenes(cell_labels=clusters,
                                           n_samples=10000, 
                                           M_permutation=10000,
                                           output_file=False,
                                           min_cells=1)

            print ('after one_vs_all_degenes')
            print ('de_res:')
            print (de_res)
            print ('de_clust:')
            print (de_clust)

            def cluster_marker_heatmap (deres, clusters, args):
                if args.noplot:
                    return

                markers = []
                for x in deres:
                    markers.append (x[:10])
                markers = pd.concat (markers)

                print ('heatmap markers:')
                print (markers)

                genes = np.asarray(markers.index)
                print ('genes')
                print (genes)
                print ('len_genes')
                print (len (genes))
                expression = [x.filter(items=genes, axis=0)['norm_mean1'] for x in deres]
                print ('expression')
                print (expression)
                expression = pd.concat(expression, axis=1)
                print ('expression')
                print (expression)
                expression = np.log10(1 + expression)
                expression.columns = de_clust

                print ('expression')
                print (expression)

                plt.figure(figsize=(50, 25))
                im = plt.imshow(expression, cmap='RdYlGn', interpolation='none', aspect='equal')
                ax = plt.gca()
                ax.set_xticks(np.arange(0, len (de_clust), 1))
                ax.set_xticklabels(de_clust, rotation='vertical')
                ax.set_yticklabels(genes)
                ax.set_yticks(np.arange(0, len (genes), 1))
                ax.tick_params(labelsize=14)
                plt.colorbar(shrink=0.2)

                fname = plots_dir + '/sample' + args.sample + '/scvi.gene_heatmap.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                save_plot (fname, args, plt=plt)
                if args.show:
                    plt.show ()
                plt.close ()


            cluster_marker_heatmap (de_res, de_clust, args)

            gene_names = {}
            scores = {}
            for x, c in zip (de_res, de_clust):
                markers = x[:100]
                gene_names [c] = np.asarray(markers.index)
                scores [c] = np.asarray(markers ['bayes1'])

                print ('Cluster:', c, gene_names[c])
                print ('Cluster:', c, scores [c])
            mg_matrix = cellGroupBins (de_clust, gene_names, scores, cell_groups, 'leiden', args, dn_mode, pmetrics,
                                      pipeline_id = 'fork', plot_id = 'scvi')
            #correlationAnalysis (smartSeq, args, dn_mode, mg_matrix, 'leiden', pipeline_id = 'fork')

            def find_markers(deres, absthres, relthres, ngenes):
                allgenes = []
                for i, x in enumerate(deres):
                    markers = x.loc[(x['mean1'] > absthres) & (x['norm_mean1'] / x['norm_mean2'] > relthres)]
                    if len(markers>0):
                        ngenes = np.min([len(markers), ngenes])
                        markers = markers[:ngenes]
                        allgenes.append(markers)
                if len(allgenes)>0:
                    markers = pd.concat(allgenes)
                    return markers
                else: 
                    return pd.DataFrame(columns=['bayes1','mean1','mean2','scale1','scale2','clusters'])

            clustermarkers = find_markers(de_res, absthres=0.5, relthres=2, ngenes=3)
            #clustermarkers[['bayes1', 'mean1', 'mean2', 'scale1', 'scale2', 'clusters']]

            print ('clustermarkers')
            print (clustermarkers)

            #Markers = ["CD3D", "SELL", "CREM", "CD8B", "GNLY", "CD79A", "FCGR3A", "CCL2", "PPBP"]
            Markers = list (cell_groups.keys ())

            def plot_marker_genes(latent_u, count, genenames, markers, args, id):
                if args.noplot:
                    return
                nrow = (len(markers) // 3 + 1)
                figh = nrow * 4
                plt.figure(figsize=(20, figh))
                for i, x in enumerate(markers):
                    if np.sum(genenames == x)==1:
                        exprs = count[:, genenames == x].ravel()
                        idx = (exprs > 0)
                        plt.subplot(nrow, 3, (i + 1))
                        plt.scatter(latent_u[:, 0], latent_u[:, 1], c='lightgrey', edgecolors='none', s=5)
                        plt.scatter(latent_u[idx, 0], latent_u[idx, 1], c=exprs[idx], cmap=plt.get_cmap('viridis_r'),
                                  edgecolors='none', s=3)
                    plt.title(x)
                    plt.tight_layout()
                fname = plots_dir + '/sample' + args.sample + '/scvi.marker_genes.' + id + '.' + get_plot_label (args, denoise_id = dn_mode) + '.' + plot_format
                save_plot (fname, args, plt=plt)
                if args.show:
                    plt.show()
                plt.close ()

            if len(clustermarkers) > 0:
                plot_marker_genes(latent_u[clusters >= 0, :], dataset.X[clusters >= 0, :], 
                              dataset.gene_names,
                              np.asarray(Markers), args, 'basic')
            '''
            markergenes = ["CD3D", "CREM", "HSPH1", "SELL", "GIMAP5", "CACYBP", "GNLY", 
                "NKG7", "CCL5", "CD8A", "MS4A1", "CD79A", "MIR155HG", "NME1", "FCGR3A", 
                "VMO1", "CCL2", "S100A9", "HLA-DQA1", "GPR183", "PPBP", "GNG11", "HBA2", 
                "HBB", "TSPAN13", "IL3RA", "IGJ"]
            '''
            markergenes = list (cell_groups.keys ())

            percluster_exprs = []
            marker_names = []
            for marker in markergenes:
                if np.sum(dataset.gene_names == marker) == 1:
                    mean = [np.mean(dataset.X[clusters == i, dataset.gene_names == marker]) for i in np.unique(clusters)]
                    mean = np.asarray(mean)
                    percluster_exprs.append(np.log10(mean / np.mean(mean) + 1))
                    marker_names.append(marker)

            print ('percluster_exprs')
            print (percluster_exprs)
            print ('marker_names')
            print (marker_names)

            if len(percluster_exprs) > 0:
                percluster_exprs = pd.DataFrame(percluster_exprs, index=marker_names)
                sns.clustermap(percluster_exprs, row_cluster=False, col_cluster=True)

            plot_marker_genes(latent_u[clusters >= 0, :], dataset.X[clusters >= 0, :],
                  dataset.gene_names, np.asarray(clustermarkers.index), args, 'perc')

            de_res_stim, de_clust_stim = full.within_cluster_degenes(cell_labels=clusters,
                                     states=dataset.batch_indices.ravel() == 1,
                                     output_file=False, batch1=[1], batch2=[0],
                                     #save_dir=save_path, filename='Harmonized_StimDE',
                                     min_cells=1)

            genelist = []
            for i, x in enumerate(de_clust_stim):
                de = de_res_stim[i].loc[de_res_stim[i]["mean1"] > 1]
                de = de.loc[de["bayes1"] > 2]
                if len(de) > 0:
                    de["cluster"] = np.repeat(x, len(de))
                    genelist.append(de)

            print ('initial genelist')
            print (genelist)
        
            if len(genelist) > 0:
                genelist = pd.concat(genelist)
                genelist["genenames"] = list(genelist.index)
                degenes, nclusterde = np.unique(genelist.index, return_counts=True)

            if len(genelist) > 0:
                print(", ".join(degenes[nclusterde > 11]))

            if len(genelist) > 0:
                cluster0shared = genelist.loc[genelist['genenames'].isin(degenes[nclusterde > 10])]
                cluster0shared = cluster0shared.loc[cluster0shared['cluster'] == 0]

            print ('final genelist')
            print (genelist)

            def plot_marker_genes_compare(latent_u, count, genenames, markers, subset, args, id):
                if args.noplot:
                    return
                nrow = len(markers)
                figh = nrow * 4
                plt.figure(figsize=(8, figh))
                notsubset = np.asarray([not x for x in subset])
                for i, x in enumerate(markers):
                    if np.sum(genenames == x) == 1:
                        exprs = count[:, genenames == x].ravel()
                        idx = (exprs > 0)
                        plt.subplot(nrow, 2, (i * 2 + 1))
                        plt.scatter(latent_u[subset, 0], latent_u[subset, 1], c='lightgrey', edgecolors='none', s=5)
                        plt.scatter(latent_u[idx, 0][subset[idx]], latent_u[idx, 1][subset[idx]], c=exprs[idx][subset[idx]],
                                    cmap=plt.get_cmap('viridis_r'), edgecolors='none', s=3)
                        plt.title(x + ' control')
                        plt.tight_layout()
                        plt.subplot(nrow, 2, (i * 2 + 2))
                        plt.scatter(latent_u[notsubset, 0], latent_u[notsubset, 1], c='lightgrey', edgecolors='none', s=5)
                        plt.scatter(latent_u[idx, 0][notsubset[idx]], latent_u[idx, 1][notsubset[idx]],
                                    c=exprs[idx][notsubset[idx]], cmap=plt.get_cmap('viridis_r'), edgecolors='none', s=3)
                        plt.title(x + ' stimulated')
                fname = plots_dir + '/sample' + args.sample + '/scvi.marker_genes_compare.' + get_plot_label (args, denoise_id = dn_mode) + '.' + id + '.' + plot_format
                save_plot (fname, args, plt=plt)
                if args.show:
                    plt.show()
                plt.close()

            plot_marker_genes_compare(latent_u, dataset.X, dataset.gene_names, 
                          ["CD3D", "GNLY", "IFI6", "ISG15", "CD14", "CXCL10"], batch_indices == 0, args, 'basic')

            if len(genelist) > 0:
                plot_marker_genes_compare(latent_u, dataset.X, 
                          dataset.gene_names, cluster0shared.index, 
                          batch_indices == 0, args, 'other')

            if len(genelist) > 0 and len(nclusterde) > 0:
                degenes[nclusterde == 1]
                clusteruniq = genelist.loc[genelist['genenames'].isin(degenes[nclusterde == 1])]
                clusteruniq = clusteruniq.loc[clusteruniq['cluster'] == 3]
                plot_marker_genes_compare(latent_u, dataset.X, dataset.gene_names, clusteruniq.index, batch_indices == 0, args, 'last')



        def upstream_preparation (smartSeq, pctl, args, dn_mode):

            print ('upstream_preparation.......')

            """# Quality Control"""

            plotHighestExprGenes (smartSeq, dn_mode)

            """## Violin plot before removing bulk and empty cells"""

            plotViolinQuality(smartSeq, dn_mode, listVariables=['n_genes', 'n_counts', 'percent_mito', "ercc_content"], id = 'raw')

            ''' Remove bulk and empty cells (added by me from Andrea's subsequent additions)'''

            print(" * smartSeq object before removing bulk and empty cells: %d genes across %d single cells"%
                     (smartSeq.n_vars, smartSeq.n_obs))

            smartSeq.obs["type"] = 'Single'
            if not args.nofoetal:
                bulk  = smartSeq.obs["type"][smartSeq.obs["COHORT"] == "Bulk"]
                smartSeq.obs['type'].replace(bulk, "Bulk", inplace=True)
                empty = smartSeq.obs["type"][smartSeq.obs["COHORT"] == "Empty"]
                smartSeq.obs['type'].replace(empty, "Empty", inplace=True)
                smartSeq.obs['type'] = smartSeq.obs.type.astype('category')

                notBulkAndEmpty = smartSeq.obs.index[smartSeq.obs["COHORT"] == "nan"].tolist()
                smartSeq= smartSeq[notBulkAndEmpty]

            print(" * smartSeq object after removing bulk and empty cells: %d genes across %d single cells"%
                     (smartSeq.n_vars, smartSeq.n_obs))

            """## Perform QC after removing bulk and empty cells"""

            use_old_method=False

            if use_old_method:
                thresNumGen   = 1000
                thresPercMit  = 0.25
                thresErccCont = 0.6

                plotHistogramQuality(smartSeq, dn_mode,
                         thresNumGen   = thresNumGen,
                         thresPercMit  = thresPercMit,
                         thresErccCont = thresErccCont,
                         bins=100)
            else:
                min_counts  = 10000
                max_counts  = 2800000
                min_genes   = 1500
                max_pctMito = 0.25
                max_erccCon = 0.6

            if not args.nofoetal:
                """## Cropping cells"""

                print(" * SmartSeq2 object before Cropping cells: %d genes across %d single cells"%
                      (smartSeq.n_vars, smartSeq.n_obs))

                nCells = smartSeq.n_obs

                def old_cropping_operations (smartSeq):
                    tmp = smartSeq[smartSeq.obs['n_genes'] >= thresNumGen, :].copy()
                    tmp = tmp[tmp.obs['percent_mito'] <= thresPercMit, :].copy()
                    tmp = tmp[tmp.obs['ercc_content'] <= thresErccCont, :]
                    return tmp

                def new_cropping_operations (smartSeq):
                    tmp = smartSeq.copy ()
                    print('Total number of cells before cropping: %d'%(tmp.n_obs))

                    sc.pp.filter_cells(tmp, min_counts = min_counts)
                    loosing = (100.-100.*(float(tmp.n_obs)/nCells))
                    print('Number of cells after min count filter: %d, lost %2.f%% of cells'%(tmp.n_obs, loosing))

                    if not args.nofiltermaxcount:
                        sc.pp.filter_cells(tmp, max_counts = max_counts)
                        loosing = (100.-100.*(float(tmp.n_obs)/nCells))
                    print('Number of cells after max count filter: %d, lost %2.f%% of cells'%(tmp.n_obs, loosing))

                    sc.pp.filter_cells(tmp, min_genes = min_genes)
                    print('Number of cells after gene filter: %d, lost %2.f%% of cells'%(tmp.n_obs, loosing))

                    tmp = tmp[tmp.obs['percent_mito'] < max_pctMito]
                    loosing = (100.-100.*(float(tmp.n_obs)/nCells))
                    print('Number of cells after MT filter: %d, lost %2.f%% of cells'%(tmp.n_obs, loosing))

                    tmp = tmp[tmp.obs['ercc_content'] < max_erccCon]
                    loosing = (100.-100.*(float(tmp.n_obs)/nCells))
                    print('Number of cells after ERCC filter: %d, lost %2.f%% of cells'%(tmp.n_obs, loosing))

                    sc.pp.filter_genes(tmp, min_cells=20)
                    return tmp                    


                if use_old_method:
                    tmp = old_cropping_operations (smartSeq)
                else:
                    tmp = new_cropping_operations (smartSeq)

                beforeRemovingS  = smartSeq.obs.index
                afterRemovingS   = tmp.obs.index
                indeces          = np.invert(beforeRemovingS.isin(afterRemovingS.tolist()))
                removedCellsS    = beforeRemovingS[indeces]

                smartSeq.obs['removed'] = 'No'
                removed = smartSeq.obs['removed'][removedCellsS]
                smartSeq.obs['removed'].replace(removed, "Yes", inplace=True)
                smartSeq.obs['removed'] = smartSeq.obs.removed.astype('category')

                plotScatterPlotQuality(smartSeq, dn_mode, colors=["gray", "red"], cat='removed', pointSize=150)

                # setting the correct name for downstream analysis
                smartSeq = tmp

                print(" * SmartSeq2 object after Cropping cells: %d genes across %d single cells"%
                  (smartSeq.n_vars, smartSeq.n_obs))

                if use_old_method:
                    print(" \t* %d cells removed using thresholds %d %.2f %.2f"%(nCells - smartSeq.n_obs, thresNumGen, thresPercMit, thresErccCont))

                print(" \t* %.3f%% of cells removed"%(100.-100.*(float(smartSeq.n_obs)/nCells)))

            save_h5ad (smartSeq, args, dca_id = get_plot_label (args, denoise_id = dn_mode), id = 'AfterCropping', denoise_id = dn_mode)
            write_text_matrix (smartSeq, args,
                results_dir + '/sample' + args.sample + '/sample' + args.sample + '.AfterCropping.tsv')

            if args.extractHVGs:
                tmp = smartSeq.copy ()
                min_mean=0.1
                max_mean=10
                min_disp=0.25
                sc.pp.normalize_per_cell(tmp, counts_per_cell_after=1e4)
                sc.pp.log1p(tmp)
                sc.pp.highly_variable_genes(tmp, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
                examine_adata (smartSeq, "Before extractHVGs")
                print ('extract_HVGs')
                print (tmp.var['highly_variable'])
                print (tmp.var['highly_variable'].values)
                print (smartSeq[:, tmp.var['highly_variable']])
                print (smartSeq[:, tmp.var['highly_variable']].X)
                print (np.sum (tmp.var['highly_variable'].index is None))
                print (np.sum(smartSeq.X, axis=0) > 10)
                #smartSeq = smartSeq[:, np.sum (smartSeq.X, axis = 0) > 100].copy ()
                smartSeq = smartSeq[:, tmp.var['highly_variable'].values].copy ()
                # this was the cause of a lot of pain and it is of dubious use
                smartSeq.raw = smartSeq
                examine_adata (smartSeq, "After extractHVGs")

            print ('Zero analysis:')
            zeroes = np.count_nonzero (smartSeq.X == 0, axis=1)
            print (zeroes)
            non_zeroes = np.count_nonzero (smartSeq.X, axis=1)
            print (non_zeroes)
            max_vals = np.max (smartSeq.X, axis=1)
            print (max_vals)
            min_vals = np.min(np.where(smartSeq.X==0, smartSeq.X.max(), smartSeq.X), axis=1)
            print (min_vals)
            print ('Percentiles:')
            pcs = np.percentile (smartSeq.X, [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=1)
            print (pcs)
            for n, z in zip (non_zeroes, zeroes):
                print ('zeroes', z, ':', n, 'non zeroes')
            for n, x in zip (min_vals, max_vals):
                print ('min', n, ':', x, 'max')

            return smartSeq


        if args.load_upstream is None or pctl ['upstream_loaded']:
            smartSeq = upstream_preparation (smartSeq, pctl, args, dn_mode)
            examine_adata (smartSeq, "after upstream_preparation")
        elif args.load_upstream == 'after_cropping':
            pathSample     = raw_dir + '/' + arg_to_filename(int(prog_args.sample))
            if prog_args.debug:
                print ('LOAD UPSTREAM after_cropping:', pathSample)
            smartSeq = sc.read (pathSample, first_column_names = True)
            pctl ['upstream_loaded'] = True
        else:
            print ('ERROR: non valid upstream load point')
            quit ()



        smartSeq = denoise_decide (smartSeq, pc, pctl ['prog_metrics'], args, dn_mode, moment = 'denoise_after_cropping')


        if args.denoiser == 'scvi' and not args.noscvifork:
            scvi_fork (smartSeq, pc ['scvi_control'], cell_groups, args, dn_mode, pctl ['prog_metrics'])
            if args.onlyscvifork:
                return




        plotViolinQuality(smartSeq, dn_mode, listVariables=['n_genes', 'n_counts', 'percent_mito', "ercc_content"])
        plotScatterPlotDifferentGroup(smartSeq, dn_mode)

        save_h5ad (smartSeq, args, dca_id = get_plot_label (args, denoise_id = dn_mode), id = 'BeforePreAnalysis', denoise_id = dn_mode)

        listRegress=['n_genes', 'percent_mito']

        if not args.nopreanalysis:
            print ('before preAnalysis')
            print (smartSeq.X.shape)
            print (smartSeq)
            preAnalysis(smartSeq, dn_mode, args, min_mean=0.125, max_mean=3, min_disp=0.6, listRegress=listRegress)
            print ('after preAnalysis')
            print (smartSeq)

        smartSeq = denoise_decide (smartSeq, pc, pctl ['prog_metrics'], args, dn_mode, moment = 'denoise_after_preanalysis')

        integratedAnalysis(smartSeq, dn_mode, n_pcs=9)
        smartSeq = denoise_decide (smartSeq, pc, pctl ['prog_metrics'], args, dn_mode, moment = 'denoise_after_integratedanalysis')

        for method in clustering_methods:
            clustering(smartSeq, args, dn_mode, pctl ['prog_metrics'],
                  #resolution=1.3, method=method, palette=myColors, spaces = ['tsne', 'umap'])
                  resolution=args.clustering_resolution, method=method, palette=myColors, spaces = ['tsne', 'umap'])

        save_h5ad (smartSeq, args, dca_id = get_plot_label (args, denoise_id = dn_mode), id = 'AfterClustering', denoise_id = dn_mode)
    
        for method in clustering_methods:
            calculateMarkerGenes(smartSeq, dn_mode, cell_groups, clustering=method)
            cellGroupAnalysis (smartSeq, cell_groups, method, args, dn_mode, pctl ['prog_metrics'])
        if dn_mode == 'latent':
            latent_correlationAnalysis (smartSeq, args, dn_mode)

        markerCleanEprs = readMarkerList(markerListPath, smartSeq, annotationMatrix)

        for gene in markerCleanEprs:
            for method in clustering_methods:
                plotViolinAndSpace(smartSeq, gene, dn_mode, palette=myColors, cluster=method, space='tsne', width=8, height=8)



    scvi_control = {'latent': None,
          'imputed_values' : None,
          'trainer_posterior' : None,
          'batch_indices' : None,
          'trainer' : None,
          'dataset': None }

    pipeline_control = {'denoising_completed':False,
          'initial_copy': None,
          'created_copy': None,
          'denoise_after_metadata': None,
          'denoise_after_cropping': None,
          'denoise_after_preanalysis': None,
          'denoise_after_integratedanalysis': None,
          'scvi_control' : scvi_control }

    smartSeqFoetal = denoise_decide (smartSeqFoetal, pipeline_control, pctl ['prog_metrics'], args, dn_mode,
            moment = 'denoise_after_metadata')
    if args.denoise_mode == 'full':
        if pipeline_control ['created_copy'] is not None:
            pipeline_control ['initial_copy'] = pipeline_control['created_copy']
        else:
            pipeline_control ['initial_copy'] = smartSeqFoetal.copy ()
            pipeline_control ['denoise_after_metadata'] = pipeline_control ['initial_copy']
    print ('MAIN RUN', dn_mode)
    denoising_pipeline (smartSeqFoetal, pctl, pipeline_control, args, dn_mode)
    if args.denoise_mode == 'full':
        print ('LATENT RUN')
        denoising_pipeline (pipeline_control['denoise_after_metadata'], pctl, pipeline_control, args, 'latent')




modes = ['denoise', 'latent'] if prog_args.denoise_mode == 'full' else [prog_args.denoise_mode]
for mode in modes:
    f = open (get_metrics_filename (prog_args, mode), 'w')
    f.close ()


if prog_args.skip_to is not None:
    if prog_args.skip_to == 'cell_group_analysis':
        dn_mode = prog_args.denoise_mode
        if dn_mode != 'denoise' and dn_mode != 'latent':
            print ('Incorrect denoise mode for cell group analysis', dn_mode)
            exit (1)
        smartSeqFoetal = load_h5ad (prog_args, dca_id = get_plot_label (prog_args, denoise_id = dn_mode), id = 'AfterRankGenesGroups')
        print ('CELL_GROUP_ANALYSIS:skip_to:')
        print (smartSeqFoetal)
        print (smartSeqFoetal.X.shape)
        cell_groups = getGeneCellGroups (prog_args)
        cellGroupAnalysis (smartSeqFoetal, cell_groups, 'louvain', prog_args, dn_mode)
    exit (0)




prog_metrics = {
    'hash':None,
    'sysargs':sys.argv,
    'progargs':pformat (prog_args),
    'execution_time':0,
    'params': {
        'denoiser':prog_args.denoiser,
        'reconstruction_loss':prog_args.reconstruction_loss,
        'dropout_rate':prog_args.dropout_rate,
        'input_dropout_rate':prog_args.input_dropout_rate,
        'ridge':prog_args.ridge,
        'l1':prog_args.l1,
        'l2':prog_args.l2,
        'l1_enc':prog_args.l1_enc,
        'l2_enc':prog_args.l2_enc,
        'batchnorm':not prog_args.nobatchnorm,
        'extractHVGs':prog_args.extractHVGs,
        'filter_genes':prog_args.emulate_dca_preprocess,
        'lr':str (prog_args.lr),
        'kl':prog_args.kl,
        'n_latent':prog_args.n_latent,
        'n_layers':prog_args.n_layers,
        'n_hidden':prog_args.n_hidden,
        'n_epochs':str(prog_args.n_epochs),
        'patience':prog_args.patience,
        'cl_res':prog_args.clustering_resolution,
        'hidden_size':prog_args.hidden_size,
        'variational':not prog_args.novariational,
        'size_factor_algorithm':prog_args.size_factor_algorithm if not prog_args.nosizefactors else None,
        'patience_reduce_lr':prog_args.patience_reduce_lr,
        'factor_reduce_lr':prog_args.factor_reduce_lr,
        'discriminator_wasserstein':prog_args.discriminator_wasserstein,
        'wasserstein_lambda':prog_args.wasserstein_lambda,
        'discriminator_hidden_size':prog_args.discriminator_hidden_size,
        'discriminator_prior_sigma':prog_args.discriminator_prior_sigma
        },
    'loss': {
        'train': {
            'full':None,
            'final':None
            },
        'test': {
            'full':None,
            'final':None
            },
        'discriminator': {
            'train': {
                'full':None,
                'final':None
                },
            'test': {
                'full':None,
                'final':None
                }
            },
        'generator': {
            'train': {
                'full':None,
                'final':None
                },
            'test': {
                'full':None,
                'final':None
                }
            }
        },
    'cluster_confidence': {
        'imputed': {
            'louvain': {
                'FractionCount':None,
                'FractionScore':None,
                'FractionMeanScore':None
                },
            'leiden': {
                'FractionCount':None,
                'FractionScore':None,
                'FractionMeanScore':None
                }
             },
        'latent': {
            'louvain': {
                'FractionCount':None,
                'FractionScore':None,
                'FractionMeanScore':None
                },
            'leiden': {
                'FractionCount':None,
                'FractionScore':None,
                'FractionMeanScore':None
                }
             }
        },
    'cluster_quality': {
        'imputed': {
            'louvain': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                },
            'leiden': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                },
            'leiden-scvi': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                }
            },
        'latent': {
            'louvain': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                },
            'leiden': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                },
            'leiden-scvi': {
                'silhouette':None,
                'calinski_harabaz':None,
                'davies_bouldin':None
                }
            }
        }
    }
prog_control = {'upstream_loaded':False, 'prog_metrics':prog_metrics}


currentDT = datetime.datetime.now()
jsonfile_metrics = get_metrics_pathname (prog_args) + '/metrics'
if prog_args.evaluation_bundle is not None:
    jsonfile_metrics = jsonfile_metrics + '.' + prog_args.evaluation_bundle
jsonfile_metrics = jsonfile_metrics + '.' + currentDT.strftime("%Y-%m-%d_%H-%M-%S")
jsonfile_metrics = jsonfile_metrics + '.json'

exitVal = 0
try:
    smartSeqFoetal, annotationMatrix, cell_groups = prepBeforePipeline (prog_control)
    pipeline (smartSeqFoetal, prog_control, prog_args, dn_mode = ('latent' if prog_args.denoise_mode == 'latent' else 'denoise'))
except Exception as e:
    print (e)
    traceback.print_exc()
    exitVal = 1

prog_metrics ['hash'] = str(int(hashlib.md5(str(prog_metrics['params']).encode('utf-8')).hexdigest(), 16))
print (prog_metrics)
print ('types:')
for k, v in prog_metrics.items():
    print (k, v)
    print (type (k))
print (type(prog_metrics))
print (type(prog_args.dropout_rate))
print (type(prog_args.input_dropout_rate))
print (type(prog_args.ridge))
print (type(prog_args.l1))
print (type(prog_args.l2))
print (type(prog_args.l1_enc))
print (type(prog_args.l2_enc))
print (type(prog_args.lr))
print (type(prog_args.kl))
print (type(prog_metrics ['loss']['train']['final']))
print (type(prog_metrics ['loss']['test']['final']))
if prog_args.evaluation_bundle is None or (prog_args.evaluation_bundle is not None and exitVal == 0):
    json.dump (prog_metrics, open (jsonfile_metrics, 'w'), indent=2)
exit (exitVal)

