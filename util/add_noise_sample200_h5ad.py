#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:09:06 2019

@author: 8228T
"""

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy.api as sc

import scipy as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as stats

import sys, argparse, os
from shutil import copyfile



def parse_args():
    parser = argparse.ArgumentParser(description='Plotter')

    parser.add_argument('--base_dir', type=str, help='base directory')
    parser.add_argument('--sample', type=str, help='Sample to be made noisy, e.g. 1, 2, Merged')
    parser.add_argument('--cropped', type=str, help='Noise a cropped sample or a sample that is not cropped: yes for cropped')
    parser.add_argument('--method', type=str, help='Noise method: pink or gaussian')
    parser.add_argument('--noise', type=float, help='The noise content for gaussian denoising: between 0.0 and 1.0')
    parser.add_argument('--sigma', type=float, help='The standard deviation for gaussian denoising')
    parser.add_argument('--pink_factor', type=float, help='The factor of the pink noise, factor/f')
    parser.add_argument('--debug', type=str, help='print off values? Yes if you want to, omit if you do not')

    return parser.parse_args()

args = parse_args ()


print ('sample = ', args.sample, ' cropped = ', args.cropped, ' method = ', args.method, ' noise = ', args.noise, ' sigma = ', args.sigma, ' debug = ', args.debug)


data_dir = args.base_dir + '/data/RnaSeqGeneMatrix'
raw_dir = data_dir + '/raw'
processed_dir = data_dir + '/processed'


if args.cropped == 'yes':
    if args.sample == 'Merged':
        filename = raw_dir + '/MergedSamplesAfterQC.h5ad'
        
    elif args.sample in str([1,2,3,4,5]):
        filename = raw_dir + '/Sample' + args.sample + 'AfterQC.h5ad'   

    
elif args.cropped == 'no':
    if args.sample == 'Merged':
        filename = raw_dir + '/MergedSamples.h5ad'
    
    elif args.sample in str([1,2,3,4,5]):
        filename = raw_dir + '/Sample' + args.sample + '.h5ad'
        
    
print ('Using file: ', filename)

#filename = processed_dir + '/sample200/' + 'Sample200_scvi_denoise_after_cropping_denoise_AfterCropping.h5ad'
#filename_clustered = processed_dir + '/sample1/' + 'Sample1_dca_denoise_after_metadata_denoise_AfterClusteringByLouvain.h5ad'
#filename_raw = raw_dir + '/MergedSamples.h5ad'

#%%


#cluster = '0'
#gene = 'CFH'
#gene = 'TSPAN6'

def examine_adata (adata, msg):
    print (msg, ':Number of genes: ', adata.n_vars, ' Number of cells: ', adata.n_obs)

'''
def select_samples (adata, cluster, gene):

    cell_samples = adata[adata.obs['leiden'].isin([cluster])].X
    #cell_samples = np.array (adata[adata.obs['leiden'].isin([cluster])].X)
    
    #print ('Analysing the cells in cluster: ' + cluster)
    
    #print ('CELL SAMPLES:')
    #print (cell_samples)
    #print ('Size of category' + cluster + ': ', cell_samples.shape)

    gene_list = adata.var_names.isin ([gene])
    #print ('Analysing the expression of gene: ' + gene)
    
    #print ('Number of True:', np.sum (gene_list))
    #print (gene_list)
    #print ('Size of gene' + gene + ': ', gene_list.shape)
    
    samples = cell_samples[:,gene_list]
    
    #print ('Size of array after indexing: ', samples.shape)
    return samples

def select_genes (adata, gene):
    
    #print ('Analysing the expression of gene: ' + gene)

    #samples = adata[adata.var_names.isin([gene])].X - PICKS OUT ALL GENES, WRONG!

    samples = adata[:,adata.var_names.isin([gene])].X
    
    #samples = np.array (adata[adata.var_names.isin([gene])].X)
    #print ('Before:', samples)
    #adata[adata.var_names.isin([gene])].X = samples
    #print ('AFTER', adata[adata.var_names.isin([gene])].X)
    #print ('............................')
    #print (samples)
    #print ('GENE SAMPLES')
    #print (samples)
    #print (samples.shape)
    return samples
'''

#%%
#samples = select_samples (cluster, gene)
#samples = select_genes (gene)
#select_genes (gene)
#select_samples (cluster, gene)

#%%
#noisy_samples = samples.copy()

def generate_pink_noise (samples):
  
    noisy_samples = samples.copy()
    #print ('Size of noisy samples before noising: ', noisy_samples.shape)
    
    #f = plt.figure (1)
    #(n, bins, patches) = plt.hist (np.hstack (samples), bins='auto', alpha=0.5, density=False, label = 'samples')
    #f.show ()
    #plt.close (f)
   
    (n, bins) = np.histogram (samples, bins='auto', density=False)
    print ('Computed histogram')
    #print ('n: ', n)
    #print ('bins: ', bins)

    ones = np.ones_like (n, dtype=float)
    #print ('ones: ', ones)
    #print ('size of ones', n.shape, bins.shape, ones.shape)
    #print ('computing zero like ones: ', np.zeros_like (ones))
 
    #pink_noise = np.divide (1, n)
    pink_noise = args.pink_factor * np.divide (ones, n, out=np.zeros_like (ones), where=n!=0)

    for idx, val in enumerate(samples):
        for j in range (len(bins) - 1):
            if bins [j] <= val <= bins [j+1]:
                noisy_samples [idx] = val + pink_noise [j]
                if args.debug == 'yes':
                    #print (val , bins [j], bins [j+1], pink_noise [j])
                    print (idx, val, noisy_samples [idx], bins [j], bins [j+1])
                    #if n [j] == 0:
                    #   print ('n is zero: ', n[j], pink_noise [j])
                break
                
            #if args.debug == 'yes':
            #    print (idx, val, noisy_samples [idx])
    #print ('ORIGINAL: ', samples)
    #print ('NOISY: ', noisy_samples)
    #print ('Size of noisy samples after noising: ', noisy_samples.shape)
                
    return noisy_samples

#return pink noise?

#add gaussian noise for a particular gene
def generate_gaussian_noise (samples, noise, sigma, mu):

    noisy_samples = samples.copy()
    #noisy_samples = samples
    #noise = np.random.randn (*samples.shape)
    samples_size = int (len (samples))
    subset_size = int (len (samples) * noise)
    #print ('SIZES:')
    #print (samples_size, subset_size)
    #create array of random integers:
    indices = np.random.randint (0, samples_size - 1, subset_size)
    #print ('INDICES')
    #print (indices.shape)
    
    for i in range (samples_size):
        for j in range (subset_size):
            if i == indices [j]:
                noisy_samples [i] += (sigma * np.random.randn() + mu)
                
        if args.debug == 'yes':
            print (samples [i], noisy_samples [i])
        
    '''
        if samples [i] != noisy_samples [i]:
            print ('Different!')
        else:
            print ('Same!')
       
    '''
        
    for i in range (samples_size):
        noisy_samples [i] += (sigma * np.random.randn() + mu)

        if noisy_samples [i] < 0.0:
            noisy_samples [i] = 0.0
        if noisy_samples [i] < 0.0:
            print ('Less than zero!')

    return noisy_samples

#%%
#shuffle values??


#%%
#iterate over genes

def iterate_over_genes (adata, noise, sigma):
    for idx, g in enumerate(adata.var_names):
        #samples = select_genes (adata, g)
        samples = adata[:,adata.var_names.isin([g])].X
        print ('Analysing gene: ', idx, g)
        #adata[adata.var_names.isin([g])].X = generate_pink_noise (samples)
        
        if args.method == 'pink':
            adata[:,adata.var_names.isin([g])].X = generate_pink_noise (samples).reshape ((adata.n_obs,1))
        elif args.method == 'gaussian':
            adata[:,adata.var_names.isin([g])].X = generate_gaussian_noise (samples,
                noise = noise, mu = 0, sigma = sigma).reshape((adata.n_obs,1))
        #if idx > 1000:
        #    break

        #print (adata[adata.var_names.isin([g])].X)
        
        '''
        g = plt.figure(idx)
        plt.plot (samples, color = 'b', label = 'samples')
        plt.plot (adata[adata.var_names.isin([g])].X, color = 'r', label = 'samples + pink noise')

        plt.xlabel ('Cell')
        plt.ylabel ('Relative Gene Expression')
        plt.title (g)
        plt.legend(loc='upper right')
        g.savefig ('noisy_gene_expression.png')
        g.show ()
        '''

def write_noisy_adata (adata):
    #fname=raw_dir + '/gaussian_0.1_MergedSamples.h5ad'
    if args.method == 'gaussian':

        if args.cropped == 'no':
            fname=raw_dir + '/gaussian_positive' + '/Sample' + args.sample + '_' + args.method + '_' + str(args.noise) + '_' + str(args.sigma) + '.h5ad'

        elif args.cropped == 'yes':
            fname=raw_dir + '/gaussian_positive' + '/Sample' + args.sample + 'AfterQC' + '_' + args.method + '_' + str(args.noise) + '_' + str(args.sigma) + '.h5ad'
    
    elif args.method == 'pink':

        if args.cropped == 'no':
            fname=raw_dir + '/Sample' + args.sample + '_' + args.method + '_' + str(args.pink_factor) + '_' + '.h5ad'

        elif args.cropped == 'yes':
            fname=raw_dir + '/Sample' + args.sample + 'AfterQC' + '_' + args.method + '_' + str(args.pink_factor) + '_' + '.h5ad'
    
    print ('About to write h5ad:', fname)
    examine_adata (adata, 'Noisy before write')
    adata.write (fname, compression='gzip')
    print ('Written ', fname)

#%%
adata = sc.read (filename, first_column_names = True)
examine_adata (adata, "Start of program")
#print ('Number of genes: ', adata.n_vars, ' Number of cells: ', adata.n_obs)
original = adata.copy()
iterate_over_genes (adata, noise = args.noise, sigma = args.sigma)
write_noisy_adata (adata)

#print ('ORIGINAL: ', original[original.var_names.isin(['CFH'])].X)
#print ('NOISY: ', adata[adata.var_names.isin(['CFH'])].X)
#adata.write (raw_dir + '/gaussian_0.1_2_MergedSamples.h5ad', compression='gzip')
#%%
'''
samples = select_genes (adata, 'CFH')
noisy_samples = generate_pink_noise (samples)
#noisy_samples = generate_gaussian_noise (noise = 0.1, sigma = 1, mu = 0)

g = plt.figure(2)

plt.plot (samples, color = 'b', label = 'samples')
plt.plot (noisy_samples, color = 'r', label = 'samples + pink noise')

plt.xlabel ('Cell')
plt.ylabel ('Relative Gene Expression')
plt.legend(loc='upper right')
g.savefig ('noisy_gene_expression.png')
g.show ()
'''

#%%
'''
h = plt.figure (3)           
plt.hist (np.hstack (samples), bins='auto', alpha=0.5, density=True, label = 'samples')
#plt.hist (np.hstack (samples), bins='auto', range = [samples.min(), 0])
plt.hist (np.hstack (noisy_samples), bins='auto', alpha=0.5, density=True, color = 'r', label = 'noisy samples')

plt.xlabel ('Relative expression')
#RELATIVE??
plt.ylabel ('Density')
plt.title ('Cluster:' + cluster + ', Gene:' + gene)
plt.legend(loc='upper right')

h.show ()
h.savefig ('hist.png')
#plt.close ()
plt.show ()
'''

#%%
#create H5ad file

