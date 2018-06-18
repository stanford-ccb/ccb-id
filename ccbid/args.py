"""A consistent set of argument parser objects to share among scripts
"""
import os as _os
import argparse as _argparse
from ccbid._core import _path as ppkg
import multiprocessing as _multiprocessing


ps = _os.sep
pbase = _os.path.dirname(ppkg)
psup = ps.join([pbase, 'support_files'])

path_training = ps.join([psup, 'training.csv'])
path_reducer = ps.join([psup, 'reducer.pck'])
path_crowns = ps.join([psup, 'species_id.csv'])
path_bands = ps.join([psup, 'neon-bands.csv'])
path_rfc = ps.join([psup, 'rfc.pck'])
path_gbc = ps.join([psup, 'gbc.pck'])

# get the number of cpus to use as defaults
n_cpus = _multiprocessing.cpu_count()


# create the argument parser
def create_parser(**kwargs):
    parser = _argparse.ArgumentParser(**kwargs)
    return parser
    

# set up the arguments for dealing with file i/o
def input(parser):
    parser.add_argument('-i', '--input', help='path to an input CSV file for model training',
                        default=path_training, type=str)
    return parser
    

def crowns(parser):                        
    parser.add_argument('-c', '--crowns', help='path to the CSV file with crown ID data',
                        default=path_crowns, type=str)
    return parser


def output(parser):    
    parser.add_argument('-o', '--output', help='path to the output model', required=True, 
                        type=str)
    return parser
    

def ecodse(parser):    
    parser.add_argument('-e', '--ecodse', help='flag to run with options used in the ECODSE submission',
                        action='store_true')
    return parser
    

def reducer(parser):    
    parser.add_argument('--reducer', help='path to the data reducer (e.g., the PCA transformer)',
                        default=path_reducer, type=str)
    return parser
    

def n_features(parser):
    parser.add_argument('-n', '--n-features', help='the number of features to select after transformation',
                        default=100, type=int)
    return parser
    

def models(parser):
    parser.add_argument('-m', '--models', help='paths to classification models to use',
                        nargs='+', default=[path_gbc, path_rfc], type=str)
    return parser
    

def bands(parser):    
    parser.add_argument('-b', '--bands', help='path to a file specifying the bands to use',
                        default=path_bands, type=str)
    return parser
    

# arguments to turn on certian flags or set specific parameters
def remove_outliers(parser):
    parser.add_argument('-r', '--remove-outliers', help='flag to remove outliers using PCA',
                        choices=['PCA'], default=None)
    return parser

def split(parser):
    parser.add_argument('-s', '--split', help='method for splitting the train/test data',
                        choices=['crown', 'sample'], default='sample')
    return parser

def tune(parser):
    parser.add_argument('-t', '--tune', help='flag to perform model tuning instead of using pre-tuned hyperparameters',
                        action='store_true')
    return parser


def grids(parser):
    parser.add_argument('-g', '--grids', help='path to the param grid(s) for each model to tune', nargs='+')
    return parser
    

def feature_selection(parser):
    parser.add_argument('-f', '--feature-selection', help='flag to enable feature selection',
                        action='store_true')
    return parser
    

def cpus(parser):
    parser.add_argument('--cpus', help='the number of cores to use in model training/fitting',
                        default=n_cpus-1, type=int)
    return parser

def verbose(parser):
    parser.add_argument('-v', '--verbose', help='turn on verbose mode. Not that I have that much to say..', 
                        action='store_true')