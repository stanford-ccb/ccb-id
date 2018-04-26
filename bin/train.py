#! /usr/bin/env python
"""
"""

import os
import sys
#import ccbid
import argparse
import numpy as np
import pandas as pd
import multiprocessing

# set some paths for running on defaults
ps = os.sep
pbin = os.path.dirname(os.path.realpath(__file__))
pbase = os.path.dirname(pbin)
psup = ps.join([pbase, 'support_files'])

path_training = ps.join([psup, 'training.csv'])
path_reducer = ps.join([psup, 'reducer.pck'])
path_bands = ps.join([psup, 'neon-bands.csv'])
path_rfc = ps.join([psup, 'rfc.pck'])
path_gbc = ps.join([psup, 'gbc.pck'])

# get the number of cpus to use as defaults
cpus = multiprocessing.cpu_count()

# set up the argument parser to read command line inputs
def parse_args():
    """
    """
    
    # create the argument parser
    parser = argparse.ArgumentParser(description='Train and save a CCB-ID species classification model.')
    
    # set up the arguments for dealing with file i/o
    parser.add_argument('-i', '--input', help='path to an input CSV file for model training',
                        default=path_training, type=str)
    
    parser.add_argument('-o', '--output', help='path to the output model', required=True, 
                        type=str)
    
    parser.add_argument('-r', '--reducer', help='path to the data reducer (e.g., the PCA transformer)',
                        default=path_reducer, type=str)
                        
    parser.add_argument('-m', '--models', help='paths to classification models to use',
                        nargs='+', default=[path_gbc, path_rfc], type=str)
    
    parser.add_argument('-b', '--bands', help='path to a file specifying the bands to use',
                        default=path_bands, type=str)
     
    # arguments to turn on certian flags or set specific parameters
    parser.add_argument('-s', '--split', help='method for splitting the train/test data in model training',
                        choices=['crown', 'sample'], default='sample')
    
    parser.add_argument('-e', '--ecodse', help='flag to run with options used in the ECODSE submission',
                        action='store_true')
                        
    parser.add_argument('-t', '--tune', help='flag to perform model tuning instead of predetermined hyperparameters',
                        action='store_true')
                        
    parser.add_argument('-g', '--grids', help='path to the param grid(s) for each model to tune', nargs='+')
                        
    parser.add_argument('-f', '--feature-selection', help='flag to enable feature selection',
                        action='store_true')
    
    parser.add_argument('-n', '--n-features', help='set an int for the number of features to select, or a 0-1 float for pct of feature importance scores',
                        default=20, type=int)
                        
    parser.add_argument('-c', '--cpus', help='the number of cores to use in model training/fitting',
                        default=cpus-1, type=int)

    parser.add_argument('-v', '--verbose', help='turn on verbose mode. Not that I have that much to say..', 
                        action='store_true')
    
    # set up the arguments for dealing with 
    return parser.parse_args(sys.argv[1:])


# set up the logic to parse command line arguments and ensure consistent logic
def arg_logic(args):
    """Parses the command line arguments to ensure consistency prior to running the main script
    
    Args:
        args - the arguments returned from the argparse object
        
    Returns:
        None. This function updates the args object
    """
    
    # if the ECODSE flag is set, override whatever is set at the command line
    if args.ecodse:
        args.input = path_training
        args.reducer = path_reducer
        args.models = [path_gbc, path_rfc]
        args.bands = path_bands
        args.split = 'sample'
        args.tune = False
        args.feature_selection = False
    
    return None


# set up the main script function
def main():
    """The main function for train.py
    
    Args:
        None - just let it fly
        
    Returns:
        None - this runs the dang script
    """
    
    # first read the command line arguments
    args = parse_args()
    print(args)
    
    # parse the logic to make sure everything runs smoothly
    arg_logic(args)
    print(args)
    
    # set the seed for reproducibility (to the year the CCB was founded)
    np.random.seed(1984)


# run the dang script already
if __name__ == "__main__":
    main()