#! /usr/bin/env python
"""
"""

import os
import sys
import ccbid
import argparse
import numpy as np
import multiprocessing
from sklearn import metrics
from sklearn import model_selection

# set some paths for running on defaults
ps = os.sep
pbin = os.path.dirname(os.path.realpath(__file__))
pbase = os.path.dirname(pbin)
psup = ps.join([pbase, 'support_files'])

path_training = ps.join([psup, 'training.csv'])
path_reducer = ps.join([psup, 'reducer.pck'])
path_crowns = ps.join([psup, 'species_id.csv'])
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
                        
    parser.add_argument('-c', '--crowns', help='path to the CSV file with crow ID data',
                        default=path_crowns, type=str)
    
    parser.add_argument('-o', '--output', help='path to the output model', required=True, 
                        type=str)
    
    parser.add_argument('-e', '--ecodse', help='flag to run with options used in the ECODSE submission',
                        action='store_true')
    
    parser.add_argument('--reducer', help='path to the data reducer (e.g., the PCA transformer)',
                        default=path_reducer, type=str)
                        
    parser.add_argument('-n', '--n-features', help='the number of features to select after transformation',
                        default=100, type=int)
                        
    parser.add_argument('-m', '--models', help='paths to classification models to use',
                        nargs='+', default=[path_gbc, path_rfc], type=str)
    
    parser.add_argument('-b', '--bands', help='path to a file specifying the bands to use',
                        default=path_bands, type=str)
     
    # arguments to turn on certian flags or set specific parameters
    parser.add_argument('-r', '--remove-outliers', help='flag to remove outliers using PCA',
                        choices=['PCA'], default=None)
                        
    parser.add_argument('-s', '--split', help='method for splitting the train/test data',
                        choices=['crown', 'sample'], default='sample')
    
    parser.add_argument('-t', '--tune', help='flag to perform model tuning instead of using pre-tuned hyperparameters',
                        action='store_true')
                        
    parser.add_argument('-g', '--grids', help='path to the param grid(s) for each model to tune', nargs='+')
                        
    #parser.add_argument('-f', '--feature-selection', help='flag to enable feature selection',
    #                    action='store_true')
                        
    parser.add_argument('--cpus', help='the number of cores to use in model training/fitting',
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
        args.crowns = path_crowns
        args.reducer = path_reducer
        args.n_features = 100
        args.models = [path_gbc, path_rfc]
        args.bands = path_bands
        args.remove_outliers = 'PCA'
        args.split = 'sample'
        args.tune = False
        args.feature_selection = False


# functions for formatted printing
class prnt:
    def __init__(self):
        pass
    
    @staticmethod
    def status(msg):
        print("[ STATUS ] {}".format(msg))
        
    @staticmethod
    def error(msg):
        print("[ ERROR! ] {}".format(msg))
        
    @staticmethod
    def line_break():
        print("[ ------ ]")
        
    @staticmethod
    def model_report(ytrue, ypred, yprob):
        prnt.status("Mean accuracy score: {}".format(metrics.accuracy_score(ytrue, ypred)))
        prnt.status("Mean log loss score: {}".format(metrics.log_loss(ytrue, yprob)))
        
        
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
    
    # parse the logic to make sure everything runs smoothly
    arg_logic(args)
    
    # set the seed for reproducibility (to the year the CCB was founded)
    np.random.seed(1984)
    
    #-----
    # step 1. reading data
    #-----
    
    if args.verbose:
        prnt.line_break()
        prnt.status("Reading input data")
        
    training_id, features = ccbid.read.training_data(args.input)
    crown_id, species_id, species_name = ccbid.read.species_id(args.crowns)
    wavelengths, good_bands = ccbid.read.bands(args.bands)
    species_unique, crowns_unique, crown_labels = \
        ccbid.match_species_ids(training_id, crown_id, species_id)
        
    #-----
    # step 2. outlier removal
    #-----
    
    if args.remove_outliers is not None:
        if args.verbose:
            prnt.status("Removing outliers using {}".format(args.remove_outliers))
            
        # currently only one version of outlier removal
        if args.remove_outliers == 'PCA':
            mask = ccbid.identify_outliers.with_pca(features[:, good_bands])
            
        # subset all data using the mask for future analyses
        features = features[mask, :]
        training_id = training_id[mask]
        crown_labels = crown_labels[mask]
        
        # report on the number of samples removed
        if args.verbose:
            n_removed = mask.shape[0] - mask.sum()
            prnt.status("Removed {} samples".format(n_removed))
        
    #-----
    # step 3: data transformation and resampling
    #-----
    
    # first, transform the data
    if args.reducer is not None:
        if args.verbose:
            prnt.status("Transforming feature data")
            
        features = ccbid.transform.from_path(args.reducer, features[:, good_bands], args.n_features)
        
    # ok, now to do something kinda weird
    # in the original submission, I had resampled the data, then split into train/test sets
    # this is bad practice, since I used the same data to train/calibrate/test the model
    # so we'll keep that consistent here for reproducibility, but we'll do it better for other runs
    
    if args.verbose:
        prnt.status("Splitting train / test data")
        
    if args.ecodse:
        features, crown_labels, training_id = ccbid.resample.uniform(features, crown_labels, other_array=training_id)
        
    # set the label to split the data on samples or crowns
    if args.split == 'sample':
        stratify = crown_labels
    elif args.split == 'crown':
        # ok, so, I think this is actually not quite what I want it to be, but
        # I think it will work pretty well
        stratify = training_id
        
    # we'll split the data into three parts - model training, model calibration, and model test data
    xtrain, xcalib, ytrain, ycalib, strain, scalib = model_selection.train_test_split(\
        features, crown_labels, stratify, test_size=0.5, stratify=stratify)
        
    xctrain, xctest, yctrain, yctest, sctrain, sctest = model_selection.train_test_split(\
        xcalib, ycalib, scalib, test_size=0.5, stratify=scalib)
        
    #-----
    # step 4: model training
    #-----
    
    if args.verbose:
        prnt.line_break()
        prnt.status("Starting model training")
    
    # first, load up the models
    models = []
    for m in args.models:
        models.append(ccbid.read.pck(m))
        
    # then create the ccbid model object
    m = ccbid.model(models=models, average_proba=False)
    
    # tune 'em if you got 'em
    if args.tune:
        # deal with this guy later
        prnt.error("Sorry - not yet implemented!")
        
    # calculate the sample weights then fit the model using the training data
    wtrain = ccbid.get_sample_weights(ytrain)
    m.fit(xtrain, ytrain, sample_weight=wtrain)
    
    # assess the fit on test data
    if args.verbose:
        prnt.status("Assessing model training performance")
        ypred = m.predict(xctest)
        yprob = m.predict_proba(xctest)
        
        for i in range(m.n_models_):
            prnt.status("Model {}".format(i+1))
            print(metrics.classification_report(yctest, ypred[:,i], target_names=species_unique))
            prnt.model_report(yctest, ypred[:,i], yprob[:, :, i])
            
    # next, calibrate prediction probabilities
    m.calibrate(xctrain, yctrain)
    
    # assess the fit on test data
    if args.verbose:
        prnt.status("Asessing model calibration")
        ypred = m.predict(xctest, use_calibrated=True)
        yprob = m.predict_proba(xctest, use_calibrated=True)
        
        for i in range(m.n_models_):
            prnt.status("Model {}".format(i+1))
            print(metrics.classification_report(yctest, ypred[:,i], target_names=species_unique))
            prnt.model_report(yctest, ypred[:,i], yprob[:, :, i])
            
    # finally, re-run the training/calibration using the full data set 
    if args.verbose:
        prnt.status("Fitting final model")
    
    m.fit(np.append(xtrain, xctest, axis=0), np.append(ytrain, yctest))
    m.calibrate(xctrain, yctrain)
    m.average_proba_ = True
    
    # save the ccb model variable
    ccbid.write_pck(args.output, m)
    
    prnt.line_break()
    prnt.status("CCB-ID model training complete!")
    prnt.status("Please see the final output file:")
    prnt.status("  {}".format(args.output))
    prnt.line_break()
    
    # phew


# just run the dang script, will ya?
if __name__ == "__main__":
    main()