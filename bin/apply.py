#! /usr/bin/env python
"""
"""

import os
import sys
import ccbid
import argparse
import numpy as np
import multiprocessing
from ccbid import args
from ccbid import prnt
from sklearn import metrics
from sklearn import model_selection

"""steps
-read model
-read input spectra (from csv or from image)
-apply model to samples
-average by crown
-output predictions (as csv or image)

things to add to model object
-reducer
-bands
-species names/IDs

create ccbid model input argument
"""


# set up the argument parser to read command line inputs
def parse_args():
    """Function to read CCB-ID command line arguments
    
    Args:
        None - reads from sys.argv
        
    Returns:
        an argparse object
    """
    
    # create the argument parser
    parser = args.create_parser(description='Train and save a CCB-ID species classification model.')
    #parser = argparse.ArgumentParser(description='Train and save a CCB-ID species classification model.')
    
    # set up the arguments for dealing with file i/o
    args.input(parser)
    args.mask(parser) # write this, option to pass a good-data mask
    args.output(parser)
    args.ecodse(parser)
    args.ccbid_model(parser)

    # arguments to turn on certian flags or set specific parameters
    args.remove_outliers(parser)
    args.aggregate(parser) # write this, option to aggregate the data to other scales (e.g., crown scale)
    args.labels(parser) # write this, option to read image with crown labels
    #args.feature_selection(parser)
    args.cpus(parser) # maybe add function to model object to update the n_cpus in each model
    args.verbose(parser)

    # parse the inputs from sys.argv
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
        args.input = args.path_training
        args.remove_outliers = False
        #args.feature_selection = False


# set up the main script function
def main():
    """The main function for ccbid apply
    
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
    
    # first read the model data
    model = ccbid.read.pck(args.ccbid_model)
    
    # set up a dummy variable to determine if data should be output on a per-crown or per-=pixel basis
    
    # then read the feature data, which may come as a raster or a csv
    if ccbid.read.is_raster(args.input):
        features = ccbid.read.raster(args.input)
        # work on this later
        
    if ccbid.read.is_csv(args.input):
        testing_id, features = ccbid.read.training_data(args.input)
        crowns_unique = np.unique(testing_id)
        n_crowns = len(crowns_unique)
        
    #-----
    # step 2. outlier removal
    #-----
    
    if args.remove_outliers is not None:
        if args.verbose:
            prnt.status("Removing outliers using {}".format(args.remove_outliers))
            
        # currently only one version of outlier removal
        if args.remove_outliers == 'PCA':
            mask = ccbid.outliers.with_pca(features[:, model.good_bands])
            
        # subset all data using the mask for future analyses
        features = features[mask, :]
        testing_id = training_id[mask]
        
        # report on the number of samples removed
        if args.verbose:
            n_removed = mask.shape[0] - mask.sum()
            prnt.status("Removed {} samples".format(n_removed))
        
    #-----
    # step 3: data transformation
    #-----
    
    if model.reducer is not None:
        if args.verbose:
            prnt.status("Transforming feature data")
        
        features = model.reducer.transform(features[:, model.good_bands])
        
    #-----
    # step 4: applying the model
    #-----
    
    if args.verbose:
        prnt.line_break()
        prnt.status("Applying CCBID model to input features")
    
    pred = model.predict(features)
    prob = model.predict_proba(features, average_proba=True)
        
    # ensemble the pixels to the crown scale
    if args.aggregate:
        # do it differently for csv vs raster
        if ccbid.read.is_csv(args.input):
            # create the output array to store the results
            n_species = prob.shape[1]
            output_pr = np.zeros(n_crowns * n_species)
            
            # loop through each crown, calculate the average probability per crown, and write it to the array
            # !!! should probably move this to a ccbid function
            for i in range(n_crowns):
                crown_index = testing_id == crowns_unique[i]
                output_pr[i * n_species:(i+1) * n_species] = prob[crown_index].mean(axis=0)
                
            # create the crown id labels (also, create the model.labels property)
            cr_labels = np.repeat(crowns_unique, n_species)
            sp_labels = np.repeat(model.labels, n_crowns).reshape(n_crowns, n_species).flatten(order='F')
            
            # add everything to a pandas dataframe and save the result
            df = pd.DataFrame.from_items((('crown', cr_labels), ('species', sp_labels), 
                                          ('probability', output_pr)))
            df.to_csv(args.output, index=False)
            
        if ccbid.read.is_raster(args.input):
            # get the crown IDs from a separate raster
            try:
                testing_id = ccbid.read.raster(args.labels)
            except:
                prnt.error("Unable to read label file: {}".format(args.labels))
                prnt.error("Check the file path or run without the --aggregate option in order to obtain pixel-scale predictions")
    
    # or, output the raw predictions if not aggregating
    else:
        # do it differently for csv vs raster
        if ccbid.read.is_csv(args.input):
            # write out results as a pandas dataframe
            df = pd.DataFrame(prob, columns=model.labels)
            df.to_csv(args.output, index=False)
    
    prnt.line_break()
    prnt.status("CCB-ID model application complete!")
    prnt.status("Please see the final output file:")
    prnt.status("  {}".format(args.output))
    prnt.line_break()
    
    # phew


# just run the dang script, will ya?
if __name__ == "__main__":
    main()