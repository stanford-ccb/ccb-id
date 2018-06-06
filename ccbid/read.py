"""A series of functions for reading CCB-ID formatted data
"""
import numpy as _np
import pandas as _pd
import pickle as _pickle


def bands(path):
    """Reads the wavelengths and good data bands from a csv file
    (based on ccb-id/suport_files/neon-bands.csv')
    
    Args:
        path        - the path to the wavelength file
        
    Returns:
        list of [wavelengths, good_bands]
        wavelengths - an array with the wavelengths associated with each band
        good_bands  - a boolean array for bands to include in analysis (True = good)
    """
    df = _pd.read_csv(path)
    wavelengths = _np.array(df['Wavelength'])
    flag = _np.array(df['Flag'])
    good_bands = flag == 1
    
    return [wavelengths, good_bands]


def species_id(path):
    """Reads the species identities from a csv file 
    (based on ccb-id/suport_files/species_id.csv)
    
    Args:
        path         - the path to the species id file
        
    Returns:
        list of [crown_id, species_id, species_name]
        crown_id     - an array with the unique crown identities
        species_id   - an array with the species ID codes per crown
        species_name - an array with the species names per crown
    """
    df = _pd.read_csv(path)
    crown_id = _np.array(df['crown_id'])
    species_id = _np.array(df['species_id'])
    species_name = _np.array(df['species'])
    
    return [crown_id, species_id, species_name]
    

def genus_id(path):
    """Reads the genus identities from a csv file 
    (based on ccb-id/suport_files/species_id.csv)
    
    Args:
        path       - the path to the genus id file
        
    Returns:
        list of [crown_id, genus_id, genus_name]
        crown_id   - an array with the unique crown identities
        genus_id   - an array with the genus ID codes per crown
        genus_name - an array with the genus names per crown
    """
    df = _pd.read_csv(path)
    crown_id = _np.array(df['crown_id'])
    genus_id = _np.array(df['genus_id'])
    genus_name = _np.array(df['genus'])
    
    return [crown_id, genus_id, genus_name]


def training_data(path):
    """Reads the input training data from a csv file
    (based on ccb-id/support_files/training.csv)
    
    Args:
        path     - the path to the training data csv file
        
    Returns:
        list of [crown_id, features]
        crown_id - an array of per-sample crown IDs
        features - an array of input feature data with shape (n_samples, n_features)
    """
    df = _pd.read_csv(path)
    crown_id = _np.array(df.iloc[:,0])
    features = _np.array(df.iloc[:,1:])
    
    return [crown_id, features]


def is_raster(path):
    """Not yet implemented
    """
    pass


def raster(path):
    """Not yet implemented
    """
    pass


def pck(path):
    """Reads a python/pickle format data file
    
    Args:
        path - the path to the input pickle file
        
    Returns:
        the object stored in the pickle file
    """
    with open(path, 'r') as f:
        return _pickle.load(f)
        