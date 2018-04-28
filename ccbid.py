import numpy as _np
import pandas as _pd
import gdal as _gdal
import copy as _copy
import pickle as _pickle
from sklearn import utils as _utils
from sklearn import metrics as _metrics
from sklearn import ensemble as _ensemble
from sklearn import multiclass as _multiclass
from sklearn import calibration as _calibration
from sklearn import preprocessing as _preprocessing
from sklearn import model_selection as _model_selection
from sklearn.decomposition import PCA as _PCA
from matplotlib import pyplot as _plt


__version__ = '0.1.0'


# helper functions to read specifically formatted data
class read:
    """A series of functions for reading CCB-ID formatted data
    """
    def __init__(self):
        pass
    
    @staticmethod
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

    @staticmethod
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
        
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def is_raster(path):
        pass
    
    @staticmethod
    def raster(path):
        pass
    
    @staticmethod
    def pck(path):
        """Reads a python/pickle format data file
        
        Args:
            path - the path to the input pickle file
            
        Returns:
            the object stored in the pickle file
        """
        with open(path, 'r') as f:
            return _pickle.load(f)
    
    @staticmethod
    def neon_bands():
        pass
    
    
def match_species_ids(crown_id, label_id, labels):
    """Matches the crown IDs from the training data with the labels associated with species IDs
    
    Args:
        crown_id - an array of crown IDs from the training data
        label_id - an array of crown IDs associated with the labeled data
        labels   - an array of labels (e.g., species names, species codes)
                   label_id and labels should be of the same size
                 
    Returns:
        [unique_labels, unique_crowns, crown_labels]
        unique_labels - a list of the unique entities from the input labels variable
        unique_crowns - a list of the unique crown entities from the crown_id variable
        crown_labels  - an array with the labels aligned with the original shape of crown_id
    """
    # get the unique labels and crown id's
    unique_labels = _np.unique(labels)
    unique_crowns = _np.unique(crown_id)
    n_labels = len(unique_labels)
    n_crowns = len(unique_crowns)
    
    # set up the output array
    nchar = _np.max([len(label) for label in unique_labels])
    crown_labels = _np.chararray(len(crown_id), itemsize=nchar)
    
    for i in range(n_crowns):
        index_crown = crown_id == unique_crowns[i]
        index_label = label_id == unique_crowns[i]
        crown_labels[index_crown] = labels[index_label]
            
    return [unique_labels, unique_crowns, crown_labels]


#-----
# helper functions to write CCB-ID output data
#-----
class write_predictions:
    def __init__(self):
        pass
    
    @staticmethod
    def to_csv(path, predictions, crown_ids, species_ids):
        pass
    
    @staticmethod
    def to_raster(path, predictions, gdal_params):
        pass
    

def write_pck(path, variable):
    """Writes a python/pickle format data file
    
    Args:
       path     - the path to the output pickle file
       variable - the python variable to write
           
    Returns:
       None
    """
    with open(path, 'wb') as f:
        _pickle.dump(variable, f)

#-----
# helper functions for data transformations, resampling, and model emsembling
#-----
class identify_outliers:
    def __init__(self):
        pass
    
    @staticmethod
    def with_pca(features, n_pcs=20, thresh=3):
        """PCA-based outlier removal function
        
        Args:
            features - the input feature data for finding outliers
            n_pcs    - the number of principal components to look for outliers in
            thresh   - the standard-deviation multiplier for outlier id 
                       (e.g. thresh = 3 means values > 3 stdv from the mean will 
                       be flagged as outliers)
        
        Returns:
            mask     - a boolean array with True for good values, False for outliers
        """
        # create the bool mask where outlier samples will be flagged as False
        mask = _np.repeat(True, features.shape[0])
        
        # set up the pca reducer, then transform the data
        reducer = _PCA(n_components=n_pcs, whiten=True)
        transformed = reducer.fit_transform(features)
        
        # loop through the number of pcs set and flag values outside the threshold
        for i in range(n_pcs):
            outliers = abs(transformed[:, i]) > thresh
            mask[outliers] = False
        
        return mask


class transform:
    def __init__(self):
        pass
    
    @staticmethod
    def pca(features, n_pcs=100):
        """PCA transformation function
        
        Args:
            features - the input feature data to transform
            n_pcs    - the number of components to keep after transformation
            
        Returns:
            an array of PCA-transformed features
        """
        reducer = _PCA(n_components=n_pcs, whiten=True)
        return reducer.fit_transform(features)
        
    @staticmethod
    def from_path(path, features, n_features=None):
        """Transformation using a saved decomposition object
        
        Args:
            path       - the path to the saved decomposition object
            features   - the input feature data to transform
            n_features - the number of features to keep after transformation
        """
        reducer = read.pck(path)
        transformed = reducer.fit_transform(features)
        return transformed[:, 0:n_features]

class resample:
    def __init__(self):
        pass
    
    @staticmethod
    def uniform(features, crown_labels, n_per_class=400, other_array=None):
        """Performs a random uniform resampling of each class to a fixed number of samples
        
        Args:
            features     - the feature data to evenly resample
            crown_labels - labels that correspond to each sample in the feature
                           data and define the unique IDs to resample from
            n_per_class  - the number of samples to select per class
            
        Returns:
            list of [resample_x, resample_y]
            resample_x   - the feature data resampled with shape (n_lables * n_per_class, n_features)
            resample_y   - the class labels from 0 to n_unique_labels with shape (n_lables * n_per_class)
        """
        # get the unique species labels for balanced-class resampling
        unique_labels = _np.unique(crown_labels)
        n_labels = len(unique_labels)
        
        # set up the x and y variables for storing outputs
        resample_x = _np.zeros((n_labels*n_per_class, features.shape[1]))
        resample_y = _np.zeros(n_labels*n_per_class, dtype=_np.uint8)
        
        if other_array is not None:
            if other_array.ndim == 1:
                resample_o = _np.zeros(n_labels*n_per_class)
            else:
                resample_o = _np.zeros((n_labels*n_per_class, other_array.shape[1]))
        
        # loop through and randomly sample each species
        for i in range(n_labels):
            ind_class = _np.where(crown_labels == unique_labels[i])
            ind_randm = _np.random.randint(0, high=ind_class[0].shape[0], size=n_per_class)
            
            # assign the random samples to the balanced class outputs
            resample_x[i*n_per_class:(i+1)*n_per_class] = features[ind_class[0][ind_randm]]
            resample_y[i*n_per_class:(i+1)*n_per_class] = i
            
            if other_array is not None:
                resample_o[i*n_per_class:(i+1)*n_per_class] = other_array[ind_class[0][ind_randm]]
            
        if other_array is None:
            return [resample_x, resample_y]
        else:
            return [resample_x, resample_y, resample_o]
        
    
class crown_ensemble:
    def __init__(self):
        pass
    
    @staticmethod
    def average(predictions, crown_ids):
        pass
    

def get_sample_weights(y):
    """Calculates the balanced sample weights for a set of unique classes
    
    Args:
        y - the input class labels
        
    Returns:
        weights_sample - an array of length (y) with the per-class weights per sample
    """
    # get the unique classes in the array
    classes = _np.unique(y)
    n_classes = len(classes)
    
    # calculate the per-class weights
    weights_class = _utils.class_weight.compute_class_weight('balanced', classes, y)
    
    # create and return an array the same dimensions as the input y vector
    weights_sample = _np.zeros(len(y))
    for i in range(n_classes):
        ind_y = y == classes[i]
        weights_sample[ind_y] = weights_class[i]
        
    return weights_sample

    
#-----
# functions to handle the CCB-ID classification models
#-----
class model:
    def __init__(self, models=None, params=None, calibrator=None, run_calibration=None, 
                 average_proba=True):
        """Creates an object to build the CCB-ID models. Should approximate the functionality
        of the sklearn classifier modules, though not perfectly.
        
        Args:
            models          - a list containing the sklearn models for classification
                              (defaults to using gradient boosting and random forest classifiers)
            params          - a list of parameter values used for each model. This should be a list of length 
                              n_models, with each item containing a dictionary with model-specific parameters
            calibrator      - an sklearn CalibratedClassifier object (or other calibration object)
            run_calibration - a boolean array with True values for models you want to calibrate, 
                              and False values for models that do not require calibration
            average_proba   - flag to report the output probabilities as the average across models
            
        Returns:
            a CCB-ID model object with totally cool functions and attributes.
        """
        # set the base attributes for the model object
        if models is None:
            gbc = _ensemble.GradientBoostingClassifier()
            rfc = _ensemble.RandomForestClassifier()
            self.models_ = [gbc, rfc]
        else:
            # if a single model is passed, convert to a list so it is iterable
            if type(models) is not list:
                models = list(models)
            self.models_ = models
            
        # set an attribute with the number of models
        self.n_models_ = len(self.models_)
            
        # set the model parameters if specified
        if params is not None:
            for i in range(self.n_models_):
                self.models[i].set_params(**params[i])
        
        # set the model calibration function
        if calibrator is None:
            self.calibrator = _calibration.CalibratedClassifierCV(method='sigmoid', cv=3)
        else:
            self.calibrator = calibrator
            
        # set the attribute determining whether to perform calibration on a per-model basis
        if run_calibration is None:
            self.run_calibration_ = _np.repeat(True, self.n_models_)
        else:
            self.run_calibration_ = run_calibration
            
        # set an attribute to hold the final calibrated models
        self.calibrated_models_ = _np.repeat(None, self.n_models_)
        
        # set the flag to average the probability outputs    
        self.average_proba_ = average_proba
        
    def fit(self, x, y, sample_weight=None):
        """Fits each classification model
        
        Args:
            x             - the training features
            y             - the training labels
            sample_weight - the per-sample training weights
            
        Returns:
            None. Updates each item in self.models_
        """
        for i in range(self.n_models_):
            self.models_[i].fit(x, y, sample_weight=sample_weight)
    
    def calibrate(self, x, y, run_calibration=None):
        """Calibrates the probabilities for each classification model
        
        Args:
            x               - the probability calibration features
            y               - the probability calibration labels
            run_calibration - a boolean array with length n_models specifying
                              True for each model to calibrate
                              
        Returns:
            None. Updates each item in self.calibrated_models_
        """
        for i in range(self.n_models_):
            if self.run_calibration_[i]:
                self.calibrator.set_params(base_estimator=self.models_[i])
                self.calibrator.fit(x, y)
                self.calibrated_models_[i] = _copy.copy(self.calibrator)
            else:
                self.calibrated_models_[i] = _copy.copy(self.models_[i])
    
    def tune(self, x, y, param_grids, criterion):
        pass
    
    def predict(self, x, use_calibrated=False):
        """Predict the class labels for given feature data
        
        Args:
            x              - the input features
            use_calibrated - boolean for whether to use the calibrated model for
                             calculating predictions
                             
        Returns:
            output         - an array with the predicted class labels
        """
        for i in range(self.n_models_):
            if use_calibrated:
                predicted = self.calibrated_models_[i].predict(x)
            else:
                predicted = self.models_[i].predict(x)
            
            if i == 0:
                output = _np.expand_dims(predicted, 1)
            else:
                output = _np.append(output, _np.expand_dims(predicted, 1), axis=1)
        
        return output
    
    def predict_proba(self, x, use_calibrated=False, average_proba=None):
        """Predict the probabilities for each class label
        
        Args:
            x              - the input features
            use_calibrated - boolean for whether to use the calibrated model for
                             calculating predictions
            average_proba  - flag to report the output probabilities as the average across models
        """
        if average_proba:
            self.average_proba_ = True
            
        for i in range(self.n_models_):
            if use_calibrated:
                predicted = self.calibrated_models_[i].predict_proba(x)
            else:
                predicted = self.models_[i].predict_proba(x)
            
            if i == 0:
                output = _np.expand_dims(predicted, 2)
            else:
                output = _np.append(output, _np.expand_dims(predicted, 2), axis=2)
        
        # average the final probabilities, if set
        if self.average_proba_:
            return output.mean(axis=2)
        else:
            return output
    
    def set_params(self, params):
        """Sets the parameters for each model
           
        Args:
            params - a list of parameter dictionaries to set for each model
             
        Returns:
            None. Updates each item in self.models_
        """
        for i in range(self.n_models_):
            self.models_[i].set_params(**params[i])