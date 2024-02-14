import pandas as pd
from ukbb_preprocessing.raw_data_preprocessing.raw_data_loader import raw_data_loader

def LoadData(fold, need_test, name, version):
    """
    Load the data.

    args:
        fold: fold number
        need_test: load train and test if True,
                only train otherwise
        name: dataset name
        version: dataset version
        
    returns:
        None 

    """    
    loader = raw_data_loader()
    train_name = f"{name}_{fold}_train"
    train = loader.ws.data.get(name=train_name, version=version)
    if need_test:
        test_name = f"{name}_{fold}_test"
        test = loader.ws.data.get(name=test_name, version=version)
        return pd.read_csv(train.path, index_col=0), pd.read_csv(test.path, index_col=0)
    else:
        return pd.read_csv(train.path, index_col=0), None

def CombineFeatures(ebm_features, subset_features, horizon_features, selector_features, additional_features, orig_X_train, orig_X_test, num_features):
    """
    Combine top num_features features from each feature set.

    args:
        ebm_features: sorted list of features from full EBM model
        subset_features: sorted list of features from EBM model
            trained on subset with overrepresentation of positive cases
        horizon_features: array with sorted lists of features 
            from EBM models trained on other horizons
        selector_features: array with sorted lists of features
            from EBM models trained on subpopulations
        additional_features: list of columns to include in addition
            to selected omics columns (used to add clinical)
        orig_X_train: full training set
        orig_X_test: full test set
        num_features: number of top features from each list to include
        
    returns:
        X_train, X_test - reduced training/test sets 

    """    
    to_keep = set(ebm_features[:num_features])
    to_keep = to_keep | set(subset_features[:num_features])
    for j  in horizon_features:   
        to_keep = to_keep | set(horizon_features[j][:num_features])
    for j in selector_features:
        to_keep = to_keep | set(selector_features[j][:num_features])
    to_keep = to_keep | set(additional_features) 
    to_delete = []
    for i in range(orig_X_train.shape[1]):
        if orig_X_train.columns[i] not in to_keep:
            to_delete.append(i) 
    X_train = orig_X_train.copy(deep=True)
    X_test = orig_X_test.copy(deep=True)
    X_train = X_train.drop(X_train.columns[to_delete], axis=1)
    X_test = X_test.drop(X_test.columns[to_delete], axis=1)
    return X_train, X_test

def CollectFeatureSets(prefix, horizon, labels, selectors, selector_value, X_train):
    """
    Load feature lists from helper models.

    args:
        prefix: prefix to construct file names for each feature file
        horizon: label name
        labels: indices for labels fields in the dataset
        selectors: indices for selector variables
        selector_value: cutoff values for selectors
        X_train: training set
        
    returns:
        ebm_features, subset_features, horizon_features, selector_features
        lists and array of lists of features sorted in (descending) order of importance 

    """    
    ebm_features = pd.read_csv(prefix+".csv", index_col=False)
    del ebm_features["Unnamed: 0"]
    ebm_features = list(ebm_features['0'])
    subset_features = pd.read_csv(prefix+"_subset.csv", index_col=False)
    del subset_features["Unnamed: 0"]
    subset_features = list(subset_features['0'])
    horizon_features = {}
    for i in labels:
        if X_train.columns[i] == horizon:
            continue
        horizon_features[X_train.columns[i]] = pd.read_csv(prefix+"_model_"+ X_train.columns[i]+".csv", index_col=False)
        del horizon_features[X_train.columns[i]]["Unnamed: 0"]
        horizon_features[X_train.columns[i]] = list(horizon_features[X_train.columns[i]]['0'])
    selector_features = {}
    for i in range(len(selectors)):
        var_name = X_train.columns[selectors[i]]
        cutoff_val = str(selector_value[i])
        selector_features[var_name+"_le_"+ cutoff_val] = pd.read_csv(prefix+"_model_"+ var_name +"_le_"+ cutoff_val + ".csv", index_col=False)
        selector_features[var_name+"_gt_"+ cutoff_val] = pd.read_csv(prefix+"_model_"+ var_name +"_gt_"+ cutoff_val + ".csv", index_col=False)
        del selector_features[var_name+"_le_"+ cutoff_val]["Unnamed: 0"]
        del selector_features[var_name+"_gt_"+ cutoff_val]["Unnamed: 0"]
        selector_features[var_name+"_le_"+ cutoff_val] = list(selector_features[var_name+"_le_"+ cutoff_val]['0'])
        selector_features[var_name+"_gt_"+ cutoff_val] = list(selector_features[var_name+"_gt_"+ cutoff_val]['0'])
    return ebm_features, subset_features, horizon_features, selector_features
