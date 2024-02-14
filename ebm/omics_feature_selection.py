import argparse, os
import pandas as pd
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

from CVD_dataset_specific_parameters import *
from EBM_CVD_utils import *
  
def SortedFeatureList(model):
    """
    Retrieve feature importance data from the model,
    sort features in descending order of importance and 
    return the list.

    args:
        model: EBM model
        
    returns:
        list of features sorted by importance 

    """
    my_feature_set = set()
    my_feature_list = []

    model_global = model.explain_global(name="model")
    fnames = model_global._internal_obj["overall"]["names"] 
    fscores = model_global._internal_obj["overall"]["scores"]
    fscores, fnames = zip(*sorted(zip(fscores, fnames), reverse=True))

    for f in fnames:
        if f.find(" & ") > -1:
            f1, f2 = f.split(" & ")
            if f1 not in my_feature_set:
                my_feature_list.append(f1)
            if f2 not in my_feature_set:
                my_feature_list.append(f2)
            my_feature_set.add(f1)
            my_feature_set.add(f2)
        else:
            if f not in my_feature_set:
                my_feature_list.append(f)
            my_feature_set.add(f)
    
    return my_feature_list

def TrainAndSaveFeatures(X_train, y_train, f_file):
    """
    Train an EBM model and save the list of features.

    args:
        X_train: training set
        y_train: training set labels
        f_file: file name to store features under
        
    returns:
        None 

    """
    ebm = ExplainableBoostingClassifier(interactions=10)
    ebm.fit(X_train, y_train)    
    ebm_features = SortedFeatureList(ebm)
    pd.Series(ebm_features).to_csv(f_file)


def ConstructSubset(train, horizon, seed, frac=0.2):
    """
    Construct a subset with overrepresentation of positive samples.

    args:
        train: training set
        horizon: label
        seed: for reproducibility
        frac: fraction for downsampling the negative class
        
    returns:
        None 

    """
    # down sample negative samples to frac
    subset_train = train[train[horizon] == 0].sample(frac=frac, replace=False, random_state=seed) 
    subset_train = pd.concat([subset_train, train[train[horizon] > 0]], ignore_index=True)
    return subset_train
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=str,
        default= "ascvd_10yr_label",
        help="The time horizon for the prediction model.",
    )
    parser.add_argument(
        "--group_variables",
        type=str,
        nargs="+",
        default=["p31", "p21022", "p21001_i0"],
        help="The list of variables used to split the data for feature selection."
    )
    parser.add_argument(
        "--group_variable_values",
        type=float,
        nargs="+",
        default=[0.0, 58.0, 26.6],
        help="The list of cutoff values for group variables."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="ebm/features",
        help="The directory to store the feature files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.feature_dir):
        os.makedirs(args.feature_dir)

    for fold in range(num_folds):
        print("------------")
        print("Fold", fold)
        print("------------") 
        train, _ = LoadData(fold, False, dataset_name, dataset_version)
        hposition = { col: i for i, col in enumerate(train.columns) if col.endswith("_label")}
        labels = [i for i, col in enumerate(train.columns) if col.endswith("_label")]
        selectors = []
        for i in args.group_variables:
            s_index = list(train.columns).index(i)
            selectors.append(s_index)
        columns_to_keep = train.columns[labels].append(train.columns[omix]).append(train.columns[selectors])
        train = train[columns_to_keep]
        train.dropna(axis=0, how="any", inplace=True, subset=[args.horizon])
        print("Training set size:", train.shape)

        # drop current horizon label
        y_train = train[args.horizon]
        selection = list(range(0, hposition[args.horizon])) + list(range(hposition[args.horizon]+1,train.shape[1]))
        X_train = train.iloc[:,selection]

        # adjust indices
        selectors = list(range(-len(args.group_variables), 0))
        labels = labels [:-1]
        
        # Train full EBM
        full_X_train = X_train.copy(deep=True)
        full_X_train = full_X_train.drop(full_X_train.columns[labels+selectors], axis=1)
        TrainAndSaveFeatures(full_X_train, y_train, args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)+".csv")
    
        # Train subset EBM
        subset_train = ConstructSubset(train, args.horizon, args.seed)
        subset_y_train = subset_train[args.horizon]
        subset_X_train = subset_train.iloc[:,selection]
        subset_X_train = subset_X_train.drop(subset_X_train.columns[labels+selectors], axis=1)
        TrainAndSaveFeatures(subset_X_train, subset_y_train, args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)+"_subset.csv")

        # Train models for other horizons
        for i in range(len(labels)):
            labels_to_delete = [0, 1, 2]
            labels_to_delete.pop(i)
            to_delete = labels_to_delete + selectors
            my_X_train = X_train.copy(deep=True) 
            my_X_train = my_X_train.drop(my_X_train.columns[to_delete], axis=1)
            my_X_train.dropna(axis=0, how="any", inplace=True, subset=[X_train.columns[i]])
            my_y_train = my_X_train.iloc[:,0]
            my_X_train = my_X_train.iloc[:,1:]
            TrainAndSaveFeatures(my_X_train, my_y_train, args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)+"_model_"+ X_train.columns[i]+".csv")

        # Train models for selectors
        selector_value = args.group_variable_values
        for i in range(len(selectors)):
            selectors_to_delete = [-3, -2, -1]
            selectors_to_delete.pop(i)
            to_delete = labels + selectors_to_delete
            my_train = X_train.copy(deep=True)
            my_train = my_train.drop(my_train.columns[to_delete], axis=1)
            my_train = pd.merge(my_train, y_train, how="inner", left_index=True, right_index=True)
            my_train1 = my_train[my_train[my_train.columns[-2]] <= selector_value[i]]
            my_train2 = my_train[my_train[my_train.columns[-2]] > selector_value[i]]
            my_X_train1 = my_train1.iloc[:,0:-2]
            my_y_train1 = my_train1.iloc[:,-1]
            my_X_train2 = my_train2.iloc[:,0:-2]
            my_y_train2 = my_train2.iloc[:,-1]
            TrainAndSaveFeatures(my_X_train1, my_y_train1, args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)+"_model_"+ my_train.columns[-2]+"_le_"+ str(selector_value[i]) + ".csv")
            TrainAndSaveFeatures(my_X_train2, my_y_train2, args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)+"_model_"+ my_train.columns[-2]+"_gt_"+ str(selector_value[i]) + ".csv")
