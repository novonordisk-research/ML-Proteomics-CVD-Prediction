import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
import interpret
import pickle
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.perf import ROC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from CVD_dataset_specific_parameters import *
from EBM_CVD_utils import *


def hyperopt_train_test(X_train, X_test, y_train, y_test, params):
    """
    One step in hyperparameter optimization. 
    Trains an EBM model using the specified hyperparameter setting,
    computes auc on the test and returns it.

    args:
        X_train: training set
        X_test: test set
        y_train: training set labels
        y_test: test set labels
        params: parameter configuration

    returns:
        auc on the test set 

    """
    ebm = ExplainableBoostingClassifier(**params)
    ebm.fit(X_train, y_train)
    perf = ROC(ebm).explain_perf(X_test, y_test, name="EBM model")
    auc = round(perf._internal_obj['overall']['auc'], 4)
    return auc


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
        "--num_tries",
        type=int,
        default=40,
        help="Number of configurations to try in each fold."
    )
    parser.add_argument(
        "--num_top_features",
        type=int,
        default=70,
        help="Number of top features from each model to include."
    )
    parser.add_argument(
        "--add_clinical",
        type=bool,
        default=True,
        help="Number of configurations to try in each fold."
    )    
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="ebm/features",
        help="The directory to store the feature files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ebm/results",
        help="The directory to store the hyperparameter optimization results.",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    space = {}
    for k in param_space:
        space[k] = hp.choice(k, param_space[k])

    for fold in range(num_folds):
        print("------------")
        print("Fold", fold)
        print("------------")
        train, _ = LoadData(fold, False, dataset_name, dataset_version)
        selectors = []
        for i in args.group_variables:
            s_index = list(train.columns).index(i)
            selectors.append(s_index)
        clinical = range(clin_start, train.shape[1])
        clinical_col = train.columns[clinical]
        labels = [i for i, col in enumerate(train.columns) if col.endswith("_label")]
        train.dropna(axis=0, how="any", inplace=True, subset=[args.horizon])
        print("Training set size:", train.shape)

        # Split the training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(train, train[args.horizon], test_size=0.2, random_state=args.seed, stratify=train[args.horizon])

        file_prefix = args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)
        ebm_features, subset_features, horizon_features, selector_features = CollectFeatureSets(file_prefix, args.horizon, labels, selectors, args.group_variable_values, X_train)
        if args.add_clinical:
            my_train, my_test = CombineFeatures(ebm_features, subset_features, horizon_features, selector_features, clinical_col, X_train, X_val, args.num_top_features)
        else:
            my_train, my_test = CombineFeatures(ebm_features, subset_features, horizon_features, selector_features, [], X_train, X_val, args.num_top_features)

        def f(params):
            acc = hyperopt_train_test(my_train, my_test, y_train, y_val, params)
            return {'loss': -acc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(f, space, algo=tpe.suggest, max_evals=args.num_tries, trials=trials)

        if args.add_clinical:
            with open(args.results_dir+'/omics_clin_'+args.horizon+'_selfeat_fixed'+str(args.num_top_features)+'_fold_'+str(fold)+'_tuning.pkl', 'wb') as handle:
                pickle.dump(trials, handle)
        else:
            with open(args.results_dir+'/omics_'+args.horizon+'_selfeat_fixed'+str(args.num_top_features)+'_fold_'+str(fold)+'_tuning.pkl', 'wb') as handle:
                pickle.dump(trials, handle)
