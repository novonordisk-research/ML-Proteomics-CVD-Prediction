import argparse, os
import pandas as pd
import math
import interpret
import pickle
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.perf import ROC

from CVD_dataset_specific_parameters import *
from ebm_utils import *

def FindBestParams(folds, prefix, fraction_tries_used):
    """
    Use results of hyperparameter training run to find 
    optimal configuration settings.  To avoid overfitting,
    average parameters from top few runs at each fold,
    then average them over folds.

    args:
        folds: folds that the hyperparameter optimization ran on
        prefix: file name prefix specifying model type
                (horizon, feature selection used or not,
                 data selection: omics+clinical, omics only,
                 clinical only)
        fraction_tries_used: fraction of configurations tried
               (per fold) to use in computing final settings

    returns:
        parameter settings to use 

    """
    fold_data = {}
    params = {}
    columns = ['loss']
    for v in param_space:
        columns.append(v)
    for fold in folds:
        fold_data[fold] = pd.DataFrame(columns=columns)
        with open(prefix+str(fold)+'_tuning.pkl', 'rb') as handle:
            trials = pickle.load(handle)
        for t in trials:
            trial = {}
            trial['loss'] = t['result']['loss']
            for v in t['misc']['vals']:
                param_index = t['misc']['vals'][v][0]
                trial[v] = param_index
            trial_df = pd.DataFrame([trial], columns=columns)
            fold_data[fold] = pd.concat([fold_data[fold], trial_df])
        fold_data[fold].sort_values(by=["loss"], inplace=True)

    top = int(math.ceil(fraction_tries_used*len(fold_data[folds[0]])))
    for p in param_space:
        param_index = 0
        for fold in folds:
            param_index += fold_data[fold].iloc[:top,:][p].mean()
        param_index /= len(folds) 
        param_index = int(param_index)
        params[p] = param_space[p][param_index]
    return params


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
        "--fraction_tries_used",
        type=float,
        default=0.1,
        help="Number of top features from each model to include."
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
        "--model_dir",
        type=str,
        default="ebm/models",
        help="The directory to store the model files.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="ebm/predictions",
        help="The directory to store the prediction files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)            
    if not args.add_clinical:
        if not os.path.isdir(args.pred_dir+"/omics/"+args.horizon):        
            os.makedirs(args.pred_dir+"/omics/"+args.horizon)
    else:
        if not os.path.isdir(args.pred_dir+"/"+args.horizon):
            os.makedirs(args.pred_dir+"/"+args.horizon)


    for fold in range(num_folds):
        print("------------")
        print("Fold", fold)
        print("------------")

        if args.add_clinical:
            prefix = args.results_dir+'/omics_clin_'+args.horizon+'_selfeat_fixed'+str(args.num_top_features)+'_fold_'
        else:
            prefix = args.results_dir+'/omics_'+args.horizon+'_selfeat_fixed'+str(args.num_top_features)+'_fold_'
        params = FindBestParams([fold], prefix, args.fraction_tries_used)
        print(f"Using parameters: {params}")
        
        train, test = LoadData(fold, True, dataset_name, dataset_version)
        selectors = []
        for i in args.group_variables:
            s_index = list(train.columns).index(i)
            selectors.append(s_index)         
        labels = [i for i, col in enumerate(train.columns) if col.endswith("_label")]
        clinical = range(clin_start, train.shape[1])
        clinical_col = train.columns[clinical]
        train.dropna(axis=0, how="any", inplace=True, subset=[args.horizon])
        test.dropna(axis=0, how="any", inplace=True, subset=[args.horizon])
        print("Training set size:", train.shape)
        print("Test set size:", test.shape)
        y_train = train[args.horizon]
        y_test = test[args.horizon]
        file_prefix = args.feature_dir+"/ebm_all_features_sorted_"+args.horizon+"_fold_"+str(fold)
        ebm_features, subset_features, horizon_features, selector_features = CollectFeatureSets(file_prefix, args.horizon, labels, selectors, args.group_variable_values, train)
        if args.add_clinical:
            X_train, X_test = CombineFeatures(ebm_features, subset_features, horizon_features, selector_features, clinical_col, train, test, args.num_top_features)
        else:
            X_train, X_test = CombineFeatures(ebm_features, subset_features, horizon_features, selector_features, [], train, test, args.num_top_features)

        params['random_state'] = args.seed
        final_ebm = ExplainableBoostingClassifier(**params)
        final_ebm.fit(X_train, y_train)
        final_ebm_perf = ROC(final_ebm).explain_perf(X_test, y_test, name="Final EBM model")
        auc = round(final_ebm_perf._internal_obj['overall']['auc'], 4)
        print("final_ebm", X_train.shape[1], auc)
    
        if args.add_clinical:
            with open(args.model_dir+'/ebm_model_omics_clin_'+args.horizon+'_selfeatfixed'+str(args.num_top_features)+'_fold_'+str(fold)+'.pkl', 'wb') as handle:
                pickle.dump(final_ebm, handle)
        else:
            with open(args.model_dir+'/ebm_model_omics_'+args.horizon+'_selfeatfixed'+str(args.num_top_features)+'_fold_'+str(fold)+'.pkl', 'wb') as handle:
                pickle.dump(final_ebm, handle)      

        y_pred = final_ebm.predict(X_test)
        y_score = final_ebm.predict_proba(X_test)[:,1]
        df = pd.DataFrame(list(zip(X_test.index, y_pred, y_score)), columns=['IID', 'y_pred', 'y_score'])
        if args.add_clinical:
            df.to_csv(f'{args.pred_dir}/{args.horizon}/{dataset_name}_{fold}_test.csv', index=False)
        else:
            df.to_csv(f'{args.pred_dir}/omics/{args.horizon}/{dataset_name}_{fold}_test.csv', index=False)
