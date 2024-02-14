# Script to load AutoML model and use it for inference on test data.

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from data_utils.io_utils import load_pandas_df

def str2bool(x):  # pylint: disable=invalid-name
        return x.lower() == 'true'

def get_model_pred(ml_client, args):

    # Load up test data
    test_df = load_pandas_df(ml_client, args.asset_name_test, args.asset_version_test)
    print("Loaded test data.")

    if args.label_col:
        test_df = test_df.dropna(subset=[args.label_col])
        print(f"After dropping null labels, # Test points = {len(test_df)}")

    if args.feat_select_csv:
        fs_df = pd.read_csv(args.feat_select_csv)
        column_names = fs_df.columns.values.tolist()
        feat_cols = column_names[args.feat_col_start:] # Obtain feat columns. feat_col_start = 1 for fs_df.
        print(f"Sub-selected features. # Features = {len(feat_cols)}")
    else:
        column_names = test_df.columns.values.tolist()
        feat_cols = column_names[args.feat_col_start:] # Obtain feat columns

    test_feat = test_df[feat_cols] 

    # Load up AutoML model from .pkl
    # AutoML model .pkl already contains the transformation required on data and used during train time.
    path_to_model_pkl = os.path.join(args.model_dir, "model.pkl")
    model = joblib.load(path_to_model_pkl)
    infer_scores = model.predict_proba(test_feat)

    y_proba = infer_scores[:, 1]
    print("Computed test scores.")

    pred_data = {"IID": test_df["IID"].tolist(), "y_pred": np.zeros(len(y_proba)), "y_score": y_proba}
    pred_df = pd.DataFrame(pred_data)
    
    os.makedirs(args.pred_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(args.pred_dir, args.asset_name_test+".csv"), index=False)
    print("Written predictions to output file.")

    if args.label_col:
        y_true = test_df[args.label_col].to_numpy()

        # Compute test metrics
        test_auc = roc_auc_score(y_true=y_true, y_score=y_proba)
        print(f"Test AUC = {test_auc}")

def main():
     
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument("--asset_name_test", type=str, help="Name of registered data asset for test. Eg. npx_clin_ascvd_0_test")
    parser.add_argument("--asset_version_test", type=str, help="Version of registered data asset. Eg. 3")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--pred_dir", type=str, help="Path to prediction directory directory")
    parser.add_argument("--label_col", type=str, help="Label column name")
    parser.add_argument("--feat_select_csv", type=str, help="Path to a csv file containing sub-features.")
    parser.add_argument("--feat_col_start", type=int, help="Start of feature column index")
    args = parser.parse_args()
    print(args)

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential()

    # Read the config from the current directory and get 
    ml_client = MLClient.from_config(credential=credential)
    print("Created ML Client.")

    get_model_pred(ml_client, args)



if __name__ == "__main__":

    main()
