# Script to compute SHAP values from trained models. Uses the train data for SHAP valuea and global feature importance computation

import os
import joblib
import argparse
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from data_utils.io_utils import load_pandas_df


def shap_driver(ml_client, args):

    # Load up train data. Requires azureml-fsspec apart from azure-ai-ml.
    train_df = load_pandas_df(ml_client, args.asset_name_train, args.asset_version_train)
    print("Loaded train data.")

    column_names = train_df.columns.values.tolist()
    feat_cols = column_names[args.feat_col_start:] # Obtain feat columns

    train_feat = train_df[feat_cols] 

    path_to_model_pkl = os.path.join(args.model_dir, "model.pkl")
    # Requires azureml-automl-runtime
    # Use scikit-learn==0.22.1 for proper loading of model pickle (even if incompatible with azureml-automl-runtime version)
    model = joblib.load(path_to_model_pkl)
    
    if hasattr(model.steps[0][1], "get_model"):
        scaler = model.steps[0][1].get_model()  # MinMaxScaler has no attribute get_model().
    else:
        scaler = model.steps[0][1]
    lgb_clf = model.steps[1][1].get_model()

    print("Loaded normalizer and model.")
    X_train_scaled = scaler.transform(train_feat)
    
    explainer = shap.TreeExplainer(lgb_clf)
    shap_values = explainer.shap_values(X_train_scaled)
    # shap_values: List[class-0-shap, class-1-shap], same magnitude with alternate signs.
    print("Computed SHAP values.")

    assert len(feat_cols) == shap_values[1].shape[-1], "ERROR: Number of features NOT EQUAL."

    global_feat_importance = np.mean(np.abs(shap_values[1]), axis=0) # Consume only the class-1 shap values.
    shap.summary_plot(shap_values[1], X_train_scaled, feature_names=feat_cols, plot_type="bar", show=False)
    os.makedirs(args.out_dir, exist_ok=True)
    png_file_path = os.path.join(args.out_dir, f"shap_{args.asset_name_train}.png")
    plt.savefig(png_file_path, format='png')

    feat_imp_df = pd.DataFrame(data={"feature": feat_cols, "importance": global_feat_importance})
    sorted_feat_imp_df = feat_imp_df.sort_values(by="importance", ascending=False)
    print("Top 20 features")
    print(sorted_feat_imp_df.head(20))

    # Save feature importance to file
    feat_imp_file_path = os.path.join(args.out_dir, f"shap_{args.asset_name_train}.csv")
    sorted_feat_imp_df.to_csv(feat_imp_file_path, index=False)

def main():

    parser = argparse.ArgumentParser(description="SHAP computation")
    parser.add_argument("--asset_name_train", type=str, help="Name of registered data asset for train. Eg. npx_clin_ascvd_0_train")
    parser.add_argument("--asset_version_train", type=str, help="Version of registered data asset. Eg. 3")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    parser.add_argument("--out_dir", type=str, help="Path to output directory")
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

    shap_driver(ml_client, args)


if __name__ == "__main__":

    main()    
