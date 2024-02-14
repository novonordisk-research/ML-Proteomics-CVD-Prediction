# Script to load registered data, load splits and save splits to disk as MLTable folder.

import argparse
import os
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from data_utils.io_utils import load_pandas_df


def write_mltable_file(file_path, csv_name):

    with open(file_path, "w") as fptr:
        fptr.write(f"paths:\n- file: {csv_name}\ntransformations:\n- read_delimited:\n    delimiter: ','\n    empty_as_string: false\n    encoding: ascii\n    header: all_files_same_headers\n    include_path_column: false\n    infer_column_types: true\ntype: mltable")


def register_mltable(ml_client, out_dir, description = "MLTable", name = "asset_new"):

    # Create DataAsset from authored mltable.
    tbl_asset = Data(
                    path=out_dir,
                    type=AssetTypes.MLTABLE,
                    description=description,
                    name=name
                    )

    ml_client.data.create_or_update(tbl_asset)
    print(f"Registered Data Asset {name}.")
    

def load_and_filter_data(ml_client, asset_name, asset_version, args):
    
    # Load up registered data as pandas dataframe
    data_df = load_pandas_df(ml_client, asset_name, asset_version)

    # Filter by gender
    if args.filter_col:
        filtered_df = data_df[data_df[args.filter_col] == args.filter_val]
        print(f"Filtered based on column {args.filter_col} with value {args.filter_val}.")
    else:
        filtered_df = data_df
        print("No filtering of any column.")

    # Drop unnecessary columns, such as additional labels, IIDs and gender.
    drop_column_list = args.drop_columns.split(",")
    feat_label_df = filtered_df.drop(drop_column_list, axis=1)

    print(f"Dropped columns {args.drop_columns}.")

    # Drop NaN values
    feat_label_df = feat_label_df.dropna()

    print(f"After removing NaNs and optional filtering, final data size = {len(feat_label_df)}")

    # Optional feature selection.
    if args.feat_sel_path:

        if args.feat_sel_algo == "ebm":

            feat_imp_df = pd.read_csv(args.feat_sel_path, index_col=False).squeeze("columns")
            print("Loaded features from EBM.")

            top_feat_names =  feat_imp_df.tolist()
            print(f"Total # Features = {len(top_feat_names)}")
        
        elif args.feat_sel_algo == "shap":

            feat_imp_df = pd.read_csv(args.feat_sel_path)
            print("Loaded feature importance data.")

            # SHAP csv file has the fields "feature" (feature names) and "importance" (feature importances)

            # Sort features based on feature importance.
            sorted_feat_imp_df = feat_imp_df.sort_values(by=["importance"], ascending=False)

            if args.frac_feat > 1.0:
                num_feat = int(args.frac_feat)
            else:
                num_feat = int(args.frac_feat*(len(feat_label_df.columns) - 1))

            sorted_feat_imp_df = sorted_feat_imp_df.head(num_feat)
            print(f"Selected top {num_feat} based on feature importance.")

            top_feat_names =  sorted_feat_imp_df["feature"].tolist()
        
        elif args.feat_sel_algo == "mimic-lgbm":

            feat_imp_df = pd.read_csv(args.feat_sel_path)
            print("Loaded feature importance data.")

            # AutoML feature importance csv file has the fields "Category" (feature names) and "All data" (feature importances)

            # Sort features based on feature importance.
            sorted_feat_imp_df = feat_imp_df.sort_values(by=["All data"], ascending=False)

            num_feat = int(args.frac_feat*(len(feat_label_df.columns) - 1))
            sorted_feat_imp_df = sorted_feat_imp_df.head(num_feat)
            print(f"Selected top {num_feat} based on feature importance.")

            top_feat_names =  sorted_feat_imp_df["Category"].tolist()

        else:
            raise NotImplementedError

        feat_label_df = feat_label_df[[args.label_col] + top_feat_names]

        print(f"Shape of feature label array: {feat_label_df.shape}")
    
    if args.out_dir:

        os.makedirs(args.out_dir, exist_ok=True)
        data_dir = os.path.join(args.out_dir, asset_name)
        os.makedirs(data_dir, exist_ok=True)        
               
        feat_label_df.to_csv(path_or_buf=os.path.join(data_dir, f"{asset_name}.csv"), index = False)
        write_mltable_file(file_path=os.path.join(data_dir,"MLTable"), csv_name=f"{asset_name}.csv")

        print("Successfully created MLTable directories and written csv files.")  

        # Register MLTable to workspace.
        if args.register_name_prefix:
            
            register_mltable(ml_client, data_dir, description=f"MLTable {args.register_name_prefix} from {asset_name}", name=f"{args.register_name_prefix}_{asset_name}")

            print("Successfully registered MLTable data assets to workspace.")


def main():

    parser = argparse.ArgumentParser(description="MLTable Register")
    parser.add_argument("--asset_name_train", type=str, help="Name of registered data asset for train. Eg. npx_clin_ascvd_0_train)")
    parser.add_argument("--asset_version_train", type=str, help="Version of registered data asset. Eg. 3")
    parser.add_argument("--asset_name_test", type=str, help="Name of registered data asset for test. Eg. npx_clin_ascvd_0_test")
    parser.add_argument("--asset_version_test", type=str, help="Version of registered data asset. Eg. 3")    
    parser.add_argument("--drop_columns", type=str, help="Comma-separated list of columns to be dropped before saving.")
    parser.add_argument("--filter_col", type=str, help="Column name to filter rows. Eg. p31")
    parser.add_argument("--filter_val", type=int, help="Value of filter_col to filter rows.")
    parser.add_argument("--out_dir", type=str, help="Path to output directory")
    parser.add_argument("--register_name_prefix", type=str, help="Prefix of registered name to register asset to workspace.")

    # Optional : Feature selection
    parser.add_argument("--feat_sel_algo", type=str, required=False, help="Feature selection algorithm [ebm, mimic-lgbm, shap]")
    parser.add_argument("--feat_sel_path", type=str, required=False, help="Path to file containing selected features. Different format for ebm and shap.")    
    parser.add_argument("--frac_feat", type=float, default=0.1, help="Fraction of features to select")
    parser.add_argument("--label_col", type=str, help="Label column name")
    
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

    print("Train =>")
    load_and_filter_data(ml_client, args.asset_name_train, args.asset_version_train, args)
    print("Test =>")
    load_and_filter_data(ml_client, args.asset_name_test, args.asset_version_test, args)


if __name__=="__main__":

    main()
