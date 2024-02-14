# Utility script to load data from AML workspace.

import pandas as pd

def load_pandas_df(ml_client, asset_name, asset_version):

    # Load data asset in AML workspace.
    data_asset = ml_client.data.get(name=asset_name, version=asset_version)
    data_df = pd.read_csv(data_asset.path)
         
    # Convert boolean features to float.
    data_df.replace({False: 0.0, True: 1.0}, inplace=True)
    print(f"# Data points = {len(data_df)}")

    return data_df