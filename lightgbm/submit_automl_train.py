# Script to submit Azure AutoML job.

import argparse
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import automl, Input


def job_wrapper(ml_client, args):

    # Retrieve an already attached Azure Machine Learning Compute.
    cluster_name = args.cluster_name
    print(ml_client.compute.get(cluster_name))


    # ML Table registered by explicit creation of MLTable file in the csv folder.
    train_plus_dev_asset = ml_client.data.get(name=args.asset_name_train, version=args.asset_version_train)
    test_asset = ml_client.data.get(name=args.asset_name_test, version=args.asset_version_test)
    train_plus_dev_data_input = Input(type=AssetTypes.MLTABLE, path=train_plus_dev_asset.id)
    test_data_input = Input(type=AssetTypes.MLTABLE, path=test_asset.id)

    # Define classification job

    classification_job = automl.classification(
        compute=cluster_name,
        experiment_name=args.aml_experiment_name,
        training_data=train_plus_dev_data_input,
        n_cross_validations=4,
        target_column_name=args.label_col,
        primary_metric="AUC_weighted",
        test_data=test_data_input,
        enable_model_explainability=True
    )

    classification_job.set_featurization(mode="off")

    # Limits are all optional
    classification_job.set_limits(
        timeout_minutes=720, 
        trial_timeout_minutes=180, 
        max_trials=20,
        max_concurrent_trials=1,
        enable_early_termination=True
    )

    # Training properties are optional
    classification_job.set_training(
        enable_model_explainability=True,
        enable_vote_ensemble=False,
        enable_stack_ensemble=False,
        allowed_training_algorithms=["LightGBM"]
    )

    # Submit the AutoML job
    automl_job = ml_client.jobs.create_or_update(
        classification_job
    )  # submit the job to the backend

    print(f"Created job: {automl_job}")

    # Get a URL for the status of the job
    print(automl_job.services["Studio"].endpoint)

def main():

    parser = argparse.ArgumentParser(description="SHAP computation")
    parser.add_argument("--asset_name_train", type=str, help="Name of registered MLTable data asset for train. Eg. mltable_10yr_all_npx_clin_ascvd_0_train)")
    parser.add_argument("--asset_version_train", type=str, help="Version of registered MLTable data asset for train. Eg. 1")
    parser.add_argument("--asset_name_test", type=str, help="Name of registered MLTable data asset for test. Eg. mltable_10yr_all_npx_clin_ascvd_0_test")
    parser.add_argument("--asset_version_test", type=str, help="Version of registered MLTable data asset. Eg. 1")
    parser.add_argument("--aml_experiment_name", type=str, help="Name of the AML experiment. Eg. dummy_LightGBM_10yr_all_npx_clin_ascvd_0")
    parser.add_argument("--cluster_name", type=str, help="Name of the AML cluster.")
    parser.add_argument("--label_col", type=str, help="Label column name. Eg- ascvd_10yr_label")
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
    job_wrapper(ml_client, args)


if __name__=="__main__":
    main()
