# LightGBM model - Training, Inference and Interpretation

This `lightgbm` folder contains the code for training, inference and interpretation of LightGBM family of models. We utilized the Azure AutoML framework for training the LightGBM models which provides a way to tune for hyperparameters and normalizers.

## Step 1: Create MLTable for AutoML training

The data assets are assumed to be registered in the AML workspace as .csv files. However, Azure AutoML requires the data to be in a specific format such as MLTable for consumption. In the first step, we will convert data asset to MLTable format and register the MLTable to AML workspace.

Here is an example usage.

```bash
cd /path/to/repo/lightgbm

conda deactivate
conda env create -f conda_envs/automl_train.yml
conda activate automl_train

python create_mltable.py --asset_name_train npx_clin_ascvd_0_train --asset_version_train 3 --asset_name_test npx_clin_ascvd_0_test --asset_version_test 3 --drop_columns IID,ascvd_3yr_label,ascvd_5yr_label,ascvd_15yr_label --out_dir ./dummy_10yr_all_0 --register_name_prefix dummy_mltable_10yr_all
```

## Step 2: Submit AutoML job

The AutoML run requires the MLTable data assets created in previous step. The script `submit_automl_train.py` contains the code for the job submission. 

Here is an example usage.

```bash
python submit_automl_train.py --asset_name_train dummy_mltable_10yr_all_npx_clin_ascvd_0_train --asset_version_train 1 --asset_name_test dummy_mltable_10yr_all_npx_clin_ascvd_0_test --asset_version_test 1 --aml_experiment_name dummy_LightGBM_10yr_all_npx_clin_ascvd_0 --cluster_name cluster1 --label_col ascvd_10yr_label
```

## Step 3: Inference on best model checkpoint

At the end of training, the job will show the mean dev AuC for multiple trials in the AML workspace. 
![Model leaderboard](./docs/model_performance_catalog.PNG)

Select the model with the best dev AuC and download its corresponding checkpoint. 
![Download best checkpoint](./docs/model_checkpoint_download.PNG)

Unzip the model checkpoint folder and then run inference script on the checkpoint directory. For example, let the name of the unzipped folder be `automl_lgbm_best_10yr_all_npx_clin_ascvd_0`. The folder contains three files named `model.pkl`, `conda_env_v_1_0_0.yml` and `scoring_file_v_2_0_0.py`. The `model.pkl` is the file we need for inference. For inference using AutoML, we need to create and activate the `automl_infer` conda environment.

Here is an example usage of inference. It will create the model predictions for all instances in test data asset in `pred_dir`.

```bash
conda deactivate
conda env create -f conda_envs/automl_infer.yml
conda activate automl_infer

python automl_inference.py --asset_name_test npx_clin_ascvd_0_test --asset_version_test 3 --model_dir ./automl_lgbm_best_10yr_all_npx_clin_ascvd_0 --pred_dir ./pred_automl_universal_10yr --feat_col_start 5
```
The above examples show how to train and infer on a single data split, e.g. split "0". This entire process needs to be repeated for every split.

Moreover, to obtain aggregated metrics, user needs to obtain the predictions for all splits (using respective best model checkpoints) in `pred_dir` and then run model evaluation pipeline on `pred_dir`.

## (Optional) Step 4: Feature importance for best model checkpoint

Using the best model checkpoint and the train data, we compute the global feature importance as average SHAP value across all data instances.

Here is an example usage of SHAP computation.

```bash
python shap_compute.py --asset_name_train npx_clin_ascvd_0_train --asset_version_train 3 --model_dir ./automl_lgbm_best_10yr_all_npx_clin_ascvd_0 --out_dir ./dummy_shap_universal_10yr --feat_col_start 5
```
