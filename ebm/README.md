# EBM model - Training, Inference and Interpretation 

## Step 0: Install the pre-requisites

```bash
cd /path/to/repo/
export PYTHONPATH=$PYTHONPATH:.:./ebm
conda deactivate
conda env create -f ebm/conda_environment.yml
conda activate ebm_env
```

## Step 1: Proteomics feature selection

We first create different models for each fold's training set using only proteomics features:

- Model on the entire proteomics feature set
- Model on a subset of samples with overrepresentation of the positive class
- Models for other time horizons
- Models for subpopulations specified in `group_variables` with cutoffs in `group_variable_values` by default, these are men/women, young/old (split on median age 58), patients with normal or below normal weight and those overweight (split on median BMI 26.6)

For each model and fold, we save the sorted (in descending order of importance) feature sets.

Substeps:

1. Split the training set into train and validation
1. Train each of the above models on the train portion and determining best number of features on the validation portion didn't work (accuracy on validation wasn't predictive of accuracy on test).

Thus, we train these models on the entire training set and use a fixed number of features in the next step. The test set is not used.

Code: `python ebm/omics_feature_selection.py`. Parameters (none required, all have default settings):

- `horizon` (default `"ascvd_10yr_label"`)
- `group_variables` (variables to split on for separate training, default `["p31", "p21022", "p21001_i0"]`)
- `group_variable_values` (cutoff values for split variables, default `[0.0, 58.0, 26.6]`)
- `feature_dir` (directory to store feature files, default `"ebm/features"`)
- `seed` (for reproducibility, default `1234`)

## Step 2: Hyperparameter tuning

Run hyperparameter tuning on each fold using only proteomics or both proteomics and clinical variables. Some noteworthy points:

- For proteomics variables, only use the union of top features from each model created in Step 1. The number of features to use from each model is specified in `num_top_features`.
- Group variables and their values are only needed to load correct feature sets. 
- Each fold's training set is split into training and validation sets. We use the [hyperopt](http://hyperopt.github.io/hyperopt/) to run Bayesian optimization. The training occurs on the train part of the training set, the evaluation on the validation part of the training set. The test set is not used.  
- Group variables and their values are only needed to load correct feature sets. 

Code: `python ebm/hyperparameter_tuning.py`. Parameters (none required, all have default settings):

- `horizon` (default `"ascvd_10yr_label"`)
- `group_variables` (variables used in feature selection, default `["p31", "p21022", "p21001_i0"]`)
- `group_variable_values` (cutoff values for split variables, default `[0.0, 58.0, 26.6]`)
- `num_tries` (number of configuration setttings to try for each fold, default `40`)
- `num_top_features` (number of top features to include for each model trained in step 1, default `70`)
- `add_clinical` (whether to include clinical variables, default `True`)
- `feature_dir` (directory with feature files, default `"ebm/features"`)
- `results_dir` (directory to store optimization result files, default `"ebm/results"`)
- `seed` (for reproducibility, default `1234`)

## Step 3: Training the final model

This step consists of the following substeps:

1. Use the union of top `num_top_features` proteomics features from each model found in Step 1.
1. Group variables and their values are only needed to load correct feature sets. 
1. Add clinical features (or not, depending on the setting of the command line parameter).
1. Use the best set of hyperparameters determined in Step 2.  Use top `fraction_tries_used` trial runs for each fold and average parameter values.  
1. Train the final model for each fold.
1. Make predictions on each fold's test set and save.

Code: `python ebm/final_model_training.py`. Parameters (none required, all have default settings):

- `horizon` (default `"ascvd_10yr_label"`)
- `group_variables` (variables used in feature selection, default `["p31", "p21022", "p21001_i0"]`)
- `group_variable_values` (cutoff values for split variables, default `[0.0, 58.0, 26.6]`)
- `num_top_features` (number of top features to include for each model trained in step 1, default `70`)
- `add_clinical` (whether to include clinical variables, default `True`)
- `fraction_tries_used` (fraction of top configurations to average, default `0.1`)
- `feature_dir` (directory with feature files, default `"ebm/features"`)
- `results_dir` (directory with optimization result files, default `"ebm/results"`)
- `model_dir` (directory to store models, default `"ebm/models"`)
- `pred_dir` (directory to store prediction files, default "ebm/predictions")
- `seed` (for reproducibility, default `1234`)

**Note:** `num_top_features` and `add_clinical` should match settings from Step 2.