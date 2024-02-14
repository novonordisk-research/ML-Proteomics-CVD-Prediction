dataset_name = "npx_clin_ascvd"
dataset_version = "3"
omix = range(4,2945)
clin_start = 2945
num_folds = 10

param_space = {
    'interactions': range(0,20),
    'outer_bags': range(5, 40),
    'inner_bags': range(0, 30),
    'greediness':  [0, 0.5, 0.75], 
    'smoothing_rounds': range(0, 500),
    'max_leaves':  [2, 3, 4, 5],
    'max_bins':  [64, 128, 256, 512],
    'learning_rate':  [0.001, 0.005, 0.01],
    'max_rounds': range(4000, 7000),
    'min_samples_leaf':  [2, 4, 6]
}
