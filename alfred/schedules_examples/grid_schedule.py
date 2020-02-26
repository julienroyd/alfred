# (1) Enter the algorithms to be run for each experiment

ALG_NAMES = ['simpleMLP']

# (2) Enter the task (dataset or rl-environment) to be used for each experiment

TASK_NAMES = ['MNIST']

# (3) Enter the seeds to be run for each experiment

N_SEEDS = 3
SEEDS = [1 + x for x in range(N_SEEDS)]

# (4) Hyper-parameters

# Here, for each hyperparam, enter the values you want to try in a list.
# All possible combinations will be run as a separate experiment
# Unspecified (or commented out) params will be set to default defines in main.get_training_args

VARIATIONS = {
    'learning_rate': [0.1, 0.01, 0.001],
    'optimizer': ["sgd", "adam"],
}

# Security check to make sure seed, alg_name and task_name are not defined as hyperparams

assert "seed" not in VARIATIONS.keys()
assert "alg_name" not in VARIATIONS.keys()
assert "task_name" not in VARIATIONS.keys()

# Security check to make sure every specified parameter are defined only once

keys = list(VARIATIONS.keys())
counted_keys = {key: keys.count(key) for key in keys}
for key in counted_keys.keys():
    if counted_keys[key] > 1:
        raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')
