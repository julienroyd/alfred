import numpy as np
from math import floor, log10
from collections import OrderedDict
from alfred.utils.misc import round_to_two


# (1) Enter the algorithms to be run for each experiment

ALG_NAMES = ['simpleMLP']

# (2) Enter the task (dataset or rl-environment) to be used for each experiment

TASK_NAMES = ['MNIST']

# (3) Enter the seeds to be run for each experiment

N_SEEDS = 3
SEEDS = [1 + x for x in range(N_SEEDS)]


# (4) Hyper-parameters. For each hyperparam, enter the function that you want the random-search to sample from.
#     For each experiment, a set of hyperparameters will be sampled using these functions

# Examples:
# int:          np.random.randint(low=64, high=512)
# float:        np.random.uniform(low=-3., high=1.)
# bool:         bool(np.random.binomial(n=1, p=0.5))
# exp_float:    10.**np.random.uniform(low=-3., high=1.)
# fixed_value:  fixed_value

def sample_experiment():
   sampled_config = OrderedDict({
       'learning_rate': 10. ** np.random.uniform(low=-8., high=-3.),
       'optimizer': "sgd",
   })

   # Security check to make sure seed, alg_name and task_name are not defined as hyperparams

   assert "seed" not in sampled_config.keys()
   assert "alg_name" not in sampled_config.keys()
   assert "task_name" not in sampled_config.keys()

   # Simple security check to make sure every specified parameter is defined only once
   keys = list(sampled_config.keys())
   counted_keys = {key: keys.count(key) for key in keys}
   for key in counted_keys.keys():
       if counted_keys[key] > 1:
           raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')

   return sampled_config
