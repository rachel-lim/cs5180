batch_size = 64
target_update_freq = 10
obs_channel = 
crop_size = 128
n_p
n_theta
device
initialize = 
aug = False
aug_type = 'so2'
seed = 123
buffer_size = 100000
max_train_step = 20000
 

# for linear schedule
fixed_eps = False
explore = 0
init_eps = 1.0
final_eps = 0.0
per_beta = 0.4

# planning step
planner_episode = 20
load_sub = False
num_process = 5

# display
no_bar = False 

# training
training_offset = 100
training_iters = 1
time_limit = 10000
