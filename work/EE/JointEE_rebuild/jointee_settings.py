# self attan
n_head = 6
d_head = 1024

# alpha越高，召回越高
# role_alpha = 0.35
role_alpha = 0.95
# role_alpha = 0.99
role_gamma = 2

trigger_alpha = 0.3
trigger_gamma = 2

trigger_extraction_threshold = 0.5
argument_extraction_threshold = 0.5

# path
plm_path = 'bert-base-chinese'


# learning rate
plm_lr = 2e-5
others_lr = 1e-4
