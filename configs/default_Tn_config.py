import ml_collections

def get_Tn_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 1219
  config.train_bs_x = 128
  config.train_bs_t = 128
  
  config.ssm_batch = 40960
  
  config.DSM_warmup = False
  config.T = 2
  config.interval = 1000
  config.t0 = 1e-4
  config.num_stage = 40
  config.num_epoch = 1
  
  config.ckpt_freq= 50000
  config.num_itr = 400000
  config.samp_bs = 5000
  config.forward_net = 'toy'
  
  
  config.checker_board_pattern_num = 8
  
  # ResNet2
  config.model_hidden_dim = 256
  config.model_blocks = 15
  
  # ResNet
  config.model_layers = 20
  
  config.mode = 'so'

  #logging
  config.log_iter = 100
  

  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 5e-4
  config.lr_gamma = 0.9999

  model_configs=None
  return config, model_configs

