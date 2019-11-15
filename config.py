params = dict()

params['num_classes'] = 101

params['dataset'] =  '/data2/video_data/UCF-101'
#params['dataset'] = '/data/hmdb'
#params['dataset'] = '/data/kinetics-400'

params['epoch_num'] = 150#600
params['batch_size'] = 8
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 0.001
params['momentum'] = 0.9
params['weight_decay'] = 0.0005
params['display'] = 10
params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'
#params['save_path'] = 'UCF101'
params['save_path_base']='/data2/fb/project/predict-puzzle-master'
params['data']='UCF-101'

