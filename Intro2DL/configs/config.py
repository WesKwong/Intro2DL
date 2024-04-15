import itertools as it


def get_configs_dict(configs):
    configs_dict = {}
    for config_name, config_value in configs.items():
        if config_name.startswith('__') and config_name.endswith('__'):
            continue
        configs_dict[config_name] = config_value
    return configs_dict


# ======================================================== #
#                 Experiment Group Configs                 #
# ======================================================== #


class ExptGroupConfigDefault(object):
    group_name = ['main']
    dataset = ['CIFAR10']
    net = ['LeNet5']
    lr = [1e-2]
    optimizer = ['SGD']
    scheduler = [{
        'name': 'StepLR',
        'min_lr': 1e-5,
        'param': {
            'step_size': 1,
            'gamma': 1
        }
    }]
    batchsize = [64]
    iteration = [35000]
    log_freq = [100]
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())


class ExptLab1N200(object):
    group_name = ['Lab1_N200']
    dataset = [{'name': 'Lab1', 'param': {'N': 200}}]
    net = ['MLP']
    lr = [1e-2]
    optimizer = ['Adam']
    scheduler = [{
        'name': 'StepLR',
        'min_lr': 1e-4,
        'param': {
            'step_size': 50,
            'gamma': 0.9
        }
    }]
    batchsize = [16]
    iteration = [60000]
    log_freq = [100]
    # Lab1 specific settings
    model = ['Lab1']
    hidden_sizes = [[64, 32, 16]]
    activation = ['ReLU']
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())


class ExptLab1N2000(object):
    group_name = ['Lab1_N2000']
    dataset = [{'name': 'Lab1', 'param': {'N': 2000}}]
    net = ['MLP']
    lr = [1e-2]
    optimizer = ['Adam']
    scheduler = [{
        'name': 'StepLR',
        'min_lr': 1e-4,
        'param': {
            'step_size': 50,
            'gamma': 0.9
        }
    }]
    batchsize = [64]
    iteration = [60000]
    log_freq = [100]
    # Lab1 specific settings
    model = ['Lab1']
    hidden_sizes = [[128, 64]]
    activation = ['ReLU']
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())


class ExptLab1N10000(object):
    group_name = ['Lab1_N10000']
    dataset = [{'name': 'Lab1', 'param': {'N': 10000}}]
    net = ['MLP']
    lr = [1e-2]
    optimizer = ['Adam']
    scheduler = [{
        'name': 'StepLR',
        'min_lr': 1e-4,
        'param': {
            'step_size': 50,
            'gamma': 0.9
        }
    }]
    batchsize = [512]
    iteration = [60000]
    log_freq = [100]
    # Lab1 specific settings
    model = ['Lab1']
    hidden_sizes = [[256, 128]]
    activation = ['ReLU']
    # ---------------------------------------------------- #
    configs_dict = get_configs_dict(locals())


class ExptGroupConfigManager(object):
    expt_groups = [ExptGroupConfigDefault.configs_dict]

    def get_expt_groups_configs(self):
        expt_groups_configs = {}
        for expt_group in self.expt_groups:
            combinations = it.product(*(expt_group[name]
                                        for name in expt_group))
            expt_group_configs = [{
                key: value[i]
                for i, key in enumerate(expt_group)
            } for value in combinations]
            expt_groups_configs[expt_group['group_name']
                                [0]] = expt_group_configs
        return expt_groups_configs


# ======================================================== #
#                       Global Config                      #
# ======================================================== #
class GlobalConfig(object):
    expt_name = None
    data_path = "data/"
    results_path = "results/"
    random_seed = 114514
    mode = "train"
    log_level = "INFO"
    prepare_new_dataset = True


expt_group_config_manager = ExptGroupConfigManager()
global_config = GlobalConfig()
