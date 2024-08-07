import os
import time
import random
import numpy as np
import torch
from loguru import logger

import datasets


def get_time_str():
    time_str = time.strftime("%Y%m%d%H%M%S")
    year = time_str[2:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hour = time_str[8:10]
    minute = time_str[10:12]
    second = time_str[12:14]
    return year + month + day + hour + minute + second


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_unique_results_path(path, expt_name=None):
    unique_name = get_time_str()
    if expt_name is not None:
        unique_name = unique_name + "_" + expt_name
    unique_results_path = os.path.join(path, unique_name)
    if not os.path.exists(unique_results_path):
        os.makedirs(unique_results_path)
    return unique_results_path


def get_expt_dataset_config_set(expt_groups_configs):
    expt_dataset_config_set = set()
    for group_name, expt_group_configs in expt_groups_configs.items():
        for expt_group_config in expt_group_configs:
            dataset = expt_group_config['dataset']
            expt_dataset_config_set.add(dataset)
    return expt_dataset_config_set


def prepare_datasets(path, num_client, data_distribution, expt_groups_configs):
    expt_dataset_config_set = get_expt_dataset_config_set(expt_groups_configs)
    for dataset in expt_dataset_config_set:
        logger.info(f"Preparing dataset: {dataset}")
        getattr(datasets, dataset + "Splitter")(path, num_client,
                                                data_distribution)


def log_progress_detail(start_time, current_round, total_rounds):
    elapsed_t = time.time() - start_time
    remaining_rounds = total_rounds - current_round
    avg_round_time = elapsed_t / current_round
    remain_t = int(avg_round_time * remaining_rounds)
    elapsed_t_str = '{:02d}:{:02d}:{:02d}'.format(
        int(elapsed_t) // 3600,
        int(elapsed_t) % 3600 // 60,
        int(elapsed_t) % 60)
    remain_t_str = '{:02d}:{:02d}:{:02d}'.format(remain_t // 3600,
                                                 (remain_t % 3600 // 60),
                                                 remain_t % 60)
    progress_percentage = current_round / total_rounds * 100
    progress_percentage_str = "{:_>7.3f}%".format(progress_percentage)
    total_round_str_len = len(str(total_rounds))
    current_round_str = "{:_>{}}".format(current_round, total_round_str_len)
    progress_str = f"[{current_round_str}/{total_rounds}] " \
                   f"[{elapsed_t_str}->{remain_t_str}] " \
                   f"[{progress_percentage_str}]"
    logger.info(progress_str)

def log_progress_header(total_rounds):
    total_round_str_len = len(str(total_rounds))
    header_str = f"{'Round':*^{int(2*total_round_str_len)+3}} " \
                 f"**Elapse*->*Remain** " \
                 f"*Progress*"
    logger.info(header_str)


def executor_callback(worker):
    logger.info("called worker callback function")
    worker_exception = worker.exception()
    if worker_exception:
        logger.exception("Worker return exception: {}".format(worker_exception))
