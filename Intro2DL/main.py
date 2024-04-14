from loguru import logger
import tools.globvar as glob

results_path = glob.get('results_path')
# -------------------------------------------------------- #
import os
import time

import torch
from tqdm import tqdm

from tools.cuda_utils import get_device, schedule_gpu

from configs import expt_group_config_manager
from configs import global_config as config
from tools.expt_manager import create_experiments

from datasets import get_dataset
from models import get_model_obj

device = get_device()
schedule_gpu()


def run_expt(expt):
    # ==================================================== #
    #                  Prepare Environment                 #
    # ==================================================== #
    hp = expt.hyperparameters
    expt.log_hp()
    # ==================================================== #
    #                      Get Dataset                     #
    # ==================================================== #
    dataset = get_dataset(config.data_path, hp)
    train_loader = dataset.get_train_loader(hp['batchsize'])
    val_loader = dataset.get_val_loader(hp['batchsize'])
    # ==================================================== #
    #                      Init Model                      #
    # ==================================================== #
    model_obj = get_model_obj(hp)
    model = model_obj(train_loader, val_loader, hp, expt)
    # ==================================================== #
    #                         Train                        #
    # ==================================================== #
    logger.info('Training')
    start_time = time.time()
    current_epoch = 0
    loop = tqdm(range(1, 1 + hp['iteration']), desc='Training')
    for round in loop:
        train_loss = model.train(1)
        if current_epoch != model.epoch:
            val_result = model.validate()
            val_loss = val_result['loss']
            val_acc = val_result['acc']
            current_epoch = model.epoch
        # ---------------------- log --------------------- #
        if expt.is_log_round(round):
            expt.log(
                {
                    'iteration': round,
                    'epoch': model.epoch,
                    'lr': model.current_lr,
                },
                printout=False)
            expt.log(
                {
                    'train_loss': model.train_loss,
                    'val_loss': model.val_loss,
                    'val_acc': model.val_acc,
                },
                printout=False)
            expt.log({
                'time': time.time() - start_time,
            }, printout=False)
        # --------------------- tqdm --------------------- #
        train_loss = f"{model.train_loss:.6f}"
        val_loss = f"{model.val_loss:.6f}"
        val_acc = f"{model.val_acc:.6f}"
        lr = f"{model.current_lr:.6f}"
        epoch = f"{model.epoch}"
        loop.set_postfix(train_loss=train_loss,
                         val_loss=val_loss,
                         val_acc=val_acc,
                         lr=lr,
                         epoch=epoch)
    # save results
    logger.info(f"train_loss: {model.train_loss:.6f}, " + \
                f"val_loss: {model.val_loss:.6f}, " + \
                f"val_acc: {model.val_acc:.6f}")
    expt.save_to_disc(results_path)

    # test model
    if config.mode == 'test':
        logger.info('Testing')
        test_loader = dataset.get_test_loader(hp['batchsize'])
        test_loss = model.test(test_loader)
        del test_loader
        logger.info(f'test_loss: {test_loss:.6f}')
        expt.log({
            'test_loss': test_loss,
        }, printout=False)
        expt.save_to_disc(results_path)
        torch.save(model.net.state_dict(),
                   os.path.join(results_path, f"model_{expt.log_id}.pth"))

    # clear memory
    del model, dataset, train_loader, val_loader
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()



def main():
    logger.info(f"Experiment Running on {device}...")

    expt_groups_configs = expt_group_config_manager.get_expt_groups_configs()
    expt_groups = create_experiments(expt_groups_configs)
    for i, (name, expt_group) in enumerate(expt_groups.items()):
        logger.info(f"Running ({i+1}/{len(expt_groups)}) group: {name}")
        for expt_cnt, expt in enumerate(expt_group):
            logger.info(f"Running ({expt_cnt+1}/{len(expt_group)}) experiment")
            run_expt(expt)
    logger.info("Experiment Done!")
