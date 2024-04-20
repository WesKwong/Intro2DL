# ======================================================== #
#                    Prepare Environment                   #
# ======================================================== #
import os
import sys

from loguru import logger

import tools.globvar as glob
from configs import global_config as config

glob._init()
logger.remove()
logger.add(sys.stdout, level=config.log_level, backtrace=True, diagnose=True)
# ======================================================== #
#                           Main                           #
# ======================================================== #
from tools.expt_utils import set_seed, get_unique_results_path

RESULTS_PATH = get_unique_results_path(config.results_path, config.expt_name)
logger.add(os.path.join(RESULTS_PATH, 'log.log'),
           level=config.log_level,
           backtrace=True,
           diagnose=True)
glob.set('results_path', RESULTS_PATH)
set_seed(config.random_seed)


@logger.catch
def run():
    from main import main
    main()


if __name__ == "__main__":
    run()
