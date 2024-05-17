import logging
import os
import pprint
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union
import pathlib
import hydra
import pandas as pd
import pytorch_lightning as pl
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.utils import (
    run_runners,
    set_default_path,
    set_up_common_builder,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv("NUPLAN_HYDRA_CONFIG_PATH", "config/simulation")


def print_simulation_results(result):
    # root = Path(os.getcwd()) / "aggregator_metric"
    # result = "/home/oem/zkf/plantf_cache_1M_epoch_24_closed_loop_reactive_agents_val14_1221/plantf_cache_1M_epoch_49_closed_loop_reactive_agents_val14/2023.12.22.19.51.43/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2023.12.22.19.51.43.parquet"
    
    df = pd.read_parquet(result)
    final_score = df[df["scenario"] == "final_score"]
    final_score = final_score.to_dict(orient="records")[0] #将final_score转换为字典格式
    pprint.PrettyPrinter(indent=4).pprint(final_score)

# root=pathlib.Path("/home/oem/zkf/plantf_cache_1M_epoch_24_closed_loop_nonreactive_agents_val14_1221/plantf_cache_1M_epoch_49_closed_loop_nonreactive_agents_val14")
# root=root/"aggregator_metric"
result="/home/oem/zkf/plantf_cache_1M_epoch_24_closed_loop_nonreactive_agents_val14_1221/plantf_cache_1M_epoch_49_closed_loop_nonreactive_agents_val14/2023.12.22.09.28.58/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2023.12.22.09.28.58.parquet"
print_simulation_results(result=result)