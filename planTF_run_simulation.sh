cwd=$(pwd)
# CKPT_ROOT="$cwd/checkpoints"


CKPT_ROOT=/home/oem/zkf/planTF.ckpt
PLANNER="planTF"
SPLIT=val14
CHALLENGES=closed_loop_nonreactive_agents
#"closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"


JOB_NAME=simulation_planTF_num_scenarios_4000_0110_with_dipp
EXPERIMENT=simulation_planTF_num_scenarios_4000_0110_with_dipp_$CHALLENGES\_$SPLIT

export HYDRA_FULL_ERROR=1

python run_simulation.py \
+simulation=$CHALLENGES \
job_name=$JOB_NAME \
experiment_name=$EXPERIMENT \
planner=$PLANNER \
scenario_builder=nuplan \
worker=single_machine_thread_pool \
worker.max_workers=32 \
scenario_filter=$SPLIT \
verbose=true \
planner.imitation_planner.planner_ckpt=$CKPT_ROOT



