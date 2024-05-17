cwd=$(pwd)
CKPT_ROOT="/home/oem/fx/planTF/checkpoints/planTF.ckpt"

PLANNER="planTF"
SPLIT=val14
CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents open_loop_boxes"

for challenge in $CHALLENGES; do
    python /home/oem/fx/planTF/run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        scenario_builder=nuplan \
        scenario_filter=$SPLIT \
        worker.threads_per_node=20 \
        experiment_uid=$SPLIT/$planner \
        verbose=true \
        planner.imitation_planner.planner_ckpt=$CKPT_ROOT
        # +planner.imitation_planne.planner_ckpt=/home/oem/fx/planTF/checkpoints/planTF.ckpt
done


