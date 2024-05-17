cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

python /home/oem/fx/planTF/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    scenario_builder=nuplan \
    scenario_filter=val14_split \
    worker=sequential \
    verbose=true \
    +planner.imitation_planne.planner_ckpt=/home/oem/fx/planTF/checkpoints/planTF.ckpt