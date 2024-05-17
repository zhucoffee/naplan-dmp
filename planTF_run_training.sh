export PYTHONPATH=$PYTHONPATH:$(pwd)

JOB_NAME=training_planTF_reproduce_0109_with_dipp_epoch25
EXPERIMENT=training_planTF_num_scenarios_4000_0109_with_dipp
CACHE_PATH='/home/oem/zkf/cache/plantf_cache/test_plantf_cache'
USE_CACHE_WITHOUT_DATASET=True


# worker.max_workers=32 \
# scenario_filter.num_scenarios_per_type=4000 \

CUDA_VISIBLE_DEVICES=0,1,2 \
python run_training.py \
py_func=train +training=train_planTF \
job_name=$JOB_NAME \
experiment_name=$EXPERIMENT \
scenario_builder=nuplan \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
worker=single_machine_thread_pool \
data_loader.params.batch_size=32 \
data_loader.params.num_workers=32 \
lr=1e-3 \
epochs=25 \
warmup_epochs=3 \
weight_decay=0.0001 \
lightning.trainer.params.val_check_interval=0.5 \
# wandb.mode=online \
# wandb.project=nuplan \
# wandb.name=plantf