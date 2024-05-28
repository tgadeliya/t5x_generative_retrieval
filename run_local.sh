python3 ./t5x/scripts/xm_launch_local.py \
  --gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin \
  --model_dir=gs://generative_retrieval_usa/t5x/$(date +%Y%m%d) \
  --tfds_data_dir=gs://generative_retrieval_usa/tsimur.hadeliya/