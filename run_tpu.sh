export GOOGLE_CLOUD_BUCKET_NAME=generative_retrieval_usa
#export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
#export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d)

# Pre-download dataset in multi-host experiments.
# tfds build wmt_t2t_translate --data_dir=$TFDS_DATA_DIR

#git clone https://github.com/google-research/t5x
#cd ./t5x/

python3 ./t5x/scripts/xm_launch.py \
  --gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt_finetune.gin \
  --model_dir=gs://generative_retrieval_usa/t5x/$(date +%Y%m%d) \
  --tfds_data_dir=gs://generative_retrieval_usa/tsimur.hadeliya/