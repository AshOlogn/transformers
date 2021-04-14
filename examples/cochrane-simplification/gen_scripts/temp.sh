#!/bin/bash
#SBATCH -J bart-ul_no-ull_test
#SBATCH -o out/bart-ul_no-ull_test.o%j
#SBATCH -e out/bart-ul_no-ull_test.e%j
#SBATCH -p gtx                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 12:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export OUTPUT_DIR_NAME=bart_xsum_no-freeze_epochs-1_maxlen-1024_output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
export PYTHONPATH="../../src/":"${PYTHONPATH}"
python -u finetune.py \
--model_name_or_path=facebook/bart-large-xsum \
--data_dir=data/truncated-1024-inf \
--num_train_epochs=1 \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--gpus=1 \
--max_source_length=1024 \
--max_target_length=1024 \
--generate_input_prefix=test \
--generate_batch_size=4 \
--generate_epoch=1 \
--generate_start_index=0 \
--generate_end_index=10 \
--do_generate



