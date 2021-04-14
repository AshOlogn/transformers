#!/bin/bash
#SBATCH -J gen_bart
#SBATCH -o out/gen_bart.o%j
#SBATCH -e out/gen_bart.e%j
#SBATCH -p gtx                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 10:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export OUTPUT_DIR_NAME=bart_e-3_bs-1_lr-3e-5_w-linear-warmup-0_acc-1
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
export PYTHONPATH="../../src/":"${PYTHONPATH}"
python -u finetune.py \
--model_name_or_path=sshleifer/distilbart-xsum-6-6 \
--data_dir=data/truncated-1024-inf \
--num_train_epochs=3 \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--gpus=1 \
--max_source_length=1024 \
--max_target_length=1024 \
--generate_input_prefix=test \
--generate_epoch=3 \
--generate_start_index=0 \
--generate_end_index=250 \
--decode_method=nucleus \
--decode_p=0.9 \
--do_generate


#export OUTPUT_DIR_NAME=bart_e-3_bs-2048_lr-3e-5_w-constant-warmup-0.20_acc-1
#export CURRENT_DIR=${PWD}
#export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
#
## Make output directory if it doesn't exist
#mkdir -p $OUTPUT_DIR
#
## Add parent directory to python path to access lightning_base.py and utils.py
#export PYTHONPATH="../":"${PYTHONPATH}"
#export PYTHONPATH="../../src/":"${PYTHONPATH}"
#python -u finetune.py \
#--model_name_or_path=sshleifer/distilbart-xsum-6-6 \
#--data_dir=data/truncated-1024-inf \
#--num_train_epochs=3 \
#--learning_rate=3e-5 \
#--train_batch_size=1 \
#--eval_batch_size=1 \
#--output_dir=$OUTPUT_DIR \
#--gpus=1 \
#--max_source_length=1024 \
#--max_target_length=1024 \
#--generate_input_prefix=test \
#--generate_epoch=3 \
#--generate_start_index=0 \
#--generate_end_index=250 \
#--decode_method=beam \
#--decode_num_beams=5 \
#--do_generate
#
#
#
#export OUTPUT_DIR_NAME=bart_e-3_bs-2048_lr-3e-5_w-linear-warmup-0.20_acc-1
#export CURRENT_DIR=${PWD}
#export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
#
## Make output directory if it doesn't exist
#mkdir -p $OUTPUT_DIR
#
## Add parent directory to python path to access lightning_base.py and utils.py
#export PYTHONPATH="../":"${PYTHONPATH}"
#export PYTHONPATH="../../src/":"${PYTHONPATH}"
#python -u finetune.py \
#--model_name_or_path=sshleifer/distilbart-xsum-6-6 \
#--data_dir=data/truncated-1024-inf \
#--num_train_epochs=3 \
#--learning_rate=3e-5 \
#--train_batch_size=1 \
#--eval_batch_size=1 \
#--output_dir=$OUTPUT_DIR \
#--gpus=1 \
#--max_source_length=1024 \
#--max_target_length=1024 \
#--generate_input_prefix=test \
#--generate_epoch=3 \
#--generate_start_index=0 \
#--generate_end_index=250 \
#--decode_method=beam \
#--decode_num_beams=5 \
#--do_generate

