#!/bin/bash
#SBATCH -J bart-ul_xsum_maxlen-1024_ul-no-softmax_ul-num-weights-100_ul-alpha-1000_ul-exclude-4,6.sh
#SBATCH -o out/bart-ul_xsum_maxlen-1024_ul-no-softmax_ul-num-weights-100_ul-alpha-1000_ul-exclude-4,6.o%j
#SBATCH -e out/bart-ul_xsum_maxlen-1024_ul-no-softmax_ul-num-weights-100_ul-alpha-1000_ul-exclude-4,6.e%j
#SBATCH -p gtx                  # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 10:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

export OUTPUT_DIR_NAME=bart-ul_xsum_maxlen-1024_ul-no-softmax_ul-num-weights-100_ul-alpha-1000_ul-exclude-4,6_output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
export PYTHONPATH="../../src/":"${PYTHONPATH}"
python finetune.py \
--model_name_or_path=facebook/bart-large-xsum \
--data_dir=data/truncated-1024-inf \
--num_train_epochs=5 \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--gpus=1 \
--freeze_embeds \
--freeze_encoder \
--max_source_length=1024 \
--max_target_length=1024 \
--unlikelihood_training \
--unlikelihood_exclude_tokens=4,6 \
--unlikelihood_num_weights=100 \
--unlikelihood_selective_penalty \
--unlikelihood_alpha=1000 \
--do_train $@

