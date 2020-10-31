#!/bin/bash
#SBATCH -J sci-bert-mask_dataset
#SBATCH -o out/sci-bert-mask_dataset.o%j
#SBATCH -e out/sci-bert-mask_dataset.e%j
#SBATCH -p gtx          # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 05:00:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

python3 -u sample_bert_multimask.py allenai/scibert_scivocab_uncased data/data_final_1025.json sci_bert_probs-mask-0.15_dataset.txt 0.15 20 0 4459
