#!/bin/bash
#SBATCH -J sci-bert-mask-0.15-1_probs     # Job name
#SBATCH -o out/sci-bert-mask-0.15-probs.o%j # Name of stdout output file(%j expands to jobId)
#SBATCH -e out/sci-bert-mask-0.15-probs.e%j # Name of stderr output file(%j expands to jobId)
#SBATCH -p gtx          # Submit to the 'normal' or 'development' queue
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 02:30:00             # Max run time (hh:mm:ss) - 72 hours
#SBATCH --mail-user=ashwin.devaraj@utexas.edu
#SBATCH --mail-type=ALL

python3 -u sample_bert_multimask.py allenai/scibert_scivocab_uncased sci_bert_probs-mask-0.15.txt 0.15 20 0 4800
