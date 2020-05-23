#!/bin/bash

#SBATCH --job-name=viscomp_vqa
#SBATCH --output=slurm.out              # Output file name
#SBATCH --error=slurm.err               # Error file name
#SBATCH --partition=gpu2018                 # Partition
#SBATCH --time=00:30:00
#SBATCH --qos=short+                    # Queue
#SBATCH --gres=gpu:1                    # GPU needed
#SBATCH --mem=20000
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # MPI processes per node
#SBATCH --account=pi_ferraro

### exit on error
set -e
### make evaluating unset variables an error
set -o nounset

### set up the environment
module use ~/ferraro_common/module_files
module load conda/2.7

### run it
source activate cuda9-py38-pytorch1.5

# python vqa_preprocessing.py --train_questions ~/ferraro_user/viscomp_vqa/data/annotations/train_questions_471.json \
# 				--test_questions ~/ferraro_user/viscomp_vqa/data/annotations/val_questions_81.json \
# 				--train_annotations ~/ferraro_user/viscomp_vqa/data/annotations/train_annotations_471.json \
# 				--test_annotations ~/ferraro_user/viscomp_vqa/data/annotations/val_annotations_81.json

python prepro_data.py --train_input ~/ferraro_user/viscomp_vqa_generation/data/vqa_raw_train.json \
				--test_input ~/ferraro_user/viscomp_vqa_generation/data/vqa_raw_test.json \
				--output_json ~/ferraro_user/viscomp_vqa_generation/data_prepro.json \
				--variation bsq

python batch_train_pytorch.py --input_data_file ~/ferraro_user/viscomp_vqa_generation/data_prepro.json \
				--input_img_file ~/ferraro_user/viscomp_vqa_generation/data_img_471.h5 \
				--input_bert_emb ~/ferraro_user/viscomp_vqa_generation/data_text_bert_sq_471.h5 \
				--model_save ~/ferraro_user/viscomp_vqa_generation/model_save/ \
				--variation bsq

declare -a arr=("0" "20" "40" "60" "80" "100")

for num in "${arr[@]}"; do
	python batch_test_pytorch.py --input_data_file ~/ferraro_user/viscomp_vqa_generation/data_prepro.json \
				--input_img_file ~/ferraro_user/viscomp_vqa_generation/data_img_471.h5 \
				--input_bert_emb ~/ferraro_user/viscomp_vqa_generation/data_text_bert_sq_471.h5 \
				--checkpoint_path ~/ferraro_user/viscomp_vqa_generation/model_save/model_bsq/epoch-"$num".pt \
				--results_path ~/ferraro_user/viscomp_vqa_generation/results/ \
				--variation bsq
done
