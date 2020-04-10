# viscomp-vqa
This code is done in Python 3.7.4, Tensorflow 1.14 and cuda 10.0. I am attaching conda environment file which will install all the requirements necessary to run the code. Please use the same set of instructions for running both the variations in their own directories.

After activating the environment, 

#### 267-way classification 
## checkpoint_path = 'model_save_ans/'

1. First by running the notebook GenerateJsonFiles.ipynb, 4 JSON files will be generated in "data/annotations" - train_annotations.json, val_annotations.json, train_questions.json, val_questions.json. (I am attaching the generated files.)

2. Change directory to "data" and run -
		python vqa_preprocessing.py
	This will generate 2 files in the data folder, vqa_raw_train.json and vqa_raw_test.json.

3. Come back to main directory and run - 
		python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 267
	This will generate 2 files in the main folder data_prepro.h5 and data_prepro.json.

4. For extracting image features, download the train and val folders from "https://drive.google.com/open?id=1GKyFDTcOvxy7XXxyNhkHTs6WpCsBnKoy" link and extract them in the "data" directory. Then download the pretrained VGGNet 19 layer model from this site "https://gist.github.com/ksimonyan/3785162f95cd2d5fee77" and then run -
		python prepro_img.py
	This will generate data_img.h5 file. 

5. Now, train the model by running - 
		python model_viscomp_vqa.py
	This will train the model and save models at each 3000th epoch in the "model_save" directory. This will also generate the losses.json file in the main directory.

6. To test the model run the following -
		python test_viscomp_vqa.py --model_path model_save/model-3000
	This will run the saved model at 3000th epoch. Enter any model number from [3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]. This will generate data.json file which will contain the results. 

7. Finally run the follwing command to get the final results in format -
		python s2i.py
	This will generate val_results.py file in "Results" directory. I have saved the results for above listed epoch in the Results folder. 

8. Please run the VisualizeResults.ipynb notebook to visualize the results. I have attached the .html file of this notebook to view the results. 


-----------------------------------------------------------------------------------------------------------------------------
#### multiple choice 5-way classification with full data 
## checkpoint_path = 'model_save/model_save_{variation}/'
1.
 	python vqa_preprocessing.py --train_questions annotations/train_questions_8100.json --test_questions annotations/val_questions_2154.json --train_annotations annotations/train_annotations_8100.json --test_annotations annotations/val_annotations_2154.json

2. 
	python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --multiple_choice true

3. download val -
	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216435&authkey=ABHASiu99I0otX0"

	download train -
	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216434&authkey=AFoRfjqXpONXPnA"

	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216433&authkey=AD5zv58D09lkFP0"

4. 
	python prepro_img.py

5. variation = isq/iq/sq/q  
	python model_viscomp_vqa.py --variation 

6. 
	python test_viscomp_vqa.py --variation --model_number

-----------------------------------------------------------------------------------------------------------------------------
#### multiple choice 5-way classification - offline description text processing
## checkpoint_path = 'model_save_offline/'
1.
	cd data/
 	python vqa_preprocessing.py --train_questions annotations/train_questions_8100.json --test_questions annotations/val_questions_2154.json --train_annotations annotations/train_annotations_8100.json --test_annotations annotations/val_annotations_2154.json

2. 
	python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --multiple_choice true 

3. download val -
	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216435&authkey=ABHASiu99I0otX0"

	download train -
	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216434&authkey=AFoRfjqXpONXPnA"

	wget --no-check-certificate "https://onedrive.live.com/download?cid=93E4D1B07E84AB43&resid=93E4D1B07E84AB43%216433&authkey=AD5zv58D09lkFP0"

4. 
	python prepro_img.py
 
5. LSTM -
	python model_prepro_text.py
	python test_prepro_text.py
   
   BERT -
   	source activate vqa_env2
   	python get_bert_embeddings.py
   	source activate vis_vqa_env
   	python model_prepro_text.py --method='bert'
	python test_prepro_text.py --method='bert'

6. variation = isq/iq/sq/q  , offline_text = True/False
	python model_viscomp_vqa.py --variation --offline_text

7. model_save_offline/model_save_isq/model-1
	python test_viscomp_vqa.py --variation --offline_text --model_path model_save_offline/model_save_isq/model-1
