
# Answer Prediction

This code is based on the [Tensorflow implementation of Deeper LSTM Q + norm I model for VQA](https://github.com/chingyaoc/VQA-tensorflow). The input is a sequence of images, a natural language story and question for which answer is predicted from a set of 5 options.  


### Steps to execute

1. Navigate to the `data` directory in the root and run the following command
```
cd ../data/
python vqa_preprocessing.py --train_questions annotations/complex_questions/train_questions_8100.json --test_questions annotations/complex_questions/val_questions_2154.json --train_annotations annotations/complex_questions/train_annotations_8100.json --test_annotations annotations/complex_questions/val_annotations_2154.json
```
This code will generate 2 files in the `data` directory, `vqa_raw_train.json` and `vqa_raw_test.json`.
For giving summary based questions as input, provide the train and test JSON files from `annotations/summary_qa` directory instead of `annotations/complex_questions`.

2. Navigate to `vqa_5-class_classification` directory and run the following command
```
cd ../vqa_5-class_classification/
python prepro.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --multiple_choice true
```
This will generate 2 files `data_prepro.h5` and `data_prepro.json` containing pre-processed data.

3. Extracting the images features 
	- Download the pretrained VGGNet 19 layer model from [https://gist.github.com/ksimonyan/3785162f95cd2d5fee77](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)
	- Create a directory and download the images in it 
		```
		python ../data/download_images.py --input_json data_prepro.json --images_root [path to save images]
		```
		for example,
		```
		mkdir images
		python ../data/download_images.py --input_json data_prepro.json --images_root images/
		```
	- Execute the following command for extracting the image features
		```
		python prepro_img.py --input_json data_prepro.json --image_root [path to images directory] --cnn_proto [path to cnn prototxt] --cnn_model [path to cnn model]
		```
		for example,
		```
		python prepro_img.py --input_json data_prepro.json --image_root images/ --cnn_proto VGG_ILSVRC_19_layers_deploy.prototxt --cnn_model VGG_ILSVRC_19_layers.caffemodel
		```
		This will generate the `data_img.h5` file.

4. For training stories offline  
	- Using LSTM 
	```
	python model_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_path [checkpoint path]
	python test_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_model [path to checkpoint model]
	```
	for example, create a directory for saving checkpoints and execute the above commands
	```
	mkdir checkpoints_lstm
	python model_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_path checkpoints_lstm/
	python test_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_model checkpoints_lstm/model-0
	```
	This will generate the `data_text_train.h5` file containing trained story embeddings using LSTM. 

	- Using BERT
	```
	python get_bert_embeddings.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --out_name data_text_bert.h5
	python model_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_path [checkpoint path] --method bert --input_bert_emb data_text_bert.h5
	python test_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_model [path to checkpoint model] --method bert --input_bert_emb data_text_bert.h5
	```
	for example,
	```
	python get_bert_embeddings.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --out_name data_text_bert.h5
	mkdir checkpoints_bert
	python model_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_path checkpoints_bert/ --method bert --input_bert_emb data_text_bert.h5
	python test_prepro_text.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --checkpoint_model checkpoints_bert/model-0 --method bert --input_bert_emb data_text_bert.h5
	```
	This will generate the `data_text_train.h5` file containing trained story embeddings using BERT.

5. Create a directory for saving checkpoints and train the model by executing
```
python model_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_path [checkpoint path] --variation [isq/iq/sq] --offline_text [True/False] --input_text_h5 data_text_train.h5
```
for example,
```
mkdir checkpoints
python model_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_path checkpoints/ --variation isq --offline_text True --input_text_h5 data_text_train.h5
```
Here, the input `variation` can be 'isq - Images+Story+Question, iq - Images+Question or sq - Story+Question'. The `offline_text` parameter states if the stories are trained offline (True). The `input_text_h5` parameter is optional, it has to be stated only if stories are trained offline. 

6. Create a directory for saving results and evaluate the model by executing 
```
python test_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_path [path to checkpoint model] --variation [isq/iq/sq] --offline_text [True/False] --input_text_h5 data_text_train.h5 --result_path [path to save results]
```
for example,
```
mkdir results
python test_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_path checkpoints/model-0 --variation isq --offline_text True --input_text_h5 data_text_train.h5 --result_path results/
```
