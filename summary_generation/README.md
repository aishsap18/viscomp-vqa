
# Summary Generation

This code is based on the Pytorch implementation of [NMT Sequence-to-Sequence model](https://github.com/tensorflow/nmt). The input is a sequence of images and a natural language story based on which a concise summary is generated. 

### Installation 

1. Clone the repository and navigate to "summary_generation"
```
git clone https://github.com/aishsap18/viscomp-vqa
cd summary_generation
```

2. Create the conda environment using the provided `.yml` file
```
conda env create -f environment.yml
```

3. Activate the environment
```
conda activate summary_generator_env
```

4. Use `requirements.txt` file if pip packages do get installed
```
pip install -r requirements.txt
```


### Steps to execute

1. Navigate to the `data` directory in the root and run the following command
```
cd ../data/
python vqa_preprocessing.py --train_questions annotations/summary/train_input_summary_1900.json --test_questions annotations/summary/val_input_summary_603.json --train_annotations annotations/summary/train_annotations_summary_1900.json --test_annotations annotations/summary/val_annotations_summary_603.json --variation isq
```
For generating summaries, use the `isq` variation stating `Images+Story`. `q` (Question) come in handy when we experiment generating summary without story. 
This code will generate 2 files in the `data` directory, `vqa_raw_train.json` and `vqa_raw_test.json`.

2. Navigate to `summary_generation` directory and run the following command
```
python prepro_data.py --train_input data/vqa_raw_train.json --test_input data/vqa_raw_test.json --output_json data_prepro.json --output_bert_h5 data_bert_emb.h5 --variation isq
```
Here, the variations can be `isq - Images+Story`, `iq - Images+Question`, `sq - Story`, and if we wish to use BERT embeddings then prepend the former variations with `b`: `bisq - Images+BERT(Story)` and `bsq - BERT(Story)`. 
This code will generate `data_prepro.json` containing the cleaned and pre-processed data. If the variation requires BERT embeddings then it will generate an additional `data_bert_emb.h5` containing BERT embeddings of the story and question tokens.  

3. Extracting the images features 
	- Download the pretrained VGGNet 19 layer model from [https://gist.github.com/ksimonyan/3785162f95cd2d5fee77](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)
	- Change the conda environment to `vis_vqa_env`.
	- Download the images 
	```
	python ../data/download_images.py --input_json data_prepro.json --images_root [path to save images]
	```
	- Execute the following command for extracting the image features
		```
		python prepro_img.py --input_json data_prepro.json --image_root [path to images directory] --cnn_proto [path to cnn prototxt] --cnn_model [path to cnn model]
		```
		This will generate the `data_img.h5` file.

4. Train the model by executing 
```
python batch_train_pytorch.py --input_data_file data_prepro.json --input_img_file data_img.h5 --model_save [checkpoint path] --variation isq --input_bert_emb data_bert_emb.h5
```
Here, the variations can be `isq, iq, sq, bsq, bisq (b - bert embeddings), gisq, gsq (g - glove embeddings)`. 

5. Evaluate the model by executing 
```
python batch_test_pytorch.py --input_data_file data_prepro.json --input_img_file data_img.h5 --checkpoint_path [path to checkpoint model] --results_path [path to directory for saving results] --variation isq --input_bert_emb data_bert_emb.h5
```
This will generate `results.json` file which will contain the results.
