
# Answer Prediction

This code is based on the [Tensorflow implementation of Deeper LSTM Q + norm I model for VQA](https://github.com/chingyaoc/VQA-tensorflow). The input is a sequence of images, a natural language story and question for which answer is predicted from a predefined set of K answers.  

### Installation 

1. Clone the repository and navigate to "k-class_classification"
```
git clone https://github.com/aishsap18/viscomp-vqa
cd k-class_classification
```

2. Create the conda environment using the provided `.yml` file
```
conda env create -f environment.yml
```

3. Activate the environment
```
conda activate k-class_env
```

### Steps to execute

1. Navigate to the `data` directory in the root and run the following command
```
cd ../data/
python vqa_preprocessing.py --train_questions annotations/train_questions.json --test_questions annotations/val_questions.json --train_annotations annotations/train_annotations.json --test_annotations annotations/val_annotations.json --variation isq
```
For predicting answers, use the `isq` variation stating `Images+Story+Question` input.
This code will generate 2 files in the `data` directory, `vqa_raw_train.json` and `vqa_raw_test.json`.

2. Navigate to `k-class_classification` directory and run the following command
```
python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans [number of answers]
```
This will generate 2 files data_prepro.h5 and data_prepro.json containing pre-processed data.

3. Extracting the images features 
	- Download the pretrained VGGNet 19 layer model from [https://gist.github.com/ksimonyan/3785162f95cd2d5fee77](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77)
	- Download the images from [https://drive.google.com/open?id=1GKyFDTcOvxy7XXxyNhkHTs6WpCsBnKoy](https://drive.google.com/open?id=1GKyFDTcOvxy7XXxyNhkHTs6WpCsBnKoy)
	- Execute the following command for extracting the image features
		```
		python prepro_img.py --input_json data_prepro.json --image_root [path to images directory] --cnn_proto [path to cnn prototxt] --cnn_model [path to cnn model]
		```
		This will generate the `data_img.h5` file.

4. Train the model by executing
```
python model_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_save [checkpoint path] --num_ans [number of answers]
```

5. Evaluate the model by executing 
```
python test_viscomp_vqa.py --input_data_h5 data_prepro.h5 --input_data_json data_prepro.json --input_img_h5 data_img.h5 --model_save [path to checkpoint model] --results_path [path to save results]
```
