
# Visual Comprehension Baselines

The directory `summary_generation/` contains NMT Sequence-to-Sequence model for generating summary.

The directory `vqa_k-class_classification/` contains Deeper LSTM Q + norm I model for predicting answers using K-class classification approach.

The directory `vqa_5-class_classification/` contains Deeper LSTM Q + norm I model for predicting multiple choice answers using 5-class classification approach.


### Installation 

1. Clone the repository
```
git clone https://github.com/aishsap18/viscomp-vqa
```

2. Create the conda environment using the provided `environment.yml` file 
```
conda env create -f environment.yml
```

3. Activate the environment
```
conda activate vis_vqa_env
```

4. Use `requirements.txt` file if pip packages do get installed
```
pip install -r requirements.txt
```


#### Execution steps for each approach are provided in their respective directories.
