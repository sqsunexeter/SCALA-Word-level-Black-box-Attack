# SCALA-Word-level-Black-box-Attack
### An Efficient Word-level Black-box Adversarial Attack against Textual Models

This repository contains source code for the Code for the IJCNN 2023 submission: 

#### Instructions for running the attack from this repository.

#### Requirements
-  Numpy == 1.19.5  
-  Pytorch == 1.11.0    
-  Python >= 3.6
-  Tensorflow == 1.15.2
-  TensorflowHub == 0.11.0 
-  textattack == 0.3.3

#### Download Dependencies

- Download pretrained target models for each dataset [bert](https://drive.google.com/file/d/1UChkyjrSJAVBpb3DcPwDhZUE4FuL0J25/view?usp=sharing), [lstm](https://drive.google.com/drive/folders/1nnf3wrYBrSt6F3Ms10wsDTTGFodrRFEW?usp=sharing), [cnn](https://drive.google.com/drive/folders/149Y5R6GIGDpBIaJhgG8rRaOslM21aA0Q?usp=sharing) unzip it.

- Download the counter-fitted-vectors from [here](https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view) and place it in the main directory.

- Download top 50 synonym file from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view) and place it in the main directory.

- Download the glove 200 dimensional vectors from [here](https://nlp.stanford.edu/projects/glove/) unzip it.
 
#### How to Run:

Use the following command to get the results. 

For BERT model

```
python classification_attack.py \
        --target_model Type_of_taget_model (bert,cnn,lstm) \
        --target_dataset Dataset Name (mr, imdb, yelp, ag, snli, mnli)\
        --target_model_path pre-trained target model path \
        --dataset_dir directory_to_data_samples_to_attack  \
        --output_dir  directory_to_save_results \
        --word_embeddings_path path_to_embeddings \
        --counter_fitting_cos_sim_path path_to_synonym_file \
        --nclasses  how many classes for classification


```
Example of attacking BERT on IMDB dataset.

```

python3 classification_attack.py \
        --target_model bert \
        --target_dataset imdb \
        --target_model_path pretrained_models/bert/imdb \
        --dataset_dir data/ \
        --output_dir  final_results/ \
        --word_embeddings_path embedding/glove.6B.200d.txt \
        --counter_fitting_cos_sim_path counter-fitted-vectors.txt \
        --nclasses 2 \


```

Example of attacking BERT on SNLI dataset. 

```

python3 entailment.py \
        --target_model bert \
        --target_dataset snli \
        --target_model_path pretrained_models/bert/snli \
        --dataset_dir data/ \
        --output_dir  final_results/ \
        --word_embeddings_path embedding/glove.6B.200d.txt \
        --counter_fitting_cos_sim_path counter-fitted-vectors.txt \


```
#### Results
The results will be available in **final_results/classification/** directory for classification task and in **final_results/entailment/** for entailment tasks.
For attacking other target models look at the ```commands``` folder.

#### Training target models
To train BERT on a particular dataset use the commands provided in the `BERT` directory. For training LSTM and CNN models run the `train_classifier.py --<model_name> --<dataset>`.
