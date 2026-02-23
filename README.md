The repository contains two modules for fine-tuning machine translation models on multilingual datasets: preprocess and train.
To run the modules, you need to create a config.yaml file.

## config

The config file specifies the location of the pretrained model, the training and validation datasets, and all parameters required for the training and preprocessing steps. Below are the main sections:
### `model`
- `name`: Hugging Face ID of the model to be fine-tuned
### `datasets`
- `train`: Details of the training dataset (Hugging Face ID, split, and columns)
- `validation`: Details of the validation dataset
### `tokenization`
 - Tokenizer arguments such as `max_length`, `padding`, and `truncation`
### `training`
 - 	Training arguments such as the number of epochs, learning rate, and batch size.


## Preprocess
This module tokenizes the training and validation datasets and saves them to disk. To tokenize the datasets, run:

 ``` python preprocess.py --config configs/nllb-200-600m-full-dataset-finetune.yaml ```

The tokenized dataset is saved to the output directory specified in the config file.

## train 
This module runs the fine-tuning of the baseline model defined in the config file. To start training, you need to pass the config file and the location of the dataset tokenized during the preprocessing step.

``` python train.py --config configs/nllb-200-600m-full-dataset-finetune.yaml --tokenized_dataset tokneized/ ```


## Installation
``` pip install -r requirements.txt ```


## Citation

This repository is part of the [AfriNLLB](https://github.com/AfriNLP/AfriNLLB) project. 
If you use any part of the project's code, data, models, or approaches, please cite the following paper:

```
@inproceedings{moslem-etal-2026-afrinllb,
    title = "{A}fri{NLLB}: Efficient Translation Models for African Languages",
    author = "Moslem, Yasmin  and
      Wassie, Aman Kassahun  and
      Gizachew, Amanuel",
    booktitle = "Proceedings of the Seventh Workshop on African Natural Language Processing (AfricaNLP)",
    month = jul,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
}
```






