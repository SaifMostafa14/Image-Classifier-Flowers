# Image-Classifier-Flowers

This project trains a neural network to recognize 102 species of flowers using PyTorch.  It was originally built as part of the **AI Programming with Python Nanodegree**.

The repository contains:

- `Image Classifier Project.ipynb` &ndash; the development notebook.
- `train.py` &ndash; command line script to train a classifier.
- `predict.py` &ndash; use a trained model to predict flower names.
- `flowers/` &ndash; dataset organized into `train`, `valid`, and `test` directories.

## Requirements

- Python 3
- [PyTorch](https://pytorch.org/) and torchvision
- numpy, pillow, matplotlib

A GPU is optional but recommended for faster training.

## Training a Model

Run `train.py` pointing to the dataset directory.  The script accepts several optional arguments:

```bash
python train.py <data_dir> \
    --save_dir checkpoint.pth \
    --arch vgg16 \
    --learning_rate 0.001 \
    --hidden_units 120 \
    --dropout 0.5 \
    --epochs 1 \
    --gpu
```

`data_dir` should contain the `train`, `valid`, and `test` subfolders.  After training, a checkpoint file is written to `--save_dir`.

## Making Predictions

Use `predict.py` to classify an image with a trained model:

```bash
python predict.py <image_path> <checkpoint> \
    --top_k 5 \
    --category_names cat_to_name.json \
    --gpu
```

The script prints the top predicted flower names with their probabilities.

## Notebook

The Jupyter notebook `Image Classifier Project.ipynb` demonstrates the model training and prediction process step by step.

