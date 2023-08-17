# Driver Drowsiness Classification

## Project Overview:
This repository hosts the source code, dataset, and resources for a cutting-edge project focused on detecting fake or manipulated faces using deep learning techniques. With the rise of digital manipulation tools, it has become increasingly important to develop robust methods for identifying fake images, especially in contexts such as social media and digital content verification.

## Dataset
The dataset belongs to the kaggle in this link. I splitted 70/30 for validation and used filp horizontal and rotation augmentation.

<br/>


## Model
The model that I used was combination of transformers and resnet34 backbone.

<br/>

## Training 
For the training i used AdamW optimizer with 0.01 learning rate and weight decay: 0.001.

<br/>

## Results
The result of the training was very good with fast convergence.

### Training accuracy
![](logs/train_accuracy.svg)

### Training loss
![](logs/train_loss.svg)

### validation accuracy
![](logs/validation_accuracy.svg)

### validation loss
![](logs/validation_loss.svg)

