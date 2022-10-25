# Machine Learning in Computational Biology Exercises

This repository contains exercises for the machine learning part of the IN-BIOS5000/IN-BIOS9000 course at UiO. It can be run online using Binder or Google Colab. Links to open specific notebooks in either of these environments are given below.

## Exercise 1: Transcription Factor Binding Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uio-bmi/machine_learning_in_comp_bio_exercises/blob/main/Exercise_1.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uio-bmi/machine_learning_in_comp_bio_exercises/main?labpath=Exercise_1.ipnyb)

In this exercise, we will have a dataset consisting of DNA sequences which are labeled 0 or 1 if the transcription factor binds to them or 
not, respectively. 

We will run Exercise_1.ipnyb notebook to train the models and then examine the results to try to understand how the models work.

The dataset for this exercise was downloaded from https://github.com/QData/DeepMotif.

## Exercise 2: Transcription Factor Binding Prediction - selecting hyperparameters

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uio-bmi/machine_learning_in_comp_bio_exercises/blob/main/Exercise_2.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uio-bmi/machine_learning_in_comp_bio_exercises/main?labpath=Exercise_1.ipnyb)

In this exercise, we will have the same dataset as in Exercise 1 with the same aim of building a good predictive model. To that aim, 
we will include cross validation (CV) to explore different hyperparameters and models. The exercise is in the Exercise_2.ipnyb notebook.

## Exercise 3: predicting disease states from adaptive immune receptor repertoires

Adaptive immune receptors bind to antigens in the body (such as parts of viruses or bacteria) and help neutralize the threat. In the adaptive
immune receptor repertoire that includes all receptors in the body, there are approximately 10^8 unique receptors that mostly recognize 
different threats. By using machine learning, it might be possible to predict if a person has a given immune-related disease from their repertoire data.

In this exercise, we will use a public immuneML Galaxy tool to build an ML model that will be able to classify between repertoires coming from
healthy and diseased individuals.

Steps:

1. Go to https://galaxy.immuneml.uiocloud.no
2. Select the shared history from the top menu: Shared Data -> Histories -> Quickstart Data
3. Click on the plus sign in the top right corner to import history: the data will then be shown in the right sidebar and can be examined by clicking on the eye icon
4. From the menu on the left, select immuneML tools -> Create immuneML dataset tool, and provide the data to required fields and click on the Execute button to create the dataset: the dataset will show up in the right sidebar and will turn green when the tool has finished the execution
5. From the menu on the left, select immuneML tools -> Train immune repertoire classifiers (simplified interface) tool and fill in the parameters from the suggested list in the tool: when the tool has finished the execution click on the eye icon of the Summary: repertoire classification element to examine the results.

The results show the performance of algorithms and encodings selected in step 5 using nested cross validation. Look into the results in both the inner cross validation (selection) and the outer (assessment) and compare them.
