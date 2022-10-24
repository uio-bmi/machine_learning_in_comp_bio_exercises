# Machine Learning in Computational Biology Exercises

This repository contains exercises for the machine learning part of the IN-BIOS5000/IN-BIOS9000 course at UiO. It can be run online using Binder or Google Colab. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uio-bmi/machine_learning_in_comp_bio_exercises/HEAD)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uio-bmi/machine_learning_in_comp_bio_exercises/Exercise_1.ipnyb)


## Exercise 1: Transcription Factor Binding Prediction

In this exercise, we will have a dataset consisting of DNA sequences which are labeled 0 or 1 if the transcription factor binds to them or 
not, respectively. 

We will run Exercise_1.ipnyb notebook to train the models and then examine the results to try to understand how the models work.

The dataset for this exercise was downloaded from https://github.com/QData/DeepMotif.

## Exercise 2: Transcription Factor Binding Prediction - selecting hyperparameters

In this exercise, we will have the same dataset as in Exercise 1 with the same aim of building a good predictive model. To that aim, 
we will include cross validation (CV) to explore different hyperparameters and models. The exercise is in the Exercise_2.ipnyb notebook.

## Exercise 3: predicting disease states from adaptive immune receptor repertoires

Adaptive immune receptors bind to antigens in the body (such as parts of viruses or bacteria) and help neutralize the threat. In the adaptive
immune receptor repertoire that includes all receptors in the body, there are approximately 10^8 unique receptors that mostly recognize 
different threats. By using machine learning, it might be possible to predict if a person has a given immune-related disease from their repertoire data.

In this exercise, we will use a public immuneML Galaxy tool to build an ML model that will be able to classify between repertoires coming from
healthy and diseased individuals.
