# Transcription Factor Binding Prediction

In this exercise, we will use a few Python scripts to train and assess the performance of machine learning models that predict if a DNA sequence
contains a transcription factor binding site.

Transcription factors (TFs) are proteins that bind to DNA and influence gene regulation. Predicting if they will bind or not could help us understand the biology better
and allow us to preding binding for new DNA sequences which were not experimentally analyzed.

## Get the data

We will use three different datasets for three transcription factors (USF1, USF2 and YY1). 
For each transcription factor, the data includes train.fa and test.fa files. They consist of sequences and a label for each sequence 
(1 - TF will bind to the sequence, 0 - it will not bind).

The data can be downloaded from this repository (under data folder) and it is also available on the server used for the course.

## Get the code

The code to train and test a ML model is available in this repository under scripts. You will need all files from that folder.

The files are the following:

1. [`train_test_log_reg_kmer.py`](https://raw.githubusercontent.com/uio-bmi/machine_learning_in_comp_bio_exercises/main/scripts/train_test_log_reg_kmer.py): this script will take the data, encode it as k-mer frequencies, train the model on train.fa and test it on test.fa file.
It will also produce a few figures that can be useful to examine the trained model.

2. [`train_test_log_reg_onehot.py`](https://raw.githubusercontent.com/uio-bmi/machine_learning_in_comp_bio_exercises/main/scripts/train_test_log_reg_onehot.py): this script will take the data, encode it using one-hot encoding, train the logistic regression model on train.fa and test
it on test.fa file, again producing some useful figures.

3. [`util.py`](https://raw.githubusercontent.com/uio-bmi/machine_learning_in_comp_bio_exercises/main/scripts/util.py): this is where the utility code is located. If you want to extend this example and try another machine learning model (e.g., random forest or any scikit-learn classifier), you can
add a new function to this util file similar to `train_and_assess_logistic_regression(...)` and see in the other two files how this function is used.

### Running locally

To run this code locally, you will need the following:

1. python 3 (tested with python 3.7 and 3.8)
2. numpy 1.19.2 
3. pandas 1.1.3
4. scikit-learn 0.23.2
5. matplotlib 3.3.2

Previous version of the libraries should also work, but have not been tested.

## Run the analysis for transcription factor USF1

Make sure that the scripts and data are downloaded to the machine you are using.

### Task 1: create a logistic regression model to classify sequences to binders vs. non-binders using k-mer frequencies as encoding

To run the script, we need to specify the path to training data, the path to test data, result path and the length of the subsequence k (make sure that the paths are correct for your setting):

```commandline

python train_test_log_reg_kmer.py --train_data USF1_K562_USF-1_HudsonAlpha/train.fa --test_data USF1_K562_USF-1_HudsonAlpha/test.fa --k 5 --result_path ./result_5mer/

```

The output will look similar to this:

```commandline

Train dataset: 36042 examples, 18021 positive and 18021 negative ones.
Train dataset: 1000 examples, 500 positive and 500 negative ones.
Encoding data with k-mer frequencies started.
Encoding data with k-mer frequencies finished.
Training logistic regression model started.
Training logistic regression model finished.
Logistic regression achieved accuracy score: 0.831 on training set.
Logistic regression achieved accuracy score: 0.816 on test set.

Process finished with exit code 0

```

In addition to this information, in the resulting folder, there will be a few figures:

1. `k_mer_freqs_train.jpg`: showing the frequency of k-mers in positive examples vs. frequency of k-mers on negative examples on the training dataset.
2. `k_mer_freqs_test.jpg`: the same as previous one, on the test dataset.

    Questions: 
    
    - Are there any differences? 
    - What do we expect to see given that the accuracy of the model was high?

3. `confusion_matrix.jpg`: confusion matrix shows how many examples (sequences in our case) were correctly classified as binding (true positives), 
wrongly classified as binding (false positives), correctly classified non-binding (true negatives) and wrongly classified as non-binding (false negatives).

    Questions: 
        
    - How good is our classifier? 
    - What can we see from the confusion matrix?
    
4. `top_n_coefficients.jpg`: logistic regression has a coefficient for each feature (each k-mer frequency). This plot shows 10 coefficients with highest absolute
value from the model. Those values contribute the most to the prediction.

    Questions: 
    
    - What does this mean for our model? 
    - What k-mers is it detecting? 
    - Can we interpret this somehow?
    
Try running the same command with a different value of `k`. How does this change results? What do we assume by setting the value of `k` to specific value?

Note that the script will run faster for smaller values of `k`.

### Task 2: create a logistic regression model to classify sequences as binding vs. non-binding using one-hot encoding

To run the script, it is necessary to specify the path to training and test data and result path.

```commandline

python train_test_log_reg_onehot.py --train_data USF1_K562_USF-1_HudsonAlpha/train.fa --test_data USF1_K562_USF-1_HudsonAlpha/test.fa --k 5 --result_path ./result_one_hot/

```

The output should look like this:

```commandline

Train dataset: 36042 examples, 18021 positive and 18021 negative ones.
Train dataset: 1000 examples, 500 positive and 500 negative ones.
One-hot encoding of data started.
One-hot encoding of data finished.
Training logistic regression model started.
Training logistic regression model finished.
Logistic regression achieved accuracy score: 0.606 on training set.
Logistic regression achieved accuracy score: 0.649 on test set.

Process finished with exit code 0

```

Confusion matrix and top 10 coefficients plots are created now as well. 

Questions: 

- Which encoding is better? Why? 
- Are there any similarities between top 10 coefficients? 
- What is the generalization accuracy that we can expect for the problem?
- Can we conclude anything about sequences binding to the transcription factor? Do we have enough information? 
- What assumption we make when we choose one encoding or the other?

You can try this out with different TFs (there are three datasets in this repository, you can just change the paths to training and test data to use a different one).

### Programming Task 3 (optional): try to use random forest classifier instead of logistic regression

Look into the plots and performance. Does performance change?
