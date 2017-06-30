# NeuroBind--Yet Another Model for Finding Binding Sites Using Neural Networks

## Motivation

## Model Description
Prenet
Conv1d Banks
Position-wise convolutions
Hightway nets
Final Fully Connected

## Data
The UniProbe PBM data are used for training and evaluation.
My dataset scheme follows that of [DeeperBind: Enhancing Prediction of Sequence Specificities of DNA Binding Proteins](https://arxiv.org/pdf/1611.05777.pdf).

## Requirements
 * numpy >= 1.11.1
 * TensorFlow >= 1.2
 * scipy == 0.19.0
 * tqdm >= 4.14.0


## File description

 * `hyperparams.py` includes all hyper parameters that are needed.
 * `data_load.py` loads data and put them in queues.
 * `modules.py` contains building blocks for the network.
 * `train.py` is for training.
 * `eval.py` is for validation and test evaluation.

## Training
  * STEP 0. Make sure you meet the requirements.
  * STEP 1. Adjust hyper parameters in hyperparams.py if necessary.
  * STEP 2. Run `train.py` or download my [pretrained files](https://u42868014.dl.dropboxusercontent.com/u/42868014/neurobind/log.zip).

## Validation Check
  * Run `eval.py val` to find the best model.

## Test evaluation
  * Run `eval.py test` to get the test results for the final model.

## Results
I got a Spearman rank correlation coefficients of 0.60 on array #2 of CEH-22.

| TF | PRG | RKM | S&W | KHM | DBD | DEBD | NB (Proposed) |
|--|--|--|--|--|--|--|--|
| CEH-22 | 0.28 | 0.43 | 0.28 | 0.31 | 0.40 | 0.43 | **0.60**|





