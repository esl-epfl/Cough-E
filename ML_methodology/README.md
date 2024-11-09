# Cough Counting Algorithm Development

This project aims to develop an efficient edge-AI model for noninvasively monitoring patients' coughing patterns throughout the day using a wearable device.

## Getting started

#### Accessing data

The training dataset can be downloaded from [this Zenodo link](https://zenodo.org/records/7562332). Download it and then update the `data_folder` parameter in `config/config.yaml` with the path in your machine. Note: The "talking" data scenario is not publicly available for subject privacy concerns. To access this data, please contact the authors of the [Cough-E publication](https://arxiv.org/abs/2410.24066).

If you wish to augment the dataset in a semi-supervised fashion with additional coughs, the [COUGHVID dataset](https://zenodo.org/records/7024894) can be used. Download it and then update the `coughvid_folder` and `coughvid_metadata` parameters (the latter is the location of the metadata_compiled.csv file of the COUGHVID dataset) in `config/config.yaml` with the paths in your machine.

To get access to the private testing dataset, contact the authors of the [Cough-E publication](https://arxiv.org/abs/2410.24066). Once you get access to the testing data, update the `test_data_folder` parameter in `config/config.yaml` with the path in your machine.

#### Setting up the environment

Install Miniconda on the server and create the "cough" environment:

```
conda env create -f environment.yml
```

The environment should be activated before running any code:

```
conda activate cough
```

## Project structure

The `train_cross_validate.py` executable is the main function that runs the Maching Learning (ML) workflow. 

The parameters used to configure the model training and testing (i.e. signal selection, evaluation setup) are stored in the `config` folder. The [Hydra package](https://hydra.cc/docs/intro/) is used to link these parameters to the code and automatically save the outputs of the runs in the `outputs/` directory. It saves each run in a timestamped folder with a `.hydra` file that lists which exact parameters were used to genrate the results.

The `extract_features.py` executable, called in `run.py`, saves its outputs to the `intermediates/` folder so they can be re-used in multiple experiment configurations.

The `src/` directory contains important functions and data types for performing signal processing, feature extraction, and efficiently locating desired files within the dataset.

## Workflow overview

#### Implementing features to extract

The features are implemented in a call-tree structure in the `src/feature_tree.py` file. New features can be added by implementing them in this file. Better documentation of this file is in progress.

The hyperparameters of the feature extraction are configured in a `.yaml` file in the `config/feature_extraction` directory, which is then loaded into the executable with Hydra.

#### Configuring the ML flow

Running the full flow can be done by running:
```python
python run.py ml=run_all
```
which loads the `config/ml/run_all.yaml` file's parameters into the script, thus commanding it to execute every step of the ML model development and validation workflow.

The ML flow features the following steps:
* **Feature extraction:** Segment the signals in the dataset, extract the features from each segment, and save them to a file in the `intermediates/` folder. Outputs a `.npy` file with all of the features.
* **Model selection:** Performs a nested cross-validation loop to select the ML model with the best performance (ex. Random Forest, XGB) in terms of average precision. Outputs a `.csv` fole in the `outputs/` directory of the run listing the mean and standard deviation CV score of each model.
* **Recursive Feature Elimination with Cross-Validation (RFECV):** Plots the average precision of the model versus the number of input features to determine the optimal number of features to keep. Outputs a `.csv` fole in the `outputs/` directory of the run listing the CV score for each feature number.
* **Feature Selection:** Runs a nested CV loop to determine which features, on average, contribute most to the model outcome. Saves them to a `.pkl` file in the `outputs/` directory.
* **Training:** Train the model using the final parameters. Optionally pre-train the audio network with the [COUGHVID dataset](https://www.nature.com/articles/s41597-021-00937-4). Saves the model to a `.pkl` file in the outputs directory.
* **Testing:** Emulate the model running through each test subject's recording in an edge-AI fashion. Saves the results of each fold to `.csv` files in the `outputs/` directory.

Some parts of the flow take a long time to execute (ex. the feature extraction with COUGHVID pre-training takes about 4 hours for the audio signals). Therefore, the `.yaml` files in the `config/ml/` directory allow you to skip parts of the flow that have already been executed and run only the later ones with different parameters. 

If you have already extracted the features of the signals for a set overlap and window length, you can load these features from the `intermediates/` directory and use them for further processing by running:
```python
python run.py ml=already_extracted_feats
```
which uses the parameters defined in `config/ml/already_extracted_feats.yaml`.

Let's say you run the rest of the flow and observe that the XGB model is always the best one across all CV folds regardless of the configuration. You can run this flow to use only that model:
```python
python run.py ml=rfecv_onward ml.model_name="XGB"
```
which starts running from the RFECV step. Optionally, you can go to the `config/ml/rfecv_onward.yaml` file and hardcode the "model_name" parameter to "XGB".

Now, let's imagine we want to always use 50 features, because anything more than that makes the model's execution time explode on an embedded platform. You can run:
```python
python run.py ml=feat_select_onward ml.n_features_opt=50
```

There are many more possible run configurations that run different staps of the pipeline. By either creating new YAML file in `config/ml/` or changing the parameters in the command line, you can customize the workflow to fit your needs and save time by reusing previous data and skipping steps.

### Hyperparameter setup using Hydra

The `config/` directory organizes the hyperparameters into groups depending on their function. Each group is a folder that contains different common configurations of the same hyperparameters. The different groups are:
* **Signals:** Select which signals to use in the model training (i.e. audio, IMU), window length and overlap for the signal segmentation, and audio downsampling frequency
* **Feature Extraction:** Parameters related to the feature extraction implementation (ex. power spectral density bands, whether to use MFCCs or the "raw" mel spectrogram components)
* **ML:** Choose which steps of the ML pipeline to run, as well as configuration parameters of the steps (ex. the step size of RFECV, which models to test for model selection, etc.)
* **Experiment Conditions:** Which signals to use for training and testing based on experimental conditions (i.e. laughing, coughing, sitting, etc.)
* **Postprocessing:** Parameters used to post-process audio signals to delineate individual cough events
* **Scoring:** Parameters setting up the event-based scoring of the signals using the [timescoring library](https://github.com/esl-epfl/epilepsy_performance_metrics)

### Quantifying model configuration success

Once the models have been tested, you can analyze the results by running
```python
python report_results.py
```
which reports the duration-based and event-based accuracy metrics for the optimal classification threshold across the CV folds.

Please note that this is NOT the final testing accuracy, and just indicates the success of the model training setup with the chosen hyperparameters.