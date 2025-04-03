# Melody Generator

## Overview
This project is a melody generation system using deep learning. It processes musical data, trains a model, and generates melodies based on a predefined seed. {Forked from https://github.com/musikalkemist/generating-melodies-with-rnn-lstm }


## Setup
### Prerequisites
Ensure you have Python installed along with the necessary dependencies:
```sh
pip install -r requirements.txt
```

## Usage

### Step 1: Preprocessing Data (Optional - Already done by me)
Run the following command to preprocess the dataset:
```sh
python preprocess.py
```
This will generate the `dataset/` directory using data from `main_dataset/`.

### Step 2: Training the Model (Optional)
To train the model from scratch, execute:
```sh
python train.py
```
This will create a trained model saved as `model.h5`.

### Step 3: Generating Melody
Use the trained model to generate melodies:
```sh
python melody_generator.py
```
The melody will be generated using a predefined seed embedded in the code.

### Web UI: Generating Melody
Use the index.html page to generate melody in a more easy way:
```sh
python app.py
```
Open the localhost website appearing in the terminal

## Notes
- You can either use the provided `model.h5` or generate your own by training.
- The melody generation process depends on the pre-trained model and the initial seed.


