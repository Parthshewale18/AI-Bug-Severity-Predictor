# Bug Severity DNN Project

This project trains a deep neural network to classify software bug reports by
severity. It uses bug report text as input, cleans the text, converts it into
TF-IDF features, and trains a Keras/TensorFlow feed-forward neural network to
predict one of five severity classes.

## Severity Classes

The project focuses on actual bug severity labels:

- `blocker`
- `critical`
- `major`
- `minor`
- `trivial`

The original dataset also contains a `normal` class, but the notebook workflow
filters it out so the model focuses on bug severity prediction.

## Project Structure

```text
DNN_project/
|-- data/
|   |-- bug_reports_filtered.csv
|   `-- bug_reports_clean.csv
|-- models/
|   |-- bug_severity_dnn.keras
|   `-- label_encoder.pkl
|-- notebooks/
|   |-- data.ipynb
|   `-- train.ipynb
|-- src/
|   |-- preprocessing.py
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## What Each File Does

- `notebooks/data.ipynb` loads the Bugzilla Eclipse bug report dataset from
  Hugging Face, explores the severity distribution, filters the target classes,
  selects the required columns, and creates the filtered CSV.
- `src/preprocessing.py` cleans raw bug report text and saves
  `data/bug_reports_clean.csv`.
- `notebooks/train.ipynb` shows the model training workflow step by step.
- `src/train.py` trains the DNN model from the cleaned dataset.
- `models/bug_severity_dnn.keras` is the saved trained Keras model.
- `models/label_encoder.pkl` stores the fitted severity label encoder.
- `requirements.txt` lists the Python packages needed for data preparation,
  training, notebooks, and app development.

## Dataset

The notebook uses the Hugging Face dataset:

```python
AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset
```

The processed project data is stored in:

- `data/bug_reports_filtered.csv`
  - columns: `text`, `severity`
- `data/bug_reports_clean.csv`
  - columns: `text`, `severity`, `clean_text`

The `clean_text` column is produced by:

- converting text to lowercase
- removing URLs
- removing HTML tags
- removing punctuation and special characters
- normalizing extra spaces

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

## How to Run

### 1. Prepare the Dataset

If `data/bug_reports_filtered.csv` already exists, you can skip this step.

To recreate it from the original Hugging Face dataset, open and run:

```text
notebooks/data.ipynb
```

### 2. Clean the Text Data

From the project root, run:

```powershell
cd src
python preprocessing.py
```

This creates or updates:

```text
data/bug_reports_clean.csv
```

### 3. Train the Model

From the `src` directory, run:

```powershell
python train.py
```

The training script will:

1. Load `data/bug_reports_clean.csv`
2. Drop missing rows
3. Encode severity labels with `LabelEncoder`
4. Convert cleaned text to TF-IDF features
5. Split the data into training and test sets
6. Compute balanced class weights
7. Train a TensorFlow/Keras DNN model
8. Save the best model checkpoint to `models/bug_severity_dnn.keras`
9. Save the label encoder to `models/label_encoder.pkl`

## Model Details

The model uses:

- `TfidfVectorizer`
  - `max_features=10000`
  - `ngram_range=(1, 2)`
  - `min_df=2`
  - `max_df=0.95`
- Dense neural network:
  - input layer with TF-IDF feature size
  - dense layer with 500 units and ReLU
  - dropout with rate `0.4`
  - dense layer with 250 units and ReLU
  - dropout with rate `0.3`
  - dense layer with 100 units and ReLU
  - dropout with rate `0.2`
  - softmax output layer for 5 classes
- loss: `sparse_categorical_crossentropy`
- optimizer: `adam`
- metric: `accuracy`
- callbacks:
  - `EarlyStopping`
  - `ModelCheckpoint`

## Example Workflow

```powershell
# create environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# clean data
cd src
python preprocessing.py

# train model
python train.py
```

## Notes

- Run the scripts from inside the `src` directory because the current paths use
  `../data` and `../models`.
- The trained model file can be large, so it is ignored by `.gitignore` using
  model file patterns such as `*.keras`, `*.h5`, and `*.pkl`.
- The `data/` directory is also ignored by default because datasets can be large.
- If you clone this project without the data and model files, recreate the data
  with `notebooks/data.ipynb`, then run preprocessing and training again.

## Possible Improvements

- Save the fitted `TfidfVectorizer` so the trained model can be reused for
  inference on new bug reports.
- Add an inference script such as `predict.py`.
- Add evaluation metrics such as precision, recall, F1-score, and confusion
  matrix.
- Add a Streamlit interface for entering a bug report and predicting severity.
- Move hard-coded paths into a configuration file or command-line arguments.

## Requirements

Main libraries used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- datasets
- transformers
- joblib
- jupyter
- ipykernel
