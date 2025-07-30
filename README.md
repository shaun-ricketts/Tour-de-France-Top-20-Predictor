# Tour de France Top 20 Predictor

## Introduction

The aim of this project is to predict the final standings of the Tour de France. Initially, I planned to use a linear regression model to predict each rider’s overall position, with a secondary feature estimating their likelihood of finishing in the top 20. However, I found that treating this as a classification task, i.e. likelihood of finishing top 20, gave better results than trying to model exact finishing position. As a result, I shifted the project’s primary focus to classification.

All of the data was self-collected via web scraping scripts I wrote using the cycling database site firstcycling.com. *(Note: the scraping code is not included in this repository.)*

I published my predictions for the 2025 edition of the Tour de France here:  
https://x.com/DataDomestique/status/1941185578671996941

---

## Tour de France and Professional Cycling Overview

The Tour de France takes place every summer and has been running since 1903, only pausing during the World Wars. It consists of 21 stages over 23 days, with two designated rest days. At the end of the race, the rider with the shortest cumulative time wins the General Classification (GC) — this is the outcome we aim to predict.

There are also secondary competitions:
- Points Classification: For the best sprinter
- King of the Mountains: For the strongest climber
- Youth Classification: GC for riders aged 25 and under
- Each of the 21 stages is also contested individually, with its own winner

The Tour typically begins in early July. My model uses each rider’s best results in races leading up to the Tour, focusing on one-week stage races. These are categorised as WorldTour or ProTour — the first and second tiers of professional road cycling.

Additionally, the model considers riders’ performances in the Tour de France over the past three years, as well as their results in the Giro d’Italia and Vuelta a España, the only other three-week races in the professional calendar.

---

## Project Structure

- Data/
  - Raw/            - Sample CSV data (40% subset of full dataset)
  - Processed/      - Output of Data_Prep.py on raw files

- Models/             - Trained model files
- Notebooks/          - Jupyter notebooks for EDA, training, evaluation
- src/                - Reusable Python modules

- requirements.txt    - Python dependencies
- .gitignore          - Specifies files and folders to be ignored by Git
- LICENSE             - MIT License
- README.md           - This file

## Getting Started

### Recommended Setup: Using Anaconda

1. **[Download Anaconda](https://www.anaconda.com/products/distribution)** and install it.

2. Open **Anaconda Navigator**.

3. In the **Environments** tab:
   - Click **Create**
   - Name your environment (e.g., `tdf-env`)
   - **Select Python 3.9** (recommended — newer versions like 3.13 may break compatibility with some packages)
   - Click **Create**

4. After it’s created, **select the environment**, then click **"Open Terminal"** (or **CMD Prompt**).

5. Install Jupyter Notebook **inside the environment**:

   ```bash
   conda install notebook -c conda-forge

6. Once installed, you can open Jupyter directly via:
- Navigator → Environment → Open With → Notebook
- Or by typing 'jupyter notebook' in the terminal

### Alternative Setup: Using `venv` and `pip` (non-Anaconda)

1. **Create a virtual environment**:

   On **Windows** (CMD or PowerShell):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

  On macOS/Linux (bash/zsh):

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install Jupyter Notebook (if not already included):

   ```bash
   pip install notebook
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

### Clone the Repository (if you haven't already):

   ```bash
   git clone https://github.com/shaun-ricketts/Tour-de-France-Top-20-Predictor.git
   cd Tour-de-France-Top-20-Predictor
   ```

## Dependencies

All required packages are listed in requirements.txt.

To install via pip:

  ```bash
  pip install -r requirements.txt
  ```

## Data

- Data is located in the "Data/Raw/" folder.
- The CSVs contain a reduced subset (~40%) of the full data used in the original model.
- This is sufficient to run the model pipeline, but output will differ from the original model's exact results.
- The "Data/Processed/" folder is present as a placeholder for storing cleaned/transformed datasets (output of data_prep)

## Model Workflow

- Notebooks are ordered 0-8 and should be run in order, starting with 00_Data_Prep (this transforms raw files into a single processed file)
- 01 is for exploratory data analysis
- 02 - 04 tests different logic for replacing null data, data ranges and feature engineering
- 05 trains and evaluates different models (XGBoost, RandomForest, LightGBM, CatBoost, etc.)
- 06 performs hyperparameter testing for the best model(s)
- 07 runs and saves models to "Models/" directory
- 08 deploys saved model on new data

## Running the Model

To generate 2025 Tour de France predictions:

- Run 00_Data_Prep.ipynb - generates tdf_prepared.csv in Data/Processed
- Run 08_Model_Run_2025.ipynb
- This will load the trained model and output the predicted top 20 (Note: The results will differ to mine as you are running on a limited dataset).

Alternatively, you can rerun the entire pipeline using the sample data provided. Start from 00_Data_Prep.ipynb and proceed through the notebooks in order to train your own model (results will differ due to the reduced dataset).

## Limitations and Future Improvements

### Limitations

- Incomplete dataset: The dataset is a reduced sample (40%) and does not represent the full rider pool from the original model. This will affect the accuracy of the predictions. (Only applies if run using newly created models).
- Missing contextual features, the model does not currently account for:
- a rider's role in the team (team leader, domestique, etc), which will affect each rider's motivation for the overall GC.
- Likelihood of crashes/DNF
- Illness
- Team Goal and Tactics
- Team Strength

### Potential Improvements

- Test adding Weights to WT and PT races, as some races may be stronger predictors to the Tour than others, e.g. Criterium du Dauphine is likely to be a stronger predictor than Tour down Under.
- Investigate other factors such as team strength, role in the team (e.g. Leader, Domestique, etc.), likelihood of DNFing, kms raced or number of races ridden prior to Tour start.
- Build a model more specialised model to predict the likelihood of each rider winning.
- Develop a simple UI or dashboard to visualise predictions.
- Investigate other prediction techniques such as Cox proportional hazards model or Bayesian survival model for likelihood of a rider DNFing.

## License

All rights reserved © 2025 Shaun Ricketts.

This software is provided for **personal, non-commercial use** only.

You are free to:
- Download and run the code locally
- Use the project for learning, experimentation, and analysis

You may not:
- Use any part of this code for commercial purposes
- Distribute, sublicense, or create derivative works without explicit written permission

See the LICENSE file for full terms.
