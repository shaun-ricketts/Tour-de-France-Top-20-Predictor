# Tour de France Top 20 Predictor

## Introduction

The aim of this project is to predict the final standings of the Tour de France. Initially, I planned to use a linear model to predict each rider’s overall time, with a top-20 likelihood prediction as a secondary feature. However, after testing multiple approaches, I found that logistic models outperformed the linear ones in predicting whether a rider would finish inside the top 20. As a result, I made this the primary focus of the project.

All of the data was self-collected via web scraping scripts I wrote using the cycling database site firstcycling.com. Note: the scraping code is not included in this repository.

I published my predictions for the 2025 edition of the Tour de France here: https://x.com/DataDomestique/status/1941185578671996941

## Tour de France and Professional Cycling Overview
The Tour de France takes place every summer and has been running since 1903, only pausing during the World Wars. It consists of 21 stages over 23 days, with two designated rest days. At the end of the race, the rider with the shortest cumulative time wins the General Classification (GC) — this is the outcome we aim to predict.

There are also secondary competitions:
- Points Classification: For the best sprinter.
- King of the Mountains: For the strongest climber.
- Youth Classification: GC for riders aged 25 and under.
- Each of the 21 stages is also contested individually, with its own winner.

The Tour typically begins in early July. My model uses each rider’s best results in races leading up to the Tour, focusing on one-week stage races. These are categorised as WorldTour or ProTour, the first and second tiers of professional road cycling.

Additionally, the model considers riders’ performances in the Tour de France over the past three years, as well as their results in the Giro d’Italia and Vuelta a España, the only other three-week races in the professional calendar. These Grand Tours provide stronger signals for potential success at the Tour de France than shorter races.

## Project Structure

Data/
  Raw/            - Sample CSV data (40% subset of full dataset)
  Processed/      - Output of Data_Prep.py on raw files

Models/             - Trained model files
Notebooks/          - Jupyter notebooks for EDA, training, evaluation
src/                - Reusable Python modules

requirements.txt    - Python dependencies
.gitignore          - Specifies files and folders to be ignored by Git
LICENSE             - MIT License
README.md           - This file

## Getting Started

Installation steps:

1. Clone the repository:
   git clone https://github.com/shaun-ricketts/Tour-de-France-Top-20-Predictor.git
   cd Tour-de-France-Top-20-Predictor

2. (Optional) Create a virtual environment:
   python -m venv venv
   source venv/bin/activate     (On Windows: venv\Scripts\activate)

3. Install dependencies:
   pip install -r requirements.txt

## Dependencies

The project uses the following libraries:

- pandas              (data manipulation)
- numpy               (numerical operations)
- matplotlib, seaborn (data visualization)
- scikit-learn        (model training and evaluation)
- statsmodels         (statistical testing and regression)
- joblib              (model persistence)
- shap                (model interpretability)
- xgboost, lightgbm, catboost (ensemble models)
- imbalanced-learn    (class imbalance handling)
- Unidecode           (text normalization for rider names)

Install with:
pip install -r requirements.txt

## Data

- Data is located in the "Data/Raw/" folder.
- The CSVs contain a reduced subset (~40%) of the full data used in the original model.
- This is sufficient to run the model pipeline, but output will differ from the original model's exact results.
- The "Data/Processed/" folder is present as a placeholder for storing cleaned/transformed datasets (output of data_prep)

## Model Workflow

- Load and explore data using notebooks
- Perform feature engineering
- Train and evaluate models (XGBoost, LightGBM, CatBoost, etc.)
- Run and save models to "Models/" directory

You can run the model using the appropriate notebook

## Running the Model

Example of model run shown in notebook "08_Model_Run_2025.ipynb", using the models from the Models folder

## Limitations and Future Improvements

### Limitations

- **Incomplete dataset**: The dataset is a reduced sample (40%) and does not represent the full rider pool from the original model. This will affect the accuracy of the predictions.
- **Missing contextual features**: The model does not currently account for a rider's role in the team (team leader, domestique, etc), which will affect each rider's motivation for the overall GC. There are many other factors which are difficult to account for in professional cycling, such as the weather, crashes, course terrain (the race changes each year in overall distance, meters of elevation, time-trial kms), etc. 

### Potential Improvements

- Test adding Weights to WT and PT races, as some races may be stronger predictors to the Tour than others
- Investigate other factors such as team strength, role in the team (e.g. Leader, Domestique, etc.), likelihood of DNFing, kms raced or number of races ridden prior to Tour start
- Build a model more specialised model to predict the likelihood of each rider winning
- Develop a simple UI or dashboard to visualise predictions

## License

All rights reserved © 2025 Shaun Ricketts.

This software is provided for personal, non-commercial viewing purposes only. You may not copy, modify, distribute, sublicense, or create derivative works without explicit written permission from the author.

See the LICENSE file for full details.

