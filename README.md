# HeadStart

This demonstration was developed as a part of the *Introduction to Data Science* course at the Univeristy of Helsinki. **HeadStart** is a model for predicting migraine occurence based on the data from smart wearables. It helps users track their helath data and warns them about a potential migraine episode occuring the next day.

## Motivation

Migraine episodes are known to be wildly unpredictable [PFIZER REPORT],moreover, highly individual nature of the disease makes the prediction tasks extremely difficultt.

## Data Collection
For the purposes of training the data models, a Kaggle database was employed. The database contains data from wearable devices for multiple users (`raw/migraine_dataset.csv`), containing relevant variables related to sleep, mood, stress, hydration, and screen time.

Currently,
due to the project being a *solo-project* some key functionalities of the app were not implemented. Real-time data collection and analaysis were outside of the scope of this course, and therefore, the model operates on the data from 100 users from the Kaggle database.x

## How to use?
1. Download the repo
2. Open terminal and type `streamlit run src/app/main.py`
3. Run

## Report
A `.pdf` report is located in `src/reports/miniprojectreport.pdf`
