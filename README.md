# zoomcamp-HW7
## Description of the problem

The aim of the project is to identify customers that could show higher intent towards a recommended credit card given a set of customer data.

The data set is taken from Kaggle: https://www.kaggle.com/kedarnath07/credit-card-buyers/data

The aim is to predict whether customers is interested or not, which is represented by a binary variable being Is_Lead.

# Files in the repository

Data preparation and analysis steps, training (multiple) models, tuning models' performance and selecting the best model are included in notebook.ipynb.

Training the final model and saving it to a pickle file are done in train.py.

Loading the model and serving it via Flask is done predict.py.

Pipfile, Lock and Dockerfile are included in the repository under their names together with the data set as a csvfile.

Finally, HW7_test.ipynb is added to do test run via a notebook, after the project is initiated in the Docker environment. 

## Instructions on how to run the project


