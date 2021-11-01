# zoomcamp-HW7

# Description of the problem

The aim of the project is to identify customers that could show higher intent towards a recommended credit card given a set of customer data.

The data set is taken from Kaggle: https://www.kaggle.com/kedarnath07/credit-card-buyers/data

The aim is to predict whether customers are interested or not, which is represented by a binary variable "Is_Lead".

# Files in the repository

Data preparation and analysis steps, training (multiple) models, tuning models' performance and selecting the best model are included in notebook.ipynb.

Steps including "training the final model" and "saving it to a pickle file" are done in 'train.py'.

Loading the model and serving it via Flask is done in 'predict.py'.

'Pipfile', 'Lock' and 'Dockerfile' are included in the repository under their respective names together with the data set as a csvfile.

Finally, 'HW7_test.ipynb' is added to do test run via a notebook, after the project is initiated in the Docker environment. 

# Instructions on how to run the project

First, download all the files into the same folder on your computer. 

For viewing and editing notebooks, Jupyter can be used. Other files can be viewed and edited on VS Code.

On Mac OS, on Terminal window, while staying in the same folder that contains the downloaded files, run "pip install pipenv" to create 'Pipfile' and 'Pipfile.lock' files. 

Using "pipenv install" instead of "pip install" will create virtual environment. Same code is used in the Dockerfile

On Mac OS, while Docker is running as a separate program and staying in the same folder that contains all the downloaded files, type "docker build -t zoomcamp-hw7 ." on Terminal window to install container.

Type "docker run -it --rm -p 9696:9696 zoomcamp-hw7" after Docker installations are completed to activate container environment.

Finally, open "HW7_test.ipynb". Run the file and get the prediction result for the test customer.
