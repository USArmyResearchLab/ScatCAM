This repository contains two Python scripts:

1)trainer.py is the script used to train the weights and yield the reports which are 
saved in the TrainingReports directory

2)predictor.py loads the model, then loads the weights stored in TrainingReports to predict and
generate a class activation map example.

The TrainingReports directory: contains the trained weights for every mueller component
and training-validation reports for 3 cross-validation runs.  

Pre-requisites:
keras
numpy
pandas
sklearn

These scripts were tested and run in Python 3.2 and Python 3.6.
