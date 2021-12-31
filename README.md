
# Regression Analysis with Decision Tree and Random Forest

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Credits](#credits)


![](https://www.bankforeclosuressale.com/images/categories/cream-multi-family-homes.jpg)

## Overview
The project's objective is to anticipate the deal prize of Houses on the basis of previous data. The dataset is downloaded from Kaggle which contains 1460 rows and 81 columns. 43 categorical columns, 37 numerical columns and 1 target column(SalePrize). Sale Prize of Houses is continuous quantity so Regression Analysis is used in both algorithms. Firstly, Datset is trained on Decision Tree Regressor and then Random Forest Regressor. Manual Hyperparameter Tuning is done to find the best hyperparatmers for the model to give maximum accuracy. 


## Motivation
 After learning about Classification and Regression Trees(CART) and Random Forest, I wanted to apply my theoritical knowledge into practice. 

## Technical Aspect
1. The entire project is completed on Google Colab, which provide a Jupyter notebook environment that runs entirely in the cloud.
2. For Feature Engineering 
     * Sklearn.preprocessing is used
     * Category_encoders is used
3. For Model Building
     * Sklearn.trees for Decision Tree Regression
     * Sklearn.ensembles for Random Forest
4. For visualization Matplotlib is used
6. For Model Evaluation sklearn.metrics is used


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```


## To Do
1. Using other different ML algorithms to find the accuracy of the prediction.
2. automated Hyperparameter Tuning can be done to improve accuracy of the existing model.
3. Exploratory Data Analysis can be done to learn the cause and effect relationship between Independent Columns and Dependent Columns.

## Bug / Feature Request
If you find a bug (the model couldn't handle the query and / or gave undesired results), kindly open an issue [<a href = "https://github.com/itspb008/Used-Car-Quality-Detection/issues/new">here</a>] by including your search query and the expected result.


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.software.ac.uk/sites/default/files/images/content/jupyter-main-logo.svg" width=300>](https://jupyter.org/)    [<img target="_blank" src = "https://www.bgp4.com/wp-content/uploads/2019/08/Scikit_learn_logo_small.svg_-840x452.png" width=170>](https://scikit-learn.org/stable/)                     [<img target="_blank" src="https://miro.medium.com/max/3880/1*ddtqWGkJz1TUCg1WM9qKeQ.jpeg" width=280>](https://colab.research.google.com/) 



[<img target="_blank" src="https://blueorange.digital/wp-content/uploads/2019/12/logo_matplotlib.jpg" width=200>](https://matplotlib.org/stable/index.html) 


## Team
Prashant Bharti


## Credits
- https://jovian.ai/learn/machine-learning-with-python-zero-to-gbms
- Machine Learning Course by Andrew NG on Coursera
- Analytics Vidhya Blogs 
- Krish Naik Youtube Channel
- StatQuest with Josh Stramer Youtube Channel
- Kaggle.com
