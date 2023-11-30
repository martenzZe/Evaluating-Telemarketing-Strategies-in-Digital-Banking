# Overview of the M4RAI Project


# Introduction
The goal of our M4RAI project is to forecast customer behaviors by analyzing previous outcomesof marketing campaigns, using machine learning techniques. 


# Dataset used
The bank-full.csv dataset, which includes a variety of customer attributes and their choice to sign up for a bank term deposit, was used in this project. Important characteristics include things like age, occupation, marital status, level of education, and more.

# Analysis and preprocessing
Preprocessing and data exploration are the first steps:

Data preprocessing: feature scaling, data splitting, and one-hot encoding of categorical variables.

Data Importation: involves using Pandas to load the dataset.

Exploratory Data Analysis (EDA): To comprehend data distribution, use descriptive statistics and visualization tools such as boxplots and histograms.


# Models and Methods
Numerous machine learning models were assessed and trained, including:

Logistic regression: The accuracy, precision, and recall are tested.

Histogram Gradient Boosting Classifier (HGBC): Assessed for performance but was deemed less appropriate because of problems with recall and precision.

Random Forest Classifier: In addition to SMOTE, is utilized to address class imbalance.

Decision tree classifier: The goal is to strike a balance between recall, accuracy, and precision.

# Advanced Analysis
LIME (Local Interpretable Model-agnostic Explanations): Used to recognize instance-level model predictions. 

SHAP (SHapley Additive exPlanations): The impact of features on the model's output is interpreted using.

Visualization: To better understand the decision-making process, decision trees were visualized.

 
