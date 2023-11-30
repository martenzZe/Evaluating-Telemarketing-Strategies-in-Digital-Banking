# Overview of the M4RAI Project


# Introduction
The goal of our M4RAI project is to forecast customer behaviors by analyzing previous outcomesof marketing campaigns, using machine learning techniques. 

# Requirements
```
from sklearn.preprocessing import StandardScaler
from scipy import stats as sts
import pandas as pd
from scipy.stats.distributions import randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_hist_gradient_boosting
import lime as lime
from lime.lime_tabular import LimeTabularExplainer
import shap as shap
import pickle
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
```


# Dataset used
The bank-full.csv dataset, which includes a variety of customer attributes and their choice to sign up for a bank term deposit, was used in this project. Important characteristics include things like age, occupation, marital status, level of education, and more.

# Analysis and preprocessing
**Preprocessing and data exploration are the first steps:**
Data preprocessing: feature scaling, data splitting, and one-hot encoding of categorical variables.
<img width="935" alt="Screenshot 2023-11-30 at 17 56 40" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/cca2055f-0d92-4122-a02d-1b5f7f480b72">
<img width="1353" alt="Screenshot 2023-11-30 at 17 58 03" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/6e2bedad-6ba4-49e2-95a3-c8c257ec7706">


Data Importation: involves using Pandas to load the dataset.

Exploratory Data Analysis (EDA): To comprehend data distribution, use descriptive statistics and visualization tools such as boxplots and histograms.
<img width="440" alt="Screenshot 2023-11-30 at 18 00 13" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/eebb52a6-e11d-4581-b41f-0a30cf2b6193"><img width="434" alt="Screenshot 2023-11-30 at 18 00 34" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/44a8fd97-2d73-405b-b186-f2094e5967c0">




# Models and Methods
**Numerous machine learning models were assessed and trained, including:**

Logistic regression: The accuracy, precision, and recall are tested.

Histogram Gradient Boosting Classifier (HGBC): Assessed for performance but was deemed less appropriate because of problems with recall and precision.

Random Forest Classifier: In addition to SMOTE, is utilized to address class imbalance.

Decision tree classifier: The goal is to strike a balance between recall, accuracy, and precision.

# Advanced Analysis
LIME (Local Interpretable Model-agnostic Explanations): Used to recognize instance-level model predictions. 
<img width="1010" alt="Screenshot 2023-11-30 at 17 59 07" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/9cd96d8b-6b43-438c-9964-724054186ea0">

SHAP (SHapley Additive exPlanations): The impact of features on the model's output is interpreted using.
<img width="605" alt="Screenshot 2023-11-30 at 17 59 35" src="https://github.com/martenzZe/Evaluating-Telemarketing-Strategies-in-Digital-Banking/assets/152230960/ea11ae1c-4ed0-4875-8bcb-fcad905c1a32">


Visualization: To better understand the decision-making process, decision trees were visualized.


 
