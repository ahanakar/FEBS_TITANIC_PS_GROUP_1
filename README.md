## Spaceship Titanic – Data Science Problem Statement


## Repository Structure
├── train.py  #Runs EDA, preprocessing, feature engineering, and trains the model
├── test.py  #Loads trained model and generates predictions for test data
├── preprocess.py  #Data cleaning, encoding, and numerical EDA by Ahana
├── titanic.py  # Categorical EDA visualizations by Aditya
├── old.py  #Feature creation (TotalSpend, GroupSize, etc.) by Aditya
├── lor.py  #Logistic Regression training logic by Aditya
├── artifacts/
│ └── logistic_model.pkl  #Saved trained model
├── submission.csv  #Final prediction file
├── train.csv
├── test.csv
└── README.md

## Contributions 

**Aditya**
. preprocessing: dealt with splitting and manipualting certain columns to get different features 
. analysis: plotted various numerical and categorical plots including countplots, boxplots etc. showing count of transported with interpretations as comments
. model implementation: implemented pre trained class logistic regression in train.py

**Ahana**
. preprocessing: filled null values, scaled numerical data and encoded categorical data
. analysis: plotted categorical barplots with proportion transported and some numerical plots using boxplot and scatterplot with interpretations as comments 
. model implementation: wrote final code for train.py and test.py