# First import libraries for the Exploratory Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Next import the libraries for the random forest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Upload the data 
mf_df = pd.read_csv('manufacturing_defect_dataset.csv')
null_counts = mf_df.isnull().sum()

df_corr = mf_df.corr()

plt.subplots(figsize=(20,15))
sns.heatmap(df_corr, linewidth=.5, cmap= "Paired", annot = True)

sorted_corr = mf_df.corr()['DefectStatus'].sort_values(ascending=False)
top_indexes = sorted_corr.index[1:7]

sns.pairplot(mf_df, vars=top_indexes, diag_kind="kde", hue="DefectStatus", aspect = 1.85)

# Set the random state variable
random_state99 = 999

# extarct the target variable 
X = mf_df.drop('DefectStatus', axis = 1)
y = mf_df['DefectStatus']

# set the scalar
scalars = StandardScaler()
df_scaled = scalars.fit_transform(X)

# Trim off the trainging data
first_splitter = .4
second_splitter = .5
X_train, df_x, y_train, df_y = train_test_split(df_scaled, y, test_size=first_splitter, random_state=1)
cross_x, test_x, cross_y, test_y= train_test_split(df_x, df_y, test_size=second_splitter, random_state=1)

train_samp_list = []
acc_samp_list = []

train_depth_list = []
acc_depth_list = []


# test the lengths of my data
print('the lengths of the total, xtrain, xtest, and cv are:', len(mf_df)
, len(X_train), len(test_x), len(cross_x))
1944 +648 +648


# next find the optimal samples
# this splits up the two ends 2 and 300 up into even chunks
min_splits = np.linspace(2,250, 50).astype(int)
print(min_splits)
for min_s in min_splits:
    rf_model = RandomForestClassifier(min_samples_split=min_s, random_state=random_state99).fit(X_train, y_train)
    rf_preds = rf_model.predict(X_train)
    rf_cross = rf_model.predict(cross_x)
    rf_acc = accuracy_score(y_train, rf_preds)
    rf_acc_cross = accuracy_score(cross_y, rf_cross)
    train_samp_list.append(rf_acc)
    acc_samp_list.append(rf_acc_cross)

# Moving on to the depth parameter exploration
min_depths = np.linspace(2,50, 20).astype(int)
print(min_depths)
for min_d in min_depths:
    rf_model = RandomForestClassifier(random_state=random_state99, max_depth=min_d).fit(X_train, y_train)
    rf_preds = rf_model.predict(X_train)
    rf_cross = rf_model.predict(cross_x)
    rf_acc = accuracy_score(y_train, rf_preds)
    rf_acc_cross = accuracy_score(cross_y, rf_cross)
    train_depth_list.append(rf_acc)
    acc_depth_list.append(rf_acc_cross)