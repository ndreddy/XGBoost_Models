import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

# Step 1: Collect the data
data = pd.read_csv("data/Fish.csv")

# Step 2: Visualize the data

# How does the data look like?
data.head()

# Does the data have missing values?
data.isna().sum()

# What is the distribution of the numerical features?
data_num = data.drop(columns=["Species"])

fig, axes = plt.subplots(len(data_num.columns) // 3, 3, figsize=(15, 6))
i = 0
for triaxis in axes:
    for axis in triaxis:
        data_num.hist(column=data_num.columns[i], ax=axis)
        i = i + 1
# What is the distribution of the target variable(Weight) with respect to fish Species?
sns.displot(data=data, x="Weight", hue="Species", kind="hist", height=6, aspect=1.4, bins=15)
plt.show()

# Step 3: Clean the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_cleaned = data.drop("Weight", axis=1)
y = data['Weight']
x_train, x_test, y_train, y_test = train_test_split(data_cleaned, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# label encoder
label_encoder = LabelEncoder()
x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
x_test['Species'] = label_encoder.transform(x_test['Species'].values)

model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]


# Step 4: Train the model
def evauation_model(pred, y_val):
    score_MSE = round(mean_squared_error(pred, y_val), 2)
    score_MAE = round(mean_absolute_error(pred, y_val), 2)
    score_r2score = round(r2_score(pred, y_val), 2)
    return score_MSE, score_MAE, score_r2score


def models_score(model_name, X_trn, y_trn, X_tst, y_tst):
    # model_1
    if model_name == "Decision_Tree":
        reg = DecisionTreeRegressor(random_state=42)
    # model_2
    elif model_name == "Random_Forest":
        reg = RandomForestRegressor(random_state=42)
    # model_3
    elif model_name == "XGboost_Regressor":
        reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, )
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(X_trn, y_trn)
        y_hat = reg.predict(X_tst)

        score_MSE, score_MAE, score_r2score = evauation_model(y_hat, y_tst)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)


result_scores = []
for model in model_list:
    score = models_score(model, x_train, y_train, x_test, y_test)
    result_scores.append((model, score[0], score[1], score[2]))
    print(model, score)
