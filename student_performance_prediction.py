import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import seaborn as sns

# Reads the data set
student_por_df = pd.read_csv('data/student-por.csv', sep=';')

# Step1: Exploratory data analysis - Student Performance Data Set
student_por_df.head()
# student_por_df.describe()
student_por_df['G3'].describe()
student_por_df.shape

# Visualize the G3
plt.plot(sorted(student_por_df['G3']))
plt.title('Final Grade Distribution')
plt.grid()

# Step 2: Data Preparation and Feature engineering

# find all non-numerical data
non_nueric_features = [feat for feat in list(student_por_df) if feat not in list(student_por_df._get_numeric_data())]
for feat in non_nueric_features:
    print(feat, ':', set(student_por_df[feat]))

# Convert to numerical data.
for feat in non_nueric_features:
    dummies = pd.get_dummies(student_por_df[feat]).rename(columns=lambda x: feat + '_' + str(x))
    student_por_df = pd.concat([student_por_df, dummies], axis=1)

student_por_df = student_por_df[[feat for feat in list(student_por_df) if feat not in non_nueric_features]]
student_por_df.shape
student_por_df.head()

# Modeling with XGBoost
# create an xgboost model
# run simple xgboost classification model and check
# prep modeling code
outcome = 'G3'
features = [feat for feat in list(student_por_df) if feat not in outcome]

X_train, X_test, y_train, y_test = train_test_split(student_por_df, student_por_df[outcome], test_size=0.3,
                                                    random_state=42)

xgb_params = {
    'eta': 0.01,
    'max_depth': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'seed': 0
}

dtrain = xgb.DMatrix(X_train[features], y_train, feature_names=features)
dtest = xgb.DMatrix(X_test[features], y_test, feature_names=features)
evals = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params=xgb_params,
                      dtrain=dtrain,
                      num_boost_round=2000,
                      verbose_eval=50,
                      early_stopping_rounds=500,
                      evals=evals,
                      # feval = f1_score_cust,
                      maximize=False)

# plot the important features
fig, ax = plt.subplots(figsize=(6, 9))
xgb.plot_importance(xgb_model, height=0.8, ax=ax, max_num_features=20)
plt.show()

# get dataframe version of important feature for model
xgb_fea_imp = pd.DataFrame(list(xgb_model.get_fscore().items()),
                           columns=['feature', 'importance']).sort_values('importance', ascending=False)
xgb_fea_imp.head(10)

print(xgb_model.predict(dtest)[0:10])

key_features = list(xgb_fea_imp['feature'].values[0:40])
key_features

# Take students with a predicted final score of less than 10 over 20
predicted_students_in_trouble = X_test[X_test['G3'] < 10]

# See which feature they landed well below or well above peers
for index, row in predicted_students_in_trouble.iterrows():
    print('Student ID:', index)
    for feat in key_features:
        if row[feat] < student_por_df[feat].quantile(0.25):
            print('\t', 'Below:', feat, row[feat], 'Class:',
                  np.round(np.mean(student_por_df[feat]), 2))
        if row[feat] > student_por_df[feat].quantile(0.75):
            print('\t', 'Above:', feat, row[feat], 'Class:',
                  np.round(np.mean(student_por_df[feat]), 2))

# Let's create a better looking report to share our findings
lower_limit_threshold = 0.25

# See which feature they landed well below or well above peers
for index, row in predicted_students_in_trouble.iterrows():
    student_id = index
    important_low_features = []

    for feat in key_features:
        if row[feat] < student_por_df[feat].quantile(lower_limit_threshold):
            important_low_features.append(feat)

    # create new data set for this student
    at_risk_student = pd.DataFrame(row[important_low_features]).T
    at_risk_student['Retention_Risk'] = True
    student_mean = pd.DataFrame(student_por_df[important_low_features].mean(axis=0)).T
    student_mean['Retention_Risk'] = False
    student_profile = pd.concat([at_risk_student, student_mean])
    student_profile = pd.melt(student_profile, id_vars="Retention_Risk")

    print('Student ID:', student_id)
    sns.catplot(x='variable', y='value', hue='Retention_Risk', data=student_profile, kind='bar',
                palette=sns.color_palette(['blue', 'red']))

    plt.show()

# See which feature they landed well below or well above peers
upper_limit_threshold = 0.75

for index, row in predicted_students_in_trouble.iterrows():
    student_id = index
    important_above_features = []

    for feat in key_features:
        if row[feat] > student_por_df[feat].quantile(upper_limit_threshold):
            important_above_features.append(feat)

    # create new data set for this student
    at_risk_student = pd.DataFrame(row[important_above_features]).T
    at_risk_student['Retention_Risk'] = True
    student_mean = pd.DataFrame(student_por_df[important_above_features].mean(axis=0)).T
    student_mean['Retention_Risk'] = False
    student_profile = pd.concat([at_risk_student, student_mean])
    student_profile = pd.melt(student_profile, id_vars="Retention_Risk")

    print('Student ID:', student_id)
    sns.catplot(x='variable', y='value', hue='Retention_Risk', data=student_profile, kind='bar',
                palette=sns.color_palette(['blue', 'red']))
    plt.show()
