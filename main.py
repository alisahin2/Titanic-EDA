# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
PATH = ""
for dirname, _, filenames in os.walk(PATH + 'input' ):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
        
train_df=pd.read_csv(PATH + "input/titanic/train.csv")
test_df=pd.read_csv(PATH + "input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]

print("------Variables Names")
print(train_df.columns)

print("This outputs are first 5 data and last 5 data")
print(train_df.head())


      
print(train_df.describe())


print("------------About of describe: ------------")
print("PassengerId : unique id number to each passenger")
print("Survived : passenger survive(1) or died(0)")
print("Pclass : passenger class")
print("Name : name")
print("Sex : gender of passenger")
print("Age : age of passenger")
print("SibSp : number of sublings/spouses")
print("Parch : number of parents/children")
print("Ticket : ticket number")
print("Fare : amount of money spent on ticket")
print("Cabin :cabin category")
print("Embarked : port where passenger embarked( C: Cherbourg, Q: Queenstown, S:Southampton")


print("------Info Screen------")
print(train_df.info())


print("Categorical Variable")
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        otput: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    

print("------input: variable ex: Sex , otput: bar plot & value count------") 
category1 = [ "Survived", "Sex","Pclass","Embarked","SibSp", "Parch"]
for c in category1:
    bar_plot(c)
  

print("------Parametrelere gore dagilim grafigi------") 
category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
    

print("------Numerical Variable------")    
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50 )
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
    
numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)
    

print("------Basic Data Analysis------")


# Plass vs Survived
print("Class vs Survived")
print(train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending = False)
)

# Sex vs Survived
print("Sex vs Survived")
print(train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)
)
# SibSp vs Survived
print("SibSp vs Survived")
print(train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending = False)
)
# Parch vs Survived
print("Parch vs Survived")
print(train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending = False)
)


def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step
        outlier_step = IQR * 1.5
        #detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indeces
        outlier_indices.extend(outlier_list_col)
        
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2 )

    return multiple_outliers
print("------Outlier Detection------")      
print(train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
)

print("------Missing Value")
# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]), axis = 0).reset_index(drop = True)


train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)

print("------Find Missing Value------")
print(train_df.head())

train_df.columns[train_df.isnull().any()]

print("Missing value sum")
print(train_df.isnull().sum())



train_df[train_df["Embarked"].isnull()]
print("Embarked has 2 missing value")
print(train_df[train_df["Embarked"].isnull()])


print("Boxplot: Fare, Embarked")
train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()


print("Embarked value filling")
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]
print("No missing elements")
print(train_df[train_df["Embarked"].isnull()])



train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

train_df[train_df["Fare"].isnull()] 
print(train_df[train_df["Fare"].isnull()] )
    

print("------Visualization------")

print("Correlation Between SibSp-Parch-Age-Fare-Survived")
list1 = ["SibSp" , "Parch" , "Age" , "Fare" , "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()
print("Decribe: Fare feature seems to have correlation with survived feature (0.26) SibSp feature seems to have correlation with parch feature(0.35) Parch feature seems to have correlation with SibSp feature(0.35)")

print("SibSp - Survived")
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probality")
plt.show()
print("Describe: Having a lot of SIbSp have less chance to survice")
print("if sibsp == 0 or 1 or 2. passenger has more chance to survive")
print("We can consider a new feature describing these categories")


print("Parch - Survived")
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probality")
plt.show()
print("Sibsp and parch can be used for new feature extraction with th = 3 ")
print("small families have more chance survive")
print("there is a std in survival of passenger with parch = 3")



print("Pclass - Survived")
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Survived Probality")
plt.show()




print("Age - Survived")
g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()
print("age <= 10 has a high survival rate")
print("oldest passengers (80) survived")
print("large number of 20 years old did not survive")
print("most passengers are in 15-35 are range")
print("use age feature in training")
print("use age distribution for missing value of age")



print("Pclass - Survived - Age")
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()
print("pclass is important feature for model training")




print("Sex - Pclass - Survived")
g = sns.FacetGrid(train_df, row = "Embarked", size = 3)
g.map(sns.pointplot, "Pclass", "Survived","Sex")
g.add_legend()
plt.show()
print("Female passengers have much better survival rate than males")
print("Males have better survival rate in pclass 3 in C.")
print("Embarked and sex will be used in training")



print("Embarked - Sex - Fare - Survived")
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 3)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()
print("Passengers who pay higher fare have better survival.Fare can be used as categorical for training.")

print("Fill missing Age Feature")
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()
print("Sex is not informative for age prediction, age distribution seems to be same.")



sns.factorplot(x = "Sex", y = "Age", hue = "Pclass", data = train_df, kind = "box")
plt.show()
print("!st class passengers are older than 2nd, 2nd is older than 3rd class.")






sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()




train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]

sns.heatmap(train_df[["Age", "Sex","SibSp", "Parch", "Pclass"]].corr(), annot = True)
plt.show()
print("Age is not correlated with sex but it is correlated with parch and sibsp and pclass")





index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med
        
        
train_df[train_df["Age"].isnull()]

print(train_df[train_df["Age"].isnull()])


























    