# CatBoost Quickstart — ML Classification

Lately, I’ve been making use of the CatBoost Python library to create really successful classification models. I want to provide some quick to follow guides to get to grips with the basics.


In this article, I will describe three examples using CatBoost, to make: a binary classifier; a multinomial classifier; and finally a multinomial classifier which uses both categorical and numerical data.

![[Github Source](https://github.com/sk-t3ch/catboost-quickstart). Check out [https://t3chflicks.org](https://t3chflicks.org) for more content!](https://cdn-images-1.medium.com/max/4000/1*cCafusrU1Iey4flxrqbpRA.png)*

[Github Source](https://github.com/sk-t3ch/catboost-quickstart). Check out [https://t3chflicks.org](https://t3chflicks.org) for more content!*

## Classification

Statistical models can be derived from the email data with known classifications. These models can then be used to classify new observations and the process can be automated. A common example of a classification problem is that of identifying *SPAM* vs *NOT-SPAM *email. This is known as a **binary classification**. Email data includes many features such as recipient and that can be used to help identify it as either of these classifications.

## Decision Trees

Decision trees can be used to create classification models:
> [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning): A **decision tree** is a **decision** support tool that uses a **tree**-like model of **decisions** and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

This technique, whilst useful, can be made significantly more accurate by gradient boosting:
> [Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting): **Gradient boosting** is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.

## CatBoost

[CatBoost](https://catboost.ai/) is an open source algorithm based on gradient boosted decision trees. It supports numerical, categorical and text features. Check out the [docs](https://catboost.ai/docs/features/categorical-features.html#dataset-processing).
> # *Code along with the notebooks:*
> # [🔗 CatBoost QuickStart Notebooks on Github 📔](https://github.com/sk-t3ch/catboost-quickstart)

## #1 Binary Classification

In this section, we will be creating a binary classifier on the Iris plant dataset which contains three species each with four parameters (petal length, petal width, sepal width, sepal length).

![Iris Flower Dataset In a Pandas DataFrame followed by the value counts of species](https://cdn-images-1.medium.com/max/2000/1*SyYdPhtXRUdbBzPgWbIc2g.png)*

Iris Flower Dataset In a Pandas DataFrame followed by the value counts of species*

We will use the CatBoost library to create a binary classification model in order to tell the difference between *Virginica* and not.

### Getting the data

Start by using the Iris dataset from the [sklearn library](https://scikit-learn.org/stable/user_guide.html):


```
from sklearn.datasets import load_iris
import pandas as pd

def load_new_data():
    iris = load_iris()
    df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

df = load_new_data()
```

### Looking at the data

Using the [seaborne library](https://seaborn.pydata.org/introduction.html), we can inspect the data as pair-plots for each variable. The Iris dataset contains equal data points on three species. By eye, see that the graphs for *species* and *target* display an equal three way split. This is a trivial dataset to determine classifiers.

![](https://cdn-images-1.medium.com/max/2000/1*6YY2688-mcdBPrneVivdpg.png)

### Training the model

The classification problem here is to differentiate between samples of *Virginica* and *Not-Virginica. *The model parameters for this type of problem are required to use the *LogLoss *loss function.


```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

X_tr, X_eval = train_test_split(df)

y_tr = df_tr.species
y_tr_clean = y_tr == 'virginica'
X_tr.drop(columns=['species', 'target'], inplace=True)

y_eval =  X_eval.species
y_eval_clean = y_eval == 'virginica'
X_eval = X_eval.drop(columns=['species', 'target'])

train_dataset = Pool(X_tr, y_tr_clean, feature_names=list(X_tr.columns))

model_params = {
    'iterations': 10, 
    'loss_function': 'Logloss',
    'train_dir': 'crossentropy',
    'allow_writing_files': False,
    'random_seed': 42,
}

model = CatBoostClassifier(**model_params)
model.fit(train_dataset, verbose=True, plot=True)
```

### Results of the model

Extracting the predicted results on the evaluation data set and comparing to the true values generated the following graphs:

![](https://cdn-images-1.medium.com/max/2000/1*4-gg5mhgdC5PIha2V8IUHQ.png)

The ROC graph shows a near perfect score for this tiny dataset.

![](https://cdn-images-1.medium.com/max/2000/1*DnpzWBBf_MKopscweFU6Dg.png)

Extracting the relative importance of features shows us that petal width is over two times more important than any other feature in the classifier.

## #2 Multinomial Classifier

Now that we can distinguish a *Virginica*, we will move onto distinguishing multiple species by creating a multinomial classifier.

### Training the model

CatBoost handles any hard work and we only need to update the loss function to use *MultiClass:*


```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

df = load_new_data()
X_tr, X_eval = train_test_split(df)

y_tr = X_tr.species
y_tr_clean = y_tr
X_tr.drop(columns=['species', 'target'], inplace=True)
y_eval =  df_eval.species
y_eval_clean = y_eval
X_eval = df_eval.drop(columns=['species', 'target'])


train_dataset = Pool(X_tr, y_tr_clean, feature_names=list(X_tr.columns))

model_params = {
    'iterations': 10, 
    'loss_function': 'MultiClass',
    'train_dir': 'crossentropy',
    'allow_writing_files': False,
    'random_seed': 42,
}

model = CatBoostClassifier(**model_params)
model.fit(train_dataset, verbose=True, plot=True)
```

### Results of the model

Extracting the predicted results on the evaluation data set and comparing to the true values generated the following graphs:

![](https://cdn-images-1.medium.com/max/2000/1*5zjTbbu-TfAO04OYgkgz-A.png)

The confusion matrix for these results show that the model is completely accurate.

![](https://cdn-images-1.medium.com/max/2000/1*OWPFlg0sfD8Y1xOow8CHDQ.png)

Extracting the relative importance, this graph shows that the petal length is now more important than any other feature in the classifier.

## #3 Multinomial Classifier with Categorical Features

### Getting the data

Another dataset with lots of categorical features is this Car dataset from [UCL machine learning](https://archive.ics.uci.edu/ml/index.php).


```python
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
```

![](https://cdn-images-1.medium.com/max/6012/1*T58UtwDd8awYne7D9c4pgA.png)

### Train the model

The classification problem is now to distinguish different vehicles makes e.g. *Alfa Romeo*. We must define our categorical features such as *engine-location* in the CatBoost parameters.


```python
filter_ = ~(df.isin([np.nan, np.inf, -np.inf]).any(axis=1))
df = df.loc[filter_]
X_tr, X_eval = train_test_split(df, test_size=0.5)


y_tr = X_tr.make
y_eval = X_eval.make

X_tr = X_tr.drop(columns=['make'])
X_eval = X_eval.drop(columns=['make'])

features = [col_name for col_name in X_tr.columns if col_name != 'make']
cat_features = [col_name for col_name in features if X_tr[col_name].dtype == 'object']


train_dataset = Pool(X_tr, y_tr, feature_names=list(X_tr.columns), cat_features=cat_features)

model_params = {
    'iterations': 500, 
    'loss_function': 'MultiClass', 
    'train_dir': 'crossentropy',
    'allow_writing_files': False,
    'random_seed': 42,
}

model = CatBoostClassifier(**model_params)
model.fit(train_dataset, verbose=True, plot=True)
```

### Results of the model

Extracting the predicted results on the evaluation data set and comparing to the true values generated the following graphs:

![](https://cdn-images-1.medium.com/max/2000/1*nPJjnT_v4voSphNpRxvt8g.png)

The confusion matrix shows this model to be mostly accurate, ~74% of values correctly predicted. However, for such a small sample of data we cannot be too sure.

![](https://cdn-images-1.medium.com/max/2000/1*WAb-Lkcl3y-im4MwbY4m6Q.png)

An interesting note from the extracted feature importances is price is a poor predictive of vehicle make.

## Parameter Tuning

CatBoost claims to have great defaults and we’ve seen it to be quite successful on two different datasets. In order to get the most out of your model, you will have to change the parameters. I recommend looking at the [CatBoost website instructions.](https://catboost.ai/docs/concepts/parameter-tuning.html)

## Thanks For Reading

I hope you have enjoyed this article. If you like the style, check out [T3chFlicks.org](https://t3chflicks.org/) for more tech focused educational content ([YouTube](https://www.youtube.com/channel/UC0eSD-tdiJMI5GQTkMmZ-6w), [Instagram](https://www.instagram.com/t3chflicks/), [Facebook](https://www.facebook.com/t3chflicks), [Twitter](https://twitter.com/t3chflicks)).