import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

iris = datasets.load_iris()
# Make it into pandas dataframe
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(df.head())

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# We try it with default hyperparameters first. In case it doesn't provide good results,
# we can try fine-tuning it.
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

# NOTE: we use average=None in metrics function, which gives us the metrics for each class as an array
metrics = {
    'species': range(3),
    'precision': precision_score(y_test, y_pred, average=None),
    'f1_score': f1_score(y_test, y_pred, average=None),
    'recall': recall_score(y_test, y_pred, average=None)
}
# Make it as dataframe to nicely print it
metrics_df = pd.DataFrame(data=metrics)
print(metrics_df)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Make a scatterplot matrix
sns.pairplot(
    df,
    hue='target',
    diag_kind='hist',
    palette=['red', 'blue', 'green']
)
plt.suptitle('Iris Dataframe Plot', y=1.01, fontsize=14)
plt.show()