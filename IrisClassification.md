# Data Classification : Iris Dataset

## Installing Required Libraries
#

* scipy
* numpy
* matplotlib
* pandas
* sklearn

## Importing the libraries
#

Importing and checking the version of the libraries on my jupyter notebook.

```py

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

```

Output:

```txt
Python: 3.9.4 (default, Apr  5 2021, 01:50:46) 
[Clang 12.0.0 (clang-1200.0.32.29)]
scipy: 1.7.0
numpy: 1.20.2
matplotlib: 3.4.2
pandas: 1.2.5
sklearn: 0.24.2
```

## Loading the Data
#

The dataset provided was [Iris Dataset.](https://archive.ics.uci.edu/ml/datasets/Iris "Iris Dataset")

This dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.

I loaded the CSV file URL.

```py

# load through url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pandas.read_csv(url, names = attributes)
dataset.columns = attributes)

```

## Taking a look at the Data(Summarizing)
#

### Shape of the dataset

The dataset has 150 instances with 4 attributes and a class

```py
# shape of dataset
print(dataset.shape)
```

Output:

```txt
(150, 5)
```

### Peak at the data

Just to visualise the sorting of the data

```py
# head
print(dataset.head(25))
```

### Statistical View

A Statistical Summary always helps in understanding while playing with a dataset

```py
# descriptions
print(dataset.describe())
```

### Grouping by Class

I tried to look at the class distribution of the dataset

```py
# class distribution
print(dataset.groupby('class').size())
```

## Visualising
#

Now, I had a basic idea about how the dataset is and I went on to Visualise through plotting the data.

### Univariate Plot

#### Histogram

```py
# histograms
dataset.hist()
pyplot.show()
```

//Add image

### Multivariate PLot

#### ScatterPlot

Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

```py
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
```

//Add image

## Creating a Validation Dataset
#

I sliced the data set to work on the training data.

80% of the dataset would be used as training dataset while the 20% remaining I will use it as a validation dataset.

```py
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
```

I used [this documentation](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/ "Slicing numpy arrays") to understand slicing of numpy-arrays in Python.

## Building a Model
#

I don’t know which algorithms would be good on this problem or what configurations to use.

Let’s test 6 different algorithms:

* Logistic Regression (LR)
* Linear Discriminant Analysis (LDA)
* K-Nearest Neighbors (KNN)
* Classification and Regression Trees (CART)
* Gaussian Naive Bayes (NB)
* Support Vector Machines (SVM)

I found these algorithms being used by [Machine Learning Mastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ "Machine Learning Mastery")

```txt
LR: 0.941667 (0.065085)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
CART: 0.941667 (0.053359)
NB: 0.950000 (0.055277)
SVM: 0.983333 (0.033333)
```

In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%.

I also created a box and whisker plot to visualise the accuracy.

## Classifcation

*I will finish this soon*

## My Experience
#

### Hurdles
#

* No good experience in Python
* Never used Jupyter Notebook
* Never used basic Python Libraries
  * Numpy
  * Pandas
* New to documenting on .md format

### Headstarts
#

* I knew basics of Machine Learning
* Huge confidence that i can do this.

So I start with looking for some help or guides on [Google.](https://www.google.com "Google Search")

I find various webpages explaining their work and few of them catch my eyes quite well. Since I am a newbie i start following guides and tutorials on various hurdles I mentioned above.

Here are the tutorials and references which I went through for understanding stuff:

* Markdown Documentation : [Markdown Github Guide](https://guides.github.com/features/mastering-markdown/ "Some Markdown Atributes")
* Jupyter Notebook Setup : [Tutorial by CodeWithHarry on YouTube](https://www.youtube.com/watch?v=TjRXT8mkTvM&list=PLu0W_9lII9agK8pojo23OHiNz3Jm6VQCH&index=2 "Jupyter Setup")
* Numpy : [Official Numpy Documentation](https://numpy.org/doc/1.21/ "Official Numpy Doc")
* Pandas : [Official Pandas Documentation](https://pandas.pydata.org/docs/getting_started/index.html#getting-started "Pandas Doc")
* Scikit-Learn : [Iris Dataset Example](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html?highlight=iris%20classification "Iris Dataset Classification")

