
import pandas
import numpy
import pickle
from azureml.core import Dataset, Run

from sklearn import tree
from sklearn.model_selection import train_test_split

run = Run.get_context()
workspace = run.experiment.workspace

dataset_name = 'tempData'

# Get a dataset by name
titanic_ds = Dataset.get_by_name(workspace=workspace, name=dataset_name)

# Load a TabularDataset into pandas DataFrame

temp_data = titanic_ds.to_pandas_dataframe()
temp_data


# Load features and labels
X, Y = temp_data[['machine_temperature', 'machine_pressure', 'ambient_temperature', 'ambient_humidity']].values, temp_data['anomaly'].values

# Split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# Change regularization rate and you will likely get a different accuracy.
reg = 0.01

# Train a decision tree on the training set
#clf1 = LogisticRegression(C=1/reg).fit(X_train, Y_train)
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X_train, Y_train)
print (clf1)

# Evaluate the test set
accuracy = clf1.score(X_test, Y_test)

print ("Accuracy is {}".format(accuracy))


# Serialize the model and write to disk
f = open('model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()
print ("Exported the model to model.pkl")


# Test the model by importing it and providing a sample data point
print("Import the model from model.pkl")
f2 = open('model.pkl', 'rb')
clf2 = pickle.load(f2)

# Normal (not an anomaly)
#X_new = [[24.90294136, 1.44463889, 20.89537849, 24]]
#X_new = [[33.40859853, 2.413637808, 20.89162813, 26]]
#X_new = [[34.42109181, 2.528985143, 21.23903786, 25]]

# Anomaly
X_new = [[33.66995566, 2.44341267, 21.39450979, 26]]
#X_new = [[105.5457931, 10.63179922, 20.62029994, 26]]

print ('New sample: {}'.format(X_new))

pred = clf2.predict(X_new)
print('Predicted class is {}'.format(pred))


# ### 4.2 Register Model

# You can add tags and descriptions to your models. Note you need to have a `model.pkl` file in the current directory. The below call registers that file as a model with the same name `model.pkl` in the workspace.
# 
# Using tags, you can track useful information such as the name and version of the machine learning library used to train the model. Note that tags must be alphanumeric.

from azureml.core.model import Model

model = Model.register(model_path = "model.pkl",
                       model_name = "model.pkl",
                       tags = {'area': "anomaly", 'type': "classification"},
                       description = "Sample anomaly detection model for IOT tutorial",
                       workspace = workspace)


print(model.name, model.description, model.version, sep = '\t')

