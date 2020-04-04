import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

prop = pd.read_csv("property.csv")

label_encoder = preprocessing.LabelEncoder()
prop['type']= label_encoder.fit_transform(prop['type'])
prop['furnished']= label_encoder.fit_transform(prop['furnished'])
 
prop_new = prop[['type', 'noOfBedrooms', 'totalSqft', 'noOfBathrooms', 'noOfBalconies', 'price', 'distanceToNearestGym', 'distanceToNearestSchool', 'distanceToNearestHospital', 'furnished']]
#print(prop_new.head())
model = KMeans().fit(prop_new)

#print(model.labels_)

print({i: np.where(model.labels_ == i)[0] for i in range(model.n_clusters)})

#print(model.predict([[0, 4]]))