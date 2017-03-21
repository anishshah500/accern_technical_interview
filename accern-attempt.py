import numpy as np
import scipy
import pandas as pd
from pandas import Series
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans, AgglomerativeClustering

#reading unzipped data into csv file
input_data = pd.read_csv('backtest_students_sample.csv') 
original_data = input_data

#storing the names of columns and no. of rows
no_of_rows = len(input_data)

#enumerating the id attributes
id_attribs = ['article_id','story_id','entities_global_id_1', 'entities_entity_id_1']

#list for storing which variables will be used to model clustering
model_vars = []

"""
Part-1 Finding the number of null values and finding the proportion of null values
"""

nan_count_ratios = dict()
null_threshold = 0.85

#iterating over each column and finding out the number of null values and ratio of null values
for i in input_data.columns:
	nan_count_ratios[i] = input_data[i].isnull().sum() / no_of_rows

#remove the columns with ratio of null values greater than the null threshold
for i in input_data.columns:
	if nan_count_ratios[i] > null_threshold:
		input_data = input_data.drop(i,1)
		print(str(i) + 'attribute dropped with null proportion ' + str(nan_count_ratios[i]))

#filling the missing values depending on whether the ID is an attribute
for i in input_data.columns:
	if nan_count_ratios[i] > 0 and i not in id_attribs:
		input_data[i] = input_data[i].fillna(input_data[i].mode().iloc[0])
	if nan_count_ratios[i] > 0 and i in id_attribs:
		input_data[i] = input_data[i].fillna("id missing")	

"""
Part-2 Normalizing the values for further analysis
"""

input_data_normalized = input_data

for i in input_data_normalized.columns:
	if np.issubdtype(input_data_normalized[i].dtype,np.number):
		model_vars.append(i)
		input_data_normalized[i] = (input_data_normalized[i] - min(input_data_normalized[i])) / (max(input_data_normalized[i]) - min(input_data_normalized[i]))
		print('variance ' + str(input_data_normalized[i].var()))

#Data has been normalized

"""
Part-3 Checking if there is an unusually high amount of variance in a certain time frame in numerical values
"""
window_size = 1000
anomaly_var_final = dict()

for i in input_data_normalized.columns:
	if np.issubdtype(input_data_normalized[i].dtype,np.number):
		print("Variance anomaly for " + str(i) + "lies in the time periods")
		r_w = input_data_normalized[i].rolling(1000)
		mean = r_w.var().mean()
		rolling_var = r_w.var()
		period_anomaly = []
		k = []
		counting = 0
		for j in range(999,len(rolling_var)):
			if (rolling_var[j] - mean) / mean > 0.5:
				if counting == 0:
					counting = 1
					k.append(j)
			else:
				if counting == 1:
					counting = 0
					k.append(j)
					if(k[1] - k[0] > 2500):
						period_anomaly.append([input_data_normalized.iloc[k[0]-1000]['harvested_at'],input_data_normalized.iloc[k[1]]['harvested_at']])
					#print(k)
					k = []
		print(i, period_anomaly)

"""
Part-4 Assigning the string value attributes to an integer for clustering
"""
map_keys = dict()
for i in input_data_normalized.columns:
	if not(np.issubdtype(input_data_normalized[i].dtype,np.number)):
		unique_values = input_data[i].unique()
		length = unique_values.size
		if length < 1000:
			keys = []
			for j in range(length):
				keys.append([input_data[i].unique()[j],j+1])
			map_keys[i] = keys

input_data_normalized_discretized = input_data_normalized

for i in input_data_normalized_discretized.columns:
	if i == 'first_mention':
		model_vars.append(i)
		continue
	if i in map_keys and not(np.issubdtype(input_data_normalized[i].dtype,np.number)):
		model_vars.append(i)
		map_keys_current = dict()
		for j in map_keys[i]:
			map_keys_current[j[0]] = j[1]
		input_data_normalized_discretized = input_data_normalized_discretized.replace({i : map_keys_current})
		if(len(map_keys_current) > 1):
			input_data_normalized_discretized[i] = (input_data_normalized_discretized[i] - min(input_data_normalized_discretized[i])) / \
			(max(input_data_normalized_discretized[i]) - min(input_data_normalized_discretized[i]))
		print(i + " replaced and normalized")

input_data_normalized_discretized['first_mention'] = input_data_normalized_discretized['first_mention'].astype(int)
print(model_vars)

#In this part the name of the corporation and the ticker aren't chosen to avoid overfitting since they have 5817 unzique values (Doubt)

"""
Part-5 Producing the k-means clusters and pointing out anomalies
"""

model_input = []

for i in model_vars:
	model_input.append(list(input_data_normalized_discretized[i]))

model_input = list(zip(*model_input))

kmeans = KMeans(n_clusters = 10).fit(model_input)

kmeans_cluster_allocation = kmeans.labels_
kmeans_cluster_centers = kmeans.cluster_centers_
threshold_for_anomaly = 2
anomalies_kmeans = []

data_points = input_data_normalized_discretized[model_vars]

for i in range(no_of_rows):
	data_point = np.array(data_points.iloc[i])
	data_point_cluster_center = np.array(kmeans_cluster_centers[kmeans_cluster_allocation[i]])
	k = data_point - data_point_cluster_center
	distance = (np.dot(k,k))**0.5
	if(distance > threshold_for_anomaly):
		#print("anomaly")
		anomalies_kmeans.append(i)

"""
Part-6 Producing a MeanShift clustering prediction for anomalies
"""

meanshift = MeanShift(bin_seeding = True).fit(model_input)

meanshift_cluster_allocation = meanshift.labels_
meanshift_cluster_centers = meanshift.cluster_centers_

meanshift_no_of_clusters = len(np.unique(meanshift_cluster_allocation))

threshold_for_anomaly = 2
anomalies_meanshift = []

for i in range(no_of_rows):
	data_point = np.array(data_points.iloc[i])
	data_point_cluster_center = np.array(meanshift_cluster_centers[meanshift_cluster_allocation[i]])
	k = data_point - data_point_cluster_center
	distance = (np.dot(k,k))**0.5
	if(distance > threshold_for_anomaly):
		#print("anomaly")
		anomalies_meanshift.append(i)

"""
Part-7 Producing a Hierarichal clustering using minikmeans Clustering prediction for anomalies
"""

minikmeans = MiniBatchKMeans(n_clusters = 10).fit(model_input)

minikmeans_cluster_allocation = minikmeans.labels_
minikmeans_cluster_centers = minikmeans.cluster_centers_

threshold_for_anomaly = 2
anomalies_minikmeans = []

for i in range(no_of_rows):
	data_point = np.array(data_points.iloc[i])
	data_point_cluster_center = np.array(minikmeans_cluster_centers[minikmeans_cluster_allocation[i]])
	k = data_point - data_point_cluster_center
	distance = (np.dot(k,k))**0.5
	if(distance > threshold_for_anomaly):
		anomalies_minikmeans.append(i)

"""
Part-8 Hierarchical Clustering
"""

agglomerative = AgglomerativeClustering(n_clusters = 10).fit(model_input)

threshold_for_anomaly = 1.5
anomalies_agglomerative = []

agglomerative_cluster_allocation = agglomerative.labels_
input_test = input_data_normalized_discretized
input_test['class_label'] = Series(agglomerative_cluster_allocation,index = input_test.index)

agglomerative_cluster_centers = dict()
for i in range(len(np.unique(agglomerative_cluster_allocation))):
	agglomerative_cluster_centers[i] = input_test.loc[input_test['class_label'] == i][model_vars].mean()

for i in range(no_of_rows):
	data_point = np.array(data_points.iloc[i])
	data_point_cluster_center = np.array(agglomerative_cluster_centers[agglomerative_cluster_allocation[i]])
	k = data_point - data_point_cluster_center
	distance = (np.dot(k,k))**0.5
	if(distance > threshold_for_anomaly):
		anomalies_agglomerative.append(i)

"""
Part-9 Setting up weighted majority voting for ensemble
"""

print("According to the ensemble of clustering algorithms anomalies by majority voting are")
for i in range(no_of_rows):
	if i in anomalies_meanshift:
		count = count + 2
	if i in anomalies_minikmeans:
		count = count + 1
	if i in anomalies_kmeans:
		count = count + 1
	if i in anomalies_agglomerative:
		count = count + 2
	if count > 2:
		print("Record no. " + str(i))
		print(input_data.iloc[[i]])
		print("\n")