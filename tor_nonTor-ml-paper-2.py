import os
import json
import time as t
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm, datasets, metrics, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE

from utils.helper2 import *
from utils.fs_utils import *

# Create folder for the results
time_ = t.strftime("%Y%m%d-%H%M%S")
directory = os.getcwd() + '/results/' + time_
os.makedirs(directory)

# Specify feature selection method # 'FSMJ', 'corr', 'RF', 'PCA'
selection = 'RF'

# Read data
#dataset = "NetML" # NetML or CICIDS2017
df_tor = pd.read_csv("ml-paper-2/CSV_data/tor2017_ffel48_filtered_fine.csv")
# Re-set the label from 0 to 1 for Tor class
#df_tor['label'] = 1
df_nontor = pd.read_csv("ml-paper-2/CSV_data/non-tor2017_ffel48_enc_filtered_fine.csv")
# Merge two datasets
#f_df = pd.concat([df_nontor, df_tor])
# If Scenario B, only tor dataset is used
class_label_pair = {"audio":0,"browsing":1,"chat":2,"email":3,"file-transfer":4,"P2P":5,"video":6,"voip":7}
#class_label_pair = {"audio":0,"browsing":1,"chat":2,"email":3,"file-transfer":4,"P2P":5,"video":6}
f_df = df_nontor
# If Scenario A, tor-noTor
#class_label_pair = {"non-tor":0,"tor":1}

target = f_df.pop('label')
from collections import Counter
print(Counter(target))

# Define metrics to be plotted at the end
topNfeatures = [f_df.values.shape[1], 10, 5]
# Accuracy
acc = []
# False Detection FDR
Nfalse = []
# False Omission FOR (miss)
Nmiss = []
# Training Time
t_train = []
# Prediction Time
t_pred = []

# Train RF Model
class_names = sorted(list(class_label_pair.keys()))

# Plot covariance matrix and feature correlations with target
plot_cov_matrix(directory, f_df, target)
correlation = plot_feature_correlation(directory, f_df, target)
    
#
Xtrain, Xtest, ytrain, ytest = train_test_split(f_df.values, target.values, test_size=0.2, random_state=10, shuffle = True, stratify = target)

#
scaler = preprocessing.StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

#
#clf = KNeighborsClassifier(n_neighbors=3)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
#clf = svm.SVC(C=1.0, kernel='rbf', random_state=10)
#clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(int(2*len(list(f_df.columns))),), random_state=10)
#clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(174,), random_state=10)
# Measure Training time
# train 10 times and get average
train_times = []
for i in range(1):
	t0 = t.time()
	clf.fit(Xtrain_scaled, ytrain)
	t1 = t.time()
	train_times.append(t1-t0)
t_train.append(sum(train_times)/len(train_times))

# Print Sorted Feature Importances
featureList = list(f_df.columns)
if selection == 'corr':
	# Correlation based importances
	importances = correlation
elif selection == 'FSMJ':
	# Jensen-Shannon Divergence based importanecs (0: same distribution -> not important, 1.0: max importance)
	max_distances = FSMJ(f_df, target, n_bins=100)
	importances = np.asarray(max_distances)
elif selection == 'RF':
	# RF based importances / indices	
	feature_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
	feature_rf.fit(Xtrain_scaled, ytrain)
	importances = feature_rf.feature_importances_
elif selection == 'PCA':
	cov_mat = np.cov(Xtrain_scaled.T)
	eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
	# Zero init. c array
	importances = np.zeros((eigen_vecs.shape[1]))
	for j in range(eigen_vecs.shape[1]):
	    for p in range(eigen_vecs.shape[0]):
	        importances[j] += abs(eigen_vecs[p,j])
else:
	raise ValueError("Not correct selection method chosen.")

indices = np.argsort(importances)
# Get non-nan number
nonNan = np.sum(~np.isnan(importances))
# Move nan to the end of array
indices = np.roll(indices, len(indices)-nonNan)

plot_feature_importance(directory, featureList, importances, indices, title='Feature Importances-'+selection)

# Output accuracy of classifier
accu = clf.score(Xtest_scaled, ytest)
acc.append(accu)
print("Test Score: {:.4f}".format(accu))

# Print Confusion Matrix
# predict 10 times and get average
pred_times = []
for i in range(1):
	t0 = t.time()
	y_pred = clf.predict(Xtest_scaled)
	t1 = t.time()
	pred_times.append(t1-t0)
t_pred.append(sum(pred_times)/len(pred_times))
y_test = ytest

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
_, cm = plot_confusion_matrix(directory, y_test, y_pred, classes=class_names)

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

Nmiss.append(cm[1,0]/(cm[1,0]+cm[1,1]))
Nfalse.append(cm[0,1]/(cm[0,0]+cm[0,1]))

# Get Top-10
subdirectory = directory + "/Top-10"
os.makedirs(subdirectory)
plot_feature_importance(subdirectory, featureList, importances, indices, nTop=10, title='Feature Importances before retrain-'+selection)

# Let's retrain with Top-10 features
performance = retrain(subdirectory, f_df, target, 10, featureList, importances, indices, class_names, selection)
t_train.append(performance[0])
acc.append(performance[1])
t_pred.append(performance[2])
Nmiss.append(performance[3])
Nfalse.append(performance[4])

# Get Top-5
subdirectory = directory + "/Top-5"
os.makedirs(subdirectory)
plot_feature_importance(subdirectory, featureList, importances, indices, nTop=5, title='Feature Importances before retrain-'+selection)

# Let's retrain with Top-5 features
performance = retrain(subdirectory, f_df, target, 5, featureList, importances, indices, class_names, selection)
t_train.append(performance[0])
acc.append(performance[1])
t_pred.append(performance[2])
Nmiss.append(performance[3])
Nfalse.append(performance[4])

# Plot performance metrics
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('nFeatures')
ax1.set_ylabel('%', color=color)
ax1.plot(topNfeatures, acc, label="Accuracy", color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time (s)', color=color)  # we already handled the x-label with ax1
ax2.plot(topNfeatures, t_pred, label="Prediction time", color=color, marker='x')
ax2.plot(topNfeatures, t_train, label="Training time", color='k', marker='s')
ax2.tick_params(axis='y', labelcolor=color)
#plt.gca().invert_xaxis() # uncomment if you want a descending order in nFeatures axis
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc='right', bbox_to_anchor=(1.0, 0.7))
ax2.legend(loc='right', bbox_to_anchor=(1.0, 0.4))
plt.savefig(directory+"/performance.png")