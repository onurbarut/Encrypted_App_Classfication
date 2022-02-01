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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Conv1D, MaxPooling1D

from utils.helper2 import *
from utils.fs_utils import *

def load_json_model(modelname):
    print("\nLoading the model from disk ...")
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(modelname+'.h5')
    print("Loaded model from disk")
    return loaded_model

with tf.device('/cpu:0'):

    ###
    dataset = "./data/ISCX_vpn-nonvpn2016" # "./data/NetML" or "./data/CICIDS2017" or "./data/non-vpn2016" or "./data/vpn2016" or "./data/ISCX_vpn-nonvpn2016"
    anno = "top" # or "mid" or "fine"
    #submit = "both" # or "test-std" or "test-challenge"

    # Assign variables
    training_set = dataset+"/2_training_set"
    training_anno_file = dataset+"/2_training_annotations/2_training_anno_"+anno+".json.gz"
    test_set = dataset+"/1_test-std_set"
    challenge_set = dataset+"/0_test-challenge_set"

    # Specify feature selection method # 'FSMJ', 'corr', 'RF', 'PCA'
    selection = 'RF'

    # Top selected features
    nTop = 10

    N = 10 # Number of experiments to average

    # TLS, DNS, HTTP features included?
    TLS , DNS, HTTP = {}, {}, {}
    TLS['tlsOnly'] = True # returns
    TLS['use'] = True
    TLS['n_common_client'] = 10
    TLS['n_common_server'] = 5
    #
    DNS['use'] = False
    ##
    ##
    #
    HTTP['use'] = False
    ##
    ##


    # Get training data in np.array format
    annotationFileName = dataset+"/2_training_annotations/2_training_anno_top.json.gz"

    # load Xtest
    # TLS, DNS, HTTP features included?
    TLS , DNS, HTTP = {}, {}, {}
    TLS['tlsOnly'] = False # returns
    TLS['use'] = True
    TLS['n_common_client'] = 10
    TLS['n_common_server'] = 5
    #annotationFileName = dataset+"/2_training_annotations/2_training_anno_top.json.gz"
    feature_names, ids, testXdata, testlabels, class_label_pair = read_dataset(training_set, annotationFileName=annotationFileName, TLS=TLS, class_label_pairs=None)


    # Drop flows if either num_pkts_in or num_pkts_out < 1
    test_df = pd.DataFrame(data=testXdata, columns=feature_names)
    test_df['label'] = testlabels
    isFiltered = test_df['num_pkts_in'] < 1
    test_f_df = test_df[~isFiltered]
    isFiltered = test_f_df['num_pkts_out'] < 1
    test_f_df = test_f_df[~isFiltered]
    test_target = test_f_df.pop('label')

    class_names = sorted(list(class_label_pair.keys()))
        
    #
    Xtrain, Xtest, ytrain, ytest = train_test_split(test_f_df.values, test_target.values, test_size=0.2, random_state=10, shuffle = True, stratify = test_target)

    #
    scaler = preprocessing.StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    test_Xtest_scaled = scaler.transform(Xtest)

    # load Xdata
    # TLS, DNS, HTTP features included?
    TLS , DNS, HTTP = {}, {}, {}
    TLS['tlsOnly'] = True # returns
    TLS['use'] = True
    TLS['n_common_client'] = 10
    TLS['n_common_server'] = 5

    # Get training data in np.array format
    feature_names, ids, Xdata, labels, class_label_pair = read_dataset(training_set, annotationFileName=annotationFileName, TLS=TLS, class_label_pairs=None)

    # Drop flows if either num_pkts_in or num_pkts_out < 1
    df = pd.DataFrame(data=Xdata, columns=feature_names)
    df['label'] = labels
    isFiltered = df['num_pkts_in'] < 1
    f_df = df[~isFiltered]
    isFiltered = f_df['num_pkts_out'] < 1
    f_df = f_df[~isFiltered]
    target = f_df.pop('label')


    # Train RF Model
    class_names = sorted(list(class_label_pair.keys()))
        
    #
    Xtrain, Xtest, ytrain, ytest = train_test_split(f_df.values, target.values, test_size=0.2, random_state=10, shuffle = True, stratify = target)

    #
    scaler = preprocessing.StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)

    #
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")

    # train
    clf_knn.fit(Xtrain_scaled, ytrain)
    clf_rf.fit(Xtrain_scaled, ytrain)    
    
    #
    print("kNN Test Score: {:.4f}".format(clf_knn.score(Xtest_scaled, ytest)))
    print("RF Test Score: {:.4f}".format(clf_rf.score(Xtest_scaled, ytest)))

    # Measure Training time
    # predict N times and get average for kNN
    pred_times = []
    for i in range(N):
        t0 = t.time()
        y_pred = clf_knn.predict(test_Xtest_scaled)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_knn_all = test_Xtest_scaled.shape[0]/(sum(pred_times)/len(pred_times))

    # predict N times and get average for RF
    pred_times = []
    for i in range(N):
        t0 = t.time()
        y_pred = clf_rf.predict(test_Xtest_scaled)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_rf_all = test_Xtest_scaled.shape[0]/(sum(pred_times)/len(pred_times))

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

    # Get Top-10
    # Let's retrain with Top-10 features
    # Performance measurements [t_train, acc, t_pred, Nmiss, Nfalse]
    #f_df['label'] = target.values
    #
    try:
        target = f_df.pop('label')
    except:
        pass

    to_drop = [featureList[i] for i in indices[:-nTop]]
    f_df_10 = f_df.drop(to_drop, axis=1)
    test_f_df_10 = test_f_df.drop(to_drop, axis=1)
    #
    _, test_Xtest_10, _, _ = train_test_split(test_f_df_10.values, test_target.values, test_size=0.2, random_state=42, shuffle = True, stratify = test_target)


    #
    Xtrain, Xtest, ytrain, ytest = train_test_split(f_df_10.values, target.values, test_size=0.2, random_state=42, shuffle = True, stratify = target)

    #
    scaler = preprocessing.StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    test_Xtest_scaled_top10 = scaler.transform(test_Xtest_10)

    #
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
    #clf = svm.SVC(C=1.0, kernel='rbf', random_state=10)
    #clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(int(2*len(list(df.columns))),), random_state=10)
    #clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(174,), random_state=10)

    # train 
    clf_knn.fit(Xtrain_scaled, ytrain)
    clf_rf.fit(Xtrain_scaled, ytrain)

    #
    if selection == 'RF':
        # RF based importances / indices  
        feature_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
        feature_rf.fit(Xtrain_scaled, ytrain)
        importances = feature_rf.feature_importances_  
        indices = np.argsort(importances)
        featureList = list(f_df_10.columns)
    else:
        indices = np.argsort(importances)
        # Get non-nan number
        nonNan = np.sum(~np.isnan(importances))
        # Move nan to the end of array
        indices = np.roll(indices, len(indices)-nonNan)
        indices = indices[-nTop:]
        #importances = importances[indices]
        #indices = np.argsort(importances)

    #
    print("kNN Top-10 Test Score: {:.4f}".format(clf_knn.score(Xtest_scaled, ytest)))
    print("RF Top-10 Test Score: {:.4f}".format(clf_rf.score(Xtest_scaled, ytest)))

    # predict N times and get average for kNN
    pred_times = []
    for i in range(N):
        t0 = t.time()
        y_pred = clf_knn.predict(test_Xtest_scaled_top10)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_knn_top10 = test_Xtest_scaled_top10.shape[0]/(sum(pred_times)/len(pred_times))

    # predict N times and get average for RF
    pred_times = []
    for i in range(N):
        t0 = t.time()
        y_pred = clf_rf.predict(test_Xtest_scaled_top10)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_rf_top10 = test_Xtest_scaled_top10.shape[0]/(sum(pred_times)/len(pred_times))



    # load 1D CNN and perform inference
    class_names = sorted(list(class_label_pair.keys()))
        
    #
    Xtrain, Xtest, ytrain, ytest = train_test_split(f_df.values, target.values, test_size=0.2, random_state=10, shuffle = True, stratify = target)

    #
    scaler = preprocessing.StandardScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    learning_rate = 1e-3
    decay_rate = 1e-5


    X_train_1D = Xtrain_scaled.reshape(-1,Xtrain_scaled.shape[1],1)
    X_val_1D = test_Xtest_scaled.reshape(-1,test_Xtest_scaled.shape[1],1)

    # Load Model
    modelname = '/media/onur/Data/PhD/RESEARCH/NetML-Competition2020/results/00_ISCX/1D_enc_noSMOTE_20200902-091636/model-20200902-091636'
    model = load_json_model(modelname)

    model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                    metrics=['accuracy'])


    # Predict N times and get average
    pred_times = []
    for i in range(N):
        t0 = t.time()
        ypred = model.predict(X_val_1D)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_1d = X_val_1D.shape[0]/(sum(pred_times)/len(pred_times))



    # load 2D CNN and perform inference
    X_train_2D = Xtrain_scaled.reshape(-1,Xtrain_scaled.shape[1])
    X_val_2D = test_Xtest_scaled.reshape(-1,test_Xtest_scaled.shape[1])    
    #X_train_2D = Xtrain_scaled.reshape(-1,11,16,1) #reshape 121 to 11x11
    #X_val_2D = test_Xtest_scaled.reshape(-1,11,16,1) #reshape 121 to 11x1

    # Load Model
    modelname = '/media/onur/Data/PhD/RESEARCH/NetML-Competition2020/results/00_ISCX/2D_enc_noSMOTE_20200902-125550/model-20200902-125550'
    model = load_json_model(modelname)

    model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                    metrics=['accuracy'])


    # Predict N times and get average
    pred_times = []
    for i in range(N):
        t0 = t.time()
        ypred = model.predict(X_val_2D)
        t1 = t.time()
        pred_times.append(t1-t0)
    perf_fps_2d = X_val_2D.shape[0]/(sum(pred_times)/len(pred_times))




    print("test_Xtest_scaled.shape:\t{}".format(test_Xtest_scaled.shape))
    print("Flow per second predicted by RF model with all Metadata+TLS features: {:.2f}".format(perf_fps_rf_all))

    print("test_Xtest_scaled_top10.shape:\t{}".format(test_Xtest_scaled_top10.shape))
    print("Flow per second predicted by RF model with Top-10 Metadata features: {:.2f}".format(perf_fps_rf_top10))

    print("test_Xtest_scaled.shape:\t{}".format(test_Xtest_scaled.shape))
    print("Flow per second predicted by kNN model with all Metadata+TLS features: {:.2f}".format(perf_fps_knn_all))

    print("test_Xtest_scaled_top10.shape:\t{}".format(test_Xtest_scaled_top10.shape))
    print("Flow per second predicted by kNN model with Top-10 Metadata features: {:.2f}".format(perf_fps_knn_top10))

    print("X_val_1D.shape:\t{}".format(X_val_1D.shape))
    print("Flow per second predicted by 1D CNN model: {:.2f}".format(perf_fps_1d)) 


    print("X_val_2D.shape:\t{}".format(X_val_2D.shape))
    print("Flow per second predicted by 2D CNN model: {:.2f}".format(perf_fps_2d)) 
