import json
import argparse
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from utils.helper2 import *

def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def create_2D_CNN(X_train, OUTPUT,
                  encode_size=16, 
                  filters=64, 
                  kernel_size=3, 
                  strides=1,
                  dropout_rate=0., 
                  CNN_layers=2, 
                  clf_reg=1e-4):
    # Model Definition
    OUTPUTS = []
    #raw_inputs = Input(shape=(X_train.shape[1],X_train.shape[2],1))
    # Encode the 1D layer to 2D shape
    raw_inputs = Input(shape=(X_train.shape[1],))
    enc = Dense(encode_size*encode_size, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='Encoding_layer')(raw_inputs) 
    """    
    dec = Dense(X_train.shape[1], activation='tanh', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='decoder_output')(enc)
    OUTPUTS.append(dec)
    """
    xcnn = tf.keras.layers.Reshape((encode_size,encode_size,1), input_shape=(X_train.shape[1],))(enc)
    
    xcnn = Conv2D(filters, 
                    (kernel_size),
                    padding='same',
                    activation='relu',
                    strides=strides,
                    kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                    bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                    activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                    name='Conv2D_1')(xcnn)

    xcnn = BatchNormalization()(xcnn)                 
    xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

    if dropout_rate != 0:
        xcnn = Dropout(dropout_rate)(xcnn) 
    for i in range(1, CNN_layers):
        xcnn = Conv2D(filters,
                    (kernel_size),
                    padding='same',
                    activation='relu',
                    strides=strides,
                    kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                    bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                    activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                    name='Conv2D_'+str(i+1))(xcnn)

        xcnn = BatchNormalization()(xcnn)                 
        xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)  

    # we flatten for dense layer
    xcnn = Flatten()(xcnn)
    
    xcnn = Dense(32, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='FC1_layer')(xcnn)
    
    if dropout_rate != 0:
        xcnn = Dropout(dropout_rate)(xcnn) 
    
    xcnn = Dense(32, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='FC2_layer')(xcnn)            
     
    if dropout_rate != 0:
        xcnn = Dropout(dropout_rate)(xcnn) 
        
    if 'top' in OUTPUT: 
        top_level_predictions = Dense(OUTPUT['top'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xcnn)
        OUTPUTS.append(top_level_predictions)

    if 'mid' in OUTPUT: 
        mid_level_predictions = Dense(OUTPUT['mid'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='mid_level_output')(xcnn)
        OUTPUTS.append(mid_level_predictions)

    if 'fine' in OUTPUT: 
        fine_grained_predictions = Dense(OUTPUT['fine'], activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='fine_grained_output')(xcnn)
        OUTPUTS.append(fine_grained_predictions)    

    model = Model(inputs=raw_inputs, outputs=OUTPUTS)

    return model


dataset = "./data/ISCX_vpn-nonvpn2016" # "./data/NetML" or "./data/CICIDS2017" or "./data/non-vpn2016"
anno = "top" # or "mid" or "fine"
#submit = "both" # or "test-std" or "test-challenge"

# Assign variables
training_set = dataset+"/2_training_set"
training_anno_file = dataset+"/2_training_annotations/2_training_anno_"+anno+".json.gz"
test_set = dataset+"/1_test-std_set"
challenge_set = dataset+"/0_test-challenge_set"


# Create folder for the results
time_ = t.strftime("%Y%m%d-%H%M%S")
save_dir = os.getcwd() + '/results/' + time_
os.makedirs(save_dir)

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
#Xtrain, ytrain, class_label_pair, _ = get_training_data(training_set, training_anno_file)
#annotationFileName = [dataset+"/2_training_annotations/2_training_anno_top.json.gz", dataset+"/2_training_annotations/2_training_anno_fine.json.gz"]
annotationFileName = dataset+"/2_training_annotations/2_training_anno_top.json.gz"
feature_names, ids, Xtrain, ytrain, class_label_pairs = read_dataset(training_set, annotationFileName=annotationFileName, TLS=TLS, class_label_pairs=None)

# Drop flows if either num_pkts_in or num_pkts_out < 1
df = pd.DataFrame(data=Xtrain, columns=feature_names)
df['label'] = ytrain
isFiltered = df['num_pkts_in'] < 1
f_df = df[~isFiltered]
isFiltered = f_df['num_pkts_out'] < 1
f_df = f_df[~isFiltered]
target = f_df.pop('label')


# Split validation set from training data
X_train, X_val, y_train, y_val = train_test_split(f_df.values, target.values,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=target.values)
save_dict = {}
save_dict["SMOTE"] = True
# Resample training data using SMOTE
if save_dict["SMOTE"] == True:
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Get name of each class to display in confusion matrix
top_class_names = list(sorted(class_label_pairs.keys()))
#fine_class_names = list(sorted(class_label_pairs_list[1].keys()))


# Get model
# Default Training Hyperparameters
n_classes_top = len(top_class_names)
#n_classes_fine = len(fine_class_names)
learning_rate = 1e-3
decay_rate = 1e-5
dropout_rate = 0.5
n_batch = 100
n_epochs = 500  # Loop 500 times on the dataset
encode_size = 16
filters = 32
kernel_size = 3
strides = 1
CNN_layers = 2
clf_reg = 1e-5

save_dict['CNN_layers'] = CNN_layers
save_dict['encode_size'] = encode_size
save_dict['filters'] = filters
save_dict['kernel_size'] = kernel_size
save_dict['strides'] = strides
save_dict['clf_reg'] = clf_reg
save_dict['dropout_rate'] = dropout_rate
save_dict['learning_rate'] = learning_rate
save_dict['decay_rate'] = decay_rate
save_dict['n_batch'] = n_batch
save_dict['n_epochs'] = n_epochs

# Model Definition
#X_train_2D = X_train_scaled.reshape(-1,11,16,1) #reshape 176 to 11x16
#X_val_2D = X_val_scaled.reshape(-1,11,16,1) #reshape 176 to 11x16
# We know Encode  1D to 2D in the model. So we can input 1D shapes
X_train_2D = X_train_scaled.reshape(-1,X_train_scaled.shape[1])
X_val_2D = X_val_scaled.reshape(-1,X_val_scaled.shape[1])
model = create_2D_CNN(X_train_2D, {anno:n_classes_top}, 
                    encode_size = encode_size, filters=filters, kernel_size=kernel_size, 
                    strides=strides, dropout_rate=dropout_rate, 
                    CNN_layers=CNN_layers, clf_reg=clf_reg)

print(model.summary()) # summarize layers
plot_model(model, to_file=save_dir+'/model.png') # plot graph
model.compile(loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
    metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',
#                optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
#                metrics=['accuracy'])
# Train the model
history=model.fit(X_train_2D, one_hot(y_train, n_classes_top), batch_size=n_batch, epochs=n_epochs, validation_data=(X_val_2D, one_hot(y_val, n_classes_top)))

# Output accuracy of classifier
for k,v in history.history.items():
    print(k)
print("Training Score: \t{:.5f}".format(history.history['acc'][-1]))
print("Validation Score: \t{:.5f}".format(history.history['val_acc'][-1]))

# Print Confusion Matrix
ypred = model.predict(X_val_2D)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
try:
    plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred.argmax(1), 
                        classes=top_class_names, 
                        normalize=False)
except:
    plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred[1].argmax(1), 
                        classes=top_class_names, 
                        normalize=False)    

# Plot loss and accuracy
plotLoss(save_dir, history)

# Save the trained model and its hyperparameters
saveModel(save_dir, model, time_, save_dict, history)

