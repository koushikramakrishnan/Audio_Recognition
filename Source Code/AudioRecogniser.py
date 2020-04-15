import numpy as np
from numpy import loadtxt
from keras.models import load_model
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

print('===================================Audio Recogniser======================================')

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])

classes = ['Air conditioner' , 'Car horn' , 'Children playing' , 'Dog bark' , 'Drilling' , 'Engine Idling' , 'Gun Shot' , 'JackHammer' , 'Siren' , 'Street Music']



print('Enter the path of the Audio File')
source = input()
source = '../TestAudio/'+source

print('===================================MLP CLASSIFIER======================================')

start = datetime.now()



model_MLP = tf.keras.models.load_model("../SavedModels/model_final.h5")
prediction_MLP = model_MLP.predict([extract_feature(source)])

p1 = prediction_MLP.tolist()
p1=p1[0]



index = p1.index(max(p1))
print('The Given Audio as classified by MLP is : ',classes[index])


for i in range(len(p1)):
    print(classes[i],'\t %.32f'%(p1[i]))


print('Execution Time: ',datetime.now()-start)

print('===================================CNN CLASSIFIER======================================')

start = datetime.now()

max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs

model_CNN = tf.keras.models.load_model("../SavedModels/model_final_CNN.h5")

num_rows = 40
num_columns = 174
num_channels = 1
prediction_feature = extract_features(source) 
prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

prediction_CNN = model_CNN.predict([prediction_feature])


p1 = prediction_CNN.tolist()
p1=p1[0]



index = p1.index(max(p1))
print('The Given Audio as classified by CNN is : ',classes[index])


for i in range(len(p1)):
    print(classes[i],'\t %.32f'%(p1[i]))

print('Execution Time: ',datetime.now()-start)
