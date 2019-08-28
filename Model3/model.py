import pickle
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from keras.utils import np_utils

#DATAPREP

# model building
model_01 = Sequential()
model_01.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model_01.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model_01.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# model training
evolution_01 = model_01.fit(x=payload_train_x, y=payload_train_y_oneHot, validation_split=0.2, epochs=5, batch_size=128, verbose=2)



# model performance
prediction_01 = model_01.predict_classes(payload_test_x)
confusion_metrics_dict = {'pred': prediction_01,'truth': payload_test_y}
confusion_metrics_df = pd.DataFrame.from_dict(confusion_metrics_dict) 

score_01 = model_01.evaluate(payload_test_x, payload_test_y_oneHot)
comparison_metrics_dict = {'accuracy': [score_01[1]]}
comparison_metrics_df = pd.DataFrame(comparison_metrics_dict, columns=['accuracy'])



import sasmm
project = '影像辨識專案'
model = '模型 1'
sasmm.generateMetrics(project, model, confusion_metrics_df, 'confusion_metrics')
sasmm.generateMetrics(project, model, comparison_metrics_df, 'comparison_metrics') 



# model saving
pickle.dump(model_01, open("/tsmc_model/model_files/模型 1.pkl", "wb"))

#RT_METRICS:confusion_metrics_df
#BT_METRICS:comparison_metrics_df
