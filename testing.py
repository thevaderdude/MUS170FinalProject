import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('saved_model/my_model')

x = joblib.load('[gac][pop_roc]0642__2.joblib')
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(x)
probas = np.zeros(predictions.shape[1])
for i in range(len(probas)):
    probas[i] = np.mean(predictions[:, i])
# print(probas)
# print(predictions)
inst_labels = {
    'cel': 0,
    'cla': 1,
    'flu': 2,
    'gac': 3,
    'gel': 4,
    'org': 5,
    'pia': 6,
    'sax': 7,
    'tru': 8,
    'vio': 9,
}
label_inst = { value : key for key, value in inst_labels.items()}
# print(inst_labels.keys())
plt.bar(inst_labels.values(), probas, tick_label=list(inst_labels.keys()))
plt.ylabel('Probability')
plt.show()
