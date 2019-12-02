#!/usr/bin/env python
# coding: utf-8

# ## Install Dependencies

# In[86]:
from __future__ import absolute_import, division, print_function, unicode_literals

import random

# from IPython import get_ipython
#
# get_ipython().system('pip install -q matplotlib numpy pandas pathlib seaborn')
# get_ipython().system('pip install -q tensorflow ')
# get_ipython().system('pip install -q git+https://github.com/tensorflow/docs')

# !pip install -r -q requirements.txt


# ## Import Libraries

# In[87]:


import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# In[88]:


import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow import keras
from tensorflow.keras import layers

assert float(tf.__version__.split(".", 1)[0]) >= 2.0, "Please use Tensorflow version 2!"
print(tf.__version__)

# ### Load the transformed data
# Take the pkl file with extra feature columns into a dataframe

# In[89]:


dataset = pd.read_pickle('output/training_set_rel3.pkl')
dataset

# Inspect Columns

# In[90]:


dataset.dtypes

# In[91]:


dataset.isna().sum() > 0

# >You must extract a minimum of three different types of features.
#
# Please see `data_etl.ipynb` for details.
# ```
# meta_features = ['essay_length', 'avg_sentence_length', 'avg_word_length']
# grammar_features = ['sentiment', 'noun_phrases', 'syntax_errors']
# redability_features = ['readability_index', 'difficult_words']
# ```

# In[92]:


# dataset = dataset.dropna(axis='columns').drop(columns=['essay', 'essay_set'])
dataset = dataset.dropna(axis='columns').drop(columns=['essay'])


# In[93]:


def get_feature_combinations(dataset):
    attributes = list(dataset)
    attributes.remove('domain1_score')

    attribute_combinations = []
    for size in range(len(attributes)):
        attribute_combinations = attribute_combinations + list(itertools.combinations(attributes, size + 1))

    return attribute_combinations


feature_combinations = get_feature_combinations(dataset)

results = []
for feature_combination in feature_combinations:
    print("Selected feature_combination for Training: ", feature_combination)
    df = dataset.filter(list(feature_combination) + ['domain1_score'])
    print(df)

    # In[94]:

    train_dataset = df.sample(frac=0.7, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    # In[95]:

    train_stats = train_dataset.describe().pop("domain1_score").transpose()
    print(train_stats)

    # In[96]:

    train_labels = train_dataset.pop('domain1_score')
    test_labels = test_dataset.pop('domain1_score')


    # In[ ]:

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']


    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    print(normed_train_data)


    # In[ ]:

    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model


    # In[ ]:

    # Function to reset seeds for the sake of consistency
    def reset_seeds():
        SEED = 100
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)


    reset_seeds()
    model = build_model()

    # In[ ]:

    model.summary()

    # In[ ]:

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # In[ ]:

    model = build_model()

    EPOCHS = 1000
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

    early_history = model.fit(normed_train_data, train_labels,
                              epochs=EPOCHS, validation_split=0.2, verbose=0,
                              callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    # In[ ]:

    hist = pd.DataFrame(early_history.history)
    hist['epoch'] = early_history.epoch
    print(hist.tail())

    # In[ ]:

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    # In[ ]:

    plotter.plot({'Early Stopping': early_history}, metric="mae")
    plt.ylabel('MAE [domain1_score]')

    # In[ ]:

    plotter.plot({'Early Stopping': early_history}, metric="mse")
    plt.ylabel('MSE [domain1_score^2]')

    # In[ ]:

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} domain1_score".format(mae))

    # In[ ]:

    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [domain1_score]')
    plt.ylabel('Predictions [domain1_score]')
    lims = [0, 60]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    # >You should evaluate your system’s performance overall and for each subset of test essays using quadratic weighted kappa (https://www.kaggle.com/c/asap-aes/overview/evaluation).
    #
    # A weighted Kappa cab be used to calculate the similarity between predicted and actual score. A perfect score of close to 1.0 is granted when both the predictions and actuals are the same.
    # Whereas, the least possible score is -1 which is given when the predictions are furthest away from actuals.

    # In[ ]:

    print("\n\n\n")
    result = cohen_kappa_score(test_labels.values, test_predictions.astype(int), weights='quadratic')
    print("Model QWK({0}): {1}".format(feature_combination, result))

    # >You should compare the performance of your model to (at least) a baseline that predicts a random class for each test essay.

    # In[ ]:

    random_predictions = np.random.uniform(low=0, high=test_labels.values.max(), size=test_predictions.size)
    baseline = cohen_kappa_score(test_labels.values, random_predictions.astype(int), weights='quadratic')
    print("Baseline QWK({0}): {1}".format(feature_combination, baseline))

    # In[ ]:

    pct = (result - baseline) / baseline * 100
    print("Model performed {0} better than the baseline (random scoring) for {1}.".format(pct, feature_combination))
    print("\n\n\n")

    results.append((feature_combination, result, mse, mae))

results_df = pd.DataFrame(results, columns=['features', 'QWK', 'MSE', 'MAE'])
print(results_df)
