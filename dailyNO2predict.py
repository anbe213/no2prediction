# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

!pip install seaborn --upgrade
!pip install plot-keras-history


import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
import folium

import tensorflow as tf
from plot_keras_history import plot_history

train = pd.read_csv("../input/no2dataset/train.csv")
train.head()

train.shape

test = pd.read_csv("../input/no2dataset/test.csv")

gdf = gpd.GeoDataFrame(train, geometry = gpd.points_from_xy(train.lon, train.lat))
states = "../input/no2dataset/shapefile/gadm36_VNM_2.shp"
m = folium.Map(location = [18.961188266766655, 105.92066503677678], zoom_start = 7)

for row in range(gdf.shape[0]):
    folium.Circle([gdf['lat'][row], gdf['lon'][row]], radius=1).add_to(m)

m

train['sentiment5P'].fillna(float(train['sentiment5P'].notnull().mean()), inplace = True)
train['sentiment5P'].mean()

test['sentiment5P'].fillna(int(test['sentiment5P'].mean()), inplace = True)

train.drop(train[train['NO2'].isnull()].index, inplace = True)

print('The data has', (train.isnull().sum()), "missing values.")

train.shape

train.drop(train[train.NO2 > 1000].index , inplace = True)

print('Row number: ', train.shape[0], 'and Column Number: ', train.shape[1], '.')
train.describe()

train_num = train.select_dtypes(include = ['float64', 'int64']).drop(axis = 1 , labels = (['lon', "lat","time"]))
train_num.head()

train_num.hist(figsize = (16,20), bins = 50, xlabelsize = 8, ylabelsize = 8);

for i in range(1, len(train_num.columns), 5):
    sns.pairplot(data = train_num,
                x_vars = train_num.columns[i:i+5],
                y_vars = ["NO2"])

x_train=train[['pblh', 'pressure', 'temperature', 'relative_humidity','wind_speed', 'dpt', 'road_density', 'population_density', 'sentiment5P']]

y_train = train["NO2"]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size = 500)



model = tf.keras.Sequential()


model.add(tf.keras.layers.Dense(units = 128))

model.add(tf.keras.layers.Dense(units = 16))

model.add(tf.keras.layers.Dense(units = 8))

model.add(tf.keras.layers.Dense(units = 4))


model.add(tf.keras.layers.Dense(units = 1, input_shape = (7,)))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = .0005), loss = 'mean_squared_error')

history = model.fit(x_train, y_train, validation_split = 0.2, epochs = 5000)

plot_history(history)
plt.show()

predict_x_train = model.predict(x_train).flatten()

train["Predicted_NO2"]= predict_x_train

train["absolute_error"] = train["NO2"]- train["Predicted_NO2"]

fig, ax = plt.subplots(1, 1)

print(train["absolute_error"].describe())

Monitor_plot = sns.histplot(train["absolute_error"], kde = True, stat = "density")

Monitor_plot.set_xlabel("Absolute error (ppb)")




Q1 = train["absolute_error"].quantile(0.25)
Q3 = train["absolute_error"].quantile(0.75)

outlier_IQR_high = round(((Q3 - Q1 * 1.5) + Q3), 2)
outlier_IQR_low = round((Q1 - (Q3 - Q1 * 1.5)), 2)
print("IQR Outliers are greater than " + str(outlier_IQR_high) + " and less than "+ str(outlier_IQR_low))

plt.xlabel("Absolute Error (ppb)", fontsize = 20 )
plt.xticks(size = 20)
plt.ylabel("Density", fontsize=20 )
plt.yticks(size = 20)

plt.savefig('Absolute_dis_Final.png')

