import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
import pickle


# Function to prepare data by engineering new features and removing unnecessary columns
def prepare_data(train):
    train.drop(columns=['id'], inplace=True)

    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['DayofMonth'] = train['pickup_datetime'].dt.day
    train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
    train['month'] = train['pickup_datetime'].dt.month
    train['hour'] = train['pickup_datetime'].dt.hour
    train['dayofyear'] = train['pickup_datetime'].dt.dayofyear

    # Function to calculate the Haversine distance between two points
    def haversine_array(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h



    train['distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                            train['pickup_longitude'].values,
                                            train['dropoff_latitude'].values,
                                            train['dropoff_longitude'].values)

    # Function to calculate the bearing between two points
    def bearing_array(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    train['bearing'] = bearing_array(train['pickup_latitude'].values,
                                            train['pickup_longitude'].values,
                                            train['dropoff_latitude'].values,
                                            train['dropoff_longitude'].values)

    # Function to calculate a dummy Manhattan distance
    def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
        a = haversine_array(lat1, lng1, lat1, lng2)
        b = haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    train['distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                        train['pickup_longitude'].values,
                                                                        train['dropoff_latitude'].values,
                                                                        train['dropoff_longitude'].values)





    train['distance_haversine'] = np.log1p(train.distance_haversine)
    train['distance_dummy_manhattan'] = np.log1p(train.distance_dummy_manhattan)
    train['log_trip_duration'] = np.log1p(train.trip_duration)

    train.drop(columns=['trip_duration','pickup_datetime'],inplace=True)


    return train


# Function to evaluate the model and print RMSE and R2 scores
def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


if __name__ == '__main__':
    # import  model
    model_path=''
    Kmeans_model_path=''

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(Kmeans_model_path, 'rb') as file:
        kmeans = pickle.load(file)

    # read data
    path = ''
    test = pd.read_csv(path)

    # prepar data
    test = prepare_data(test)
    test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
    test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

    numeric_features = ['distance_haversine', 'distance_dummy_manhattan', 'bearing']
    categorical_features = ['passenger_count', 'store_and_fwd_flag', 'vendor_id', 'pickup_cluster', 'dropoff_cluster',
                            'DayofMonth', 'dayofweek', 'month', 'hour', 'dayofyear']
    test_features = categorical_features + numeric_features


    # Evaluate the model on test data
    predict_eval(model, test, test_features, "test")