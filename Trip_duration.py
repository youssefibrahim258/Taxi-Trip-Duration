import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
import pickle

# Function to evaluate the model and print RMSE and R2 scores
def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

# Main approach function that creates and trains the model pipeline
def approach1(train, test):
    numeric_features = ['distance_haversine','distance_dummy_manhattan','bearing']
    categorical_features = ['passenger_count','store_and_fwd_flag','vendor_id','pickup_cluster', 'dropoff_cluster',
                            'DayofMonth','dayofweek','month','hour','dayofyear']
    train_features = categorical_features + numeric_features



    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('poly', PolynomialFeatures(degree=2)),
        ('regression', Ridge(alpha=50))
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")

    return model


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

# Function to clean data by removing outliers and invalid entries
def CleanData(train):
    m = np.mean(train['trip_duration'])
    s = np.std(train['trip_duration'])
    train = train[train['trip_duration'] <= m + 2 * s]
    train = train[train['trip_duration'] >= m - 2 * s]

    train = train[train['pickup_longitude'] <= -73.75]
    train = train[train['pickup_longitude'] >= -74.03]
    train = train[train['pickup_latitude'] <= 40.85]
    train = train[train['pickup_latitude'] >= 40.63]
    train = train[train['dropoff_longitude'] <= -73.75]
    train = train[train['dropoff_longitude'] >= -74.03]
    train = train[train['dropoff_latitude'] <= 40.85]
    train = train[train['dropoff_latitude'] >= 40.63]

    train[train['passenger_count'] == 0] = np.nan
    train.dropna(axis=0, inplace=True)
    return train

# Function to create clusters for pickup and dropoff locations
def cluster_features(train, n=13, random_state=42):
    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values))

    np.random.seed(random_state)
    sample_ind = np.random.permutation(len(coords))[:500000]

    kmeans = MiniBatchKMeans(n_clusters=n, batch_size=10000, random_state=random_state).fit(coords[sample_ind])

    train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

    return train, kmeans


if __name__ == '__main__':

    train = pd.read_csv(r'Y:\01 ML\Projects\02 Trip Duration Prediction\Data\train.csv')
    test = pd.read_csv(r"Y:\01 ML\Projects\02 Trip Duration Prediction\Data\val.csv")

    train=CleanData(train)

    train, kmeans = cluster_features(train, n=100, random_state=42)
    test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
    test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

    train=prepare_data(train)
    test=prepare_data(test)
    model=approach1(train, test)

    # Save the trained KMeans model into a pickle file
    root_dir='Y:\\01 ML\\Projects\\02 Trip Duration Prediction\\finish'
    with open(os.path.join(root_dir, 'kmeans_model.pkl'), 'wb') as file:
        pickle.dump(kmeans, file)

    # Save the trained Ridge model into a pickle file
    with open(os.path.join(root_dir,'model.pkl'),'wb' )as file:
        pickle.dump(model,file)



