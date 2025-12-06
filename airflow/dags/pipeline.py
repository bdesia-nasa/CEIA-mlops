import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, accuracy_score
from pandas.core.series import Series
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from geopy.distance import geodesic


class HierarchicalImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_mean = None
        self.mean_by_location = None
        self.mean_by_location_month = None

        self.global_top = None
        self.top_by_location = None
        self.top_by_location_month = None

        self.mean_adjusted = ["Rainfall", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "MinTemp", "MaxTemp"]
        self.top_adjusted = ["WindGustDir", "WindDir9am", "WindDir3pm"]

    def is_null(self, value):
        if isinstance(value, str):
            return value == None
        if isinstance(value, Series) or isinstance(value, np.ndarray):
            return len(value) == 0
        if value is None:
            return True
        else:
            return np.isnan(value)

    def safely_extract_value(self, df, key, column):
        try:
            return df[key][column]
        except KeyError:
            return np.nan

    def impute_row_hierarchicaly(
            self,
            row,
            features,
            grouped_2_levels,
            grouped_1_level,
            global_values,
            l1_group_key = "Location",
            l2_group_key = "Month"
    ):
        for c in features:
            if not self.is_null(row[c]):
                continue
            imputation_value = self.safely_extract_value(grouped_2_levels, (row[l1_group_key], row[l2_group_key]), c)

            if self.is_null(imputation_value):
                imputation_value = self.safely_extract_value(grouped_2_levels, row[l1_group_key], c)

            if self.is_null(imputation_value):
                imputation_value = global_values[c]
            row[c] = imputation_value

        return row

    def fit(self, X, y = None):
        group_l1_l2 = lambda features: X[["Location", "Month"] + features].groupby(["Location", "Month"])
        group_l1 = lambda features: X[["Location"] + features].groupby(["Location"])

        self.mean_by_location_month = group_l1_l2(self.mean_adjusted).mean()
        self.mean_by_location = group_l1(self.mean_adjusted).mean()
        self.global_mean = X[self.mean_adjusted].mean()

        def mode(x):
            m = x.mode()
            if isinstance(x, Series) or isinstance(x, np.ndarray):
                if len(m) == 0:
                    return np.nan
                else:
                    return m[0]
            else:
                return m
        self.top_by_location_month = group_l1_l2(self.top_adjusted).agg(mode)
        self.top_by_location = group_l1(self.top_adjusted).agg(mode)
        self.global_top = X[self.top_adjusted].agg(mode)
        return self

    def transform(self, X):
        X_copy = X.copy()
        mean_imputer = lambda row: self.impute_row_hierarchicaly(row, self.mean_adjusted,  self.mean_by_location_month, self.mean_by_location, self.global_mean, l1_group_key = "Location", l2_group_key = "Month")
        top_imputer = lambda row: self.impute_row_hierarchicaly(row, self.top_adjusted, self.top_by_location_month, self.top_by_location, self.global_top, l1_group_key = "Location", l2_group_key = "Month")
        imputer = lambda row: top_imputer(mean_imputer(row))
        return X_copy.apply(imputer, axis=1)

class ExpandDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, year = True, month = True, week = False, day = False ):
        self.year = year
        self.month = month
        self.week = week
        self.day = day

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Date'] = pd.to_datetime(X['Date'])

        if self.year:
            X['Year'] = X['Date'].dt.year
        if self.month:
            X['Month'] = X['Date'].dt.month
        if self.week:
            X['Week'] = X['Week'].dt.week
        if self.day:
            X['Day'] = X['Day'].dt.day
        return X

class RainTodayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, NullCount = True):
        self.NullCount = NullCount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.NullCount:
            X["RainTodayNull"] = X["RainToday"].isnull().astype(np.int64)
            
        X["RainToday"] = X["RainToday"].apply(lambda rain: 1 if rain == "Yes" else 0)
        return X

# Custom transformer for merging location data and filling coordinates
class CoordinateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, coordinates, city_coords):
        self.coordinates = coordinates
        self.city_coords = city_coords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Merge coordinates
        merged = X.merge(self.coordinates[["Location", "lat", "lng"]], right_on="Location", left_on="Location", how="left")
        # Fill missing latitude and longitude
        merged.loc[merged["lat"].isnull(), "lat"] = merged.loc[
            merged["lat"].isnull(), "Location"
        ].apply(lambda loc: self.city_coords[loc][0] if loc in self.city_coords else None).astype(np.float64)

        merged.loc[merged["lng"].isnull(), "lng"] = merged.loc[
            merged["lng"].isnull(), "Location"
        ].apply(lambda loc: self.city_coords[loc][1] if loc in self.city_coords else None).astype(np.float64)

        return merged

class CoordinateTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, city_coords):
        self.city_coords = city_coords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Crear columnas 'Latitude' y 'Longitude' inicializadas a NaN
        X['Latitude'] = np.nan
        X['Longitude'] = np.nan

        # Llenar las columnas usando el diccionario city_coords
        for loc in X['Location']:
            if loc in self.city_coords:
                X.loc[X['Location'] == loc, 'Latitude'] = self.city_coords[loc][0]
                X.loc[X['Location'] == loc, 'Longitude'] = self.city_coords[loc][1]

        return X

# Custom transformer for handling wind directions
class WindDirectionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Define mapping of directions to degrees
        self.direction_to_degrees = {direction: i * 22.5 for i, direction in enumerate(
            ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        )}
        return self

    def transform(self, X):
        # Map directions to degrees
        X["WindGustDirDeg"] = X["WindGustDir"].map(self.direction_to_degrees)
        X["WindDir9amDeg"] = X["WindDir9am"].map(self.direction_to_degrees)
        X["WindDir3pmDeg"] = X["WindDir3pm"].map(self.direction_to_degrees)

        # Drop original direction columns
        # X.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm"], inplace=True)
        return X

# Inherited class that also computes the cosine and sine of wind directions
class ExtendedWindDirectionTransformer(WindDirectionTransformer):
    def transform(self, X):
        # Call the transform method of the base class
        X = super().transform(X)

        # Calculate the cosine and sine of the angles in radians
        X["WindGustDirCos"] = np.cos(np.radians(X["WindGustDirDeg"]))
        X["WindGustDirSin"] = np.sin(np.radians(X["WindGustDirDeg"]))

        X["WindDir9amCos"] = np.cos(np.radians(X["WindDir9amDeg"]))
        X["WindDir9amSin"] = np.sin(np.radians(X["WindDir9amDeg"]))

        X["WindDir3pmCos"] = np.cos(np.radians(X["WindDir3pmDeg"]))
        X["WindDir3pmSin"] = np.sin(np.radians(X["WindDir3pmDeg"]))

        return X

# Custom transformer for converting RainTomorrow to binary
class RainTomorrowTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["RainTomorrow"] = X["RainTomorrow"].apply(lambda rain: 1 if rain == "Yes" else 0)
        return X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)

# Custom transformer for counting nulls for instances
class CountNullsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Count null values per instance
        X['NullsCount'] = X.isnull().sum(axis=1)
        return X

class ShapeDebugger(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.shape)
        return X

class ColumnDebugger(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.columns)
        return X

def sample_array(X, y, sample_frac=0.1, stratify=None):
    combined_array = np.hstack((X, y.reshape(-1, 1)))
    y_len = len(y)
    sample_size = int(combined_array.shape[0] * sample_frac)

    if stratify is None:
        sampled_indices = np.random.choice(y_len, sample_size, replace=False)
    else:
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        stratify_len = len(stratify)
        sampled_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            cls_sample_size = int(sample_size * (class_counts[cls] / stratify_len))
            cls_sampled_indices = np.random.choice(cls_indices, cls_sample_size, replace=False)
            sampled_indices.extend(cls_sampled_indices)

    sampled_data = combined_array[sampled_indices]

    return sampled_data[:, :-1], sampled_data[:, -1]

def sample(X, y, sample = 0.1):
    df = X.copy()
    df["target"] = y

    if sample:
        df = df.sample(frac=sample)

    X_sampled = df.drop(columns=["target"])
    y_sampled = df["target"]
    return (X_sampled, y_sampled)

def eval_pipeline(pipe, X_train, y_train, sample_frac = 0.1):
    (X_train_sampled, y_train_sampled) = sample(X_train, y_train, sample_frac)

    pipe.fit(X_train_sampled, y_train_sampled)
    y_pred = pipe.predict(X_train_sampled)

    report_results(y_train_sampled, y_pred)
    return y_pred

def report_results(y, y_pred):
    # Evaluate the model
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\n Classification Report:\n", classification_report(y, y_pred))
    print("\n Roc auc Report:\n", roc_auc_score(y, y_pred))


def cross_eval_pipeline(pipe, X_train, y_train, sample = 0.1):
    scoring = ['precision_macro', 'f1', 'accuracy']
    cross_validate(pipe, X_train, y_train, scoring=scoring)

class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')

    def fit(self, X, y=None):
        self.kb.fit(X)
        return self

    def transform(self, X):
        X_binned = self.kb.transform(X)
        mult = X_binned * X['Rainfall'].to_frame().values
        X_combined = np.hstack([X, mult])
        return X_combined

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y =None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)



class DistanceFromCenterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, center_coordinates=(-25.0, 133.0)):  # Approximate center of Australia
        self.center_coordinates = center_coordinates

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate the distance from the center for each row
        X["DistanceFromCenter"] = X.apply(
            lambda row: geodesic((row["lat"], row["lng"]), self.center_coordinates).kilometers,
            axis=1
        )
        return X
