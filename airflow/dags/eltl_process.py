# ----------------------------------------------------------------

import datetime

# Airflow utilities
from airflow.decorators import dag, task

# ----------------------------------------------------------------

# Set default dag parameters.
default_args = {
    'owner': 'Juan Nervi',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

md_text = """
### ELTL Process for fuel_channel_diameter
"""

@dag(
    dag_id="eltl_fuel_channel_diameter",
    description="ELTL process for fuel_channel_diameter using TaskFlow, getting data from source, splitting and transforming it.",
    doc_md=md_text,
    tags=["ELTL", "fuel_channel_diameter"],
    default_args=default_args,
    catchup=False,
    schedule_interval=None,
)

def process_eltl_ria():

    # Some general parameters
    s3_data_path = "s3://data/"     # Directory where data is stored

    # --------------------------------------
    # FIRST STEP: extract and load the data
    # --------------------------------------

    @task.virtualenv(
        task_id="get_rawdata",
        requirements=["awswrangler==3.6.0",
                        "geopy==2.2.0",],
    )
    def get_rawdata(url):
        """
        Load the raw data from source into S3 bucket
        """
        from utilities import download_data
        import logging

        logger = logging.getLogger("airflow.task")

        # Call the auxiliary function and get the status
        status = download_data(url)
        for message in status:
            logger.info(message)

    @task.virtualenv(
            task_id="coords_by_loc",
            requirements=[  "pandas~=1.5",
                            "geopy==2.2.0",
                            "awswrangler==3.6.0",
                        ],
        )

    def get_coords_by_loc(s3_data_path):
        """
        Get the coordinates of the locations in the dataset 
        and save them into S3 bucket
        """
        from utilities import get_geocoord
        import logging
        import awswrangler as wr
        import pandas as pd

        logger = logging.getLogger("airflow.task")

        # Load the dataset from S3
        s3_input_path = s3_data_path + "RawData/weatherAUS.csv"
        logger.info("Reading dataset from : {s3_data_path}/RawData")
        rain_df = wr.s3.read_csv(s3_input_path)
        logger.info("Dataset reading successfully finished")

        logger.info("Getting locations")
        location_df = pd.DataFrame(rain_df['Location'].unique(),columns=['Location']).sort_values(by='Location')
        
        logger.info("Getting coordinates by location")
        place_coords = {}
        for location in location_df['Location']:
            place_coords[location] = get_geocoord(location)

        place_coords_df = pd.DataFrame(list(place_coords.items()),  
                            columns=["Location", "Coordinates"]
                            ).sort_values(by='Location')

        place_coords_df[['Latitude', 'Longitude']] = pd.DataFrame(
                                                    place_coords_df['Coordinates'].tolist(), 
                                                    index=place_coords_df.index
                                )
        
        place_coords_df = place_coords_df.drop(columns='Coordinates')

        # Saving datasets into S3
        output_prefix = s3_data_path + "TransformedData/"
        logger.info("Saving coordinates by location into: %s", output_prefix)
        wr.s3.to_csv(df=place_coords_df, path=f"{output_prefix}coords_by_loc.csv", index=False)

        logger.info("Coordinates by location successfully finished")

    # -----------------------------------------
    # SECOND STEP: transform and load the data
    # -----------------------------------------

    @task.virtualenv(
        task_id="split_dataset",
        requirements=[  "pandas~=1.5",
                        "scikit-learn==1.3.2",
                        "awswrangler==3.6.0",
                    ],
        )

    def split_dataset(s3_data_path):
        """
        Generates the dataset and gets test and evaluation set
        """
        import logging
        import numpy as np
        import awswrangler as wr
        import pandas as pd
        from sklearn.model_selection import train_test_split

        logger = logging.getLogger("airflow.task")

        # Load the dataset from S3
        s3_input_path = s3_data_path + "RawData/weatherAUS.csv"
        logger.info("Reading dataset from : {s3_data_path}/RawData")
        rain_df = wr.s3.read_csv(s3_input_path)
        logger.info("Dataset reading successfully finished")

        # Preprocess dataset
        logger.info("Getting features and label from dataset")
        X_full = rain_df.drop(columns=['RainTomorrow'])             # Drop the target column from features
        y_full = np.where(rain_df['RainTomorrow'] == "Yes", 1, 0)   # Target variable

        # Split the dataset
        logger.info("Splitting dataset")
        ftest = 0.20  # Data fraction for "training"
        logger.info("Test size: %s", ftest)
        logger.info("Stratify: True")

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, 
            test_size=ftest,        
            stratify=y_full,  # Keep class proportions the same       
        )

        logger.info("X_train dimension: %s", X_train.shape)
        logger.info("y_train dimension: %s", y_train.size)
        logger.info("X_test dimension: %s", X_test.shape)
        logger.info("y_test dimension: %s", len(y_test))

        # Convertir y_train y y_test en DataFrames
        y_train_df = pd.DataFrame(y_train, columns=["y_train"])
        y_test_df = pd.DataFrame(y_test, columns=["y_test"])

        # Saving datasets into S3
        output_prefix = s3_data_path + "TransformedData/"
        logger.info("Saving datasets into: %s", output_prefix)
        wr.s3.to_csv(df=X_train, path=f"{output_prefix}X_train.csv", index=False)
        wr.s3.to_csv(df=y_train_df, path=f"{output_prefix}y_train.csv", index=False)
        wr.s3.to_csv(df=X_test, path=f"{output_prefix}X_test.csv", index=False)
        wr.s3.to_csv(df=y_test_df, path=f"{output_prefix}y_test.csv", index=False)
        
        logger.info("Dataset splitting successfully finished")

    @task.virtualenv(
        task_id="prepro_pipeline",
        requirements=["pandas~=1.5", 
                      "scikit-learn==1.3.2", 
                      "awswrangler==3.6.0", 
                      "geopy==2.2.0",
                      "s3fs",
                      ]
    )

    def prepo_pipeline(s3_data_path):
        """
        Applys data pre-processing pipeline
        """
        
        import logging
        import pandas as pd
        import awswrangler as wr

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        import tempfile
        import pickle

        # Agregar el directorio de la librería a sys.path
        # import sys
        # import os
        # lib_path = os.path.abspath(r'..\..\service\src')
        # sys.path.append(lib_path)

        # Custom library For dataset analysis
        from pipeline import HierarchicalImputer
        from pipeline import CoordinateTransformer2
        from pipeline import WindDirectionTransformer
        from pipeline import ExtendedWindDirectionTransformer
        from pipeline import DropColumnsTransformer
        from pipeline import RainTodayTransformer
        from pipeline import ExpandDateTransformer
        from pipeline import CountNullsTransformer
        
        logger = logging.getLogger("airflow.task")
        
        X_train_path = s3_data_path + 'TransformedData/X_train.csv'
        X_test_path = s3_data_path + 'TransformedData/X_test.csv'
        
        logger.info(f"Reading train and test dataset from : {s3_data_path}/TransformedData")
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        logger.info("Train and Test Datasets reading successfully finished")        

        coords_by_loc_path = s3_data_path + 'TransformedData/coords_by_loc.csv'
        logger.info("Recovering coords by location from : {s3_data_path}/TransformedData")
        coords_by_loc_df = pd.read_csv(coords_by_loc_path)
        logger.info("Coordinates by location recovering successfully finished")        

        coords_by_loc_dic = coords_by_loc_df.set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
        place_coords = {loc: (coords['Latitude'], coords['Longitude']) for loc, coords in coords_by_loc_dic.items()}

        # Define PIPELINE for data treatment flow
        prepro_pipeline = Pipeline(steps = [
            ("date_expander", ExpandDateTransformer()),                                                                         # Split date in day/month/year columns
            ("imputer", HierarchicalImputer()),                                                                                 # Missing imputation                                                                  
            ("coordinates", CoordinateTransformer2(place_coords)),                                                              # Convert "Location" into Latitude and Longitude
            ("wind_direction", WindDirectionTransformer()),                                                                     # Convert cardinal direction into degree
            ("wind_direction_deg", ExtendedWindDirectionTransformer()),                                                         # Convert degree into cos and sin
            ("drop_directions", DropColumnsTransformer(columns=["WindGustDir", "WindDir9am", "WindDir3pm",])),                  # Drop categorical columns
            ("drop_directions_deg", DropColumnsTransformer(columns=["WindGustDirDeg", "WindDir9amDeg", "WindDir3pmDeg",])),     # Drop auxiliary columns
            ("drop_date_location", DropColumnsTransformer(columns=["Date","Location"])),                                        # Drop categorical columns
            ("rain_today", RainTodayTransformer()),                                                                             # Convert binary variable
            ("null_count", CountNullsTransformer()),
            ("minmax", MinMaxScaler()),                                                                                         # Scale features
        ])

        logger.info('Applying pre-processing pipeline: Fit and transform Train data')
        X_train_transformed = prepro_pipeline.fit_transform(X_train)
        X_train_transformed_df = pd.DataFrame(X_train_transformed)

        logger.info('Applying pre-processing pipeline: Transforming test data based on training fit')
        X_test_transformed = prepro_pipeline.transform(X_test)
        X_test_transformed_df = pd.DataFrame(X_test_transformed)

        # Saving datasets into S3
        output_prefix = s3_data_path + "TransformedData/"
        logger.info("Saving transformed datasets into: %s", output_prefix)
        wr.s3.to_csv(df=X_train_transformed_df, path=f"{output_prefix}X_train_transformed.csv", index=False)
        wr.s3.to_csv(df=X_test_transformed_df, path=f"{output_prefix}X_test_transformed.csv", index=False)
        logger.info("Dataset features transform successfully finished")
    
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pickle.dump(prepro_pipeline, tmp)
            local_pickle_path = tmp.name

        # Subir el pickle a S3
        s3_pickle_path = output_prefix + 'prepro_pipeline.pkl'
        wr.s3.upload(local_pickle_path, s3_pickle_path)

    # -----------------------------------------
    # WORKFLOW
    # -----------------------------------------

    # URL Base de datos principal
    url = 'https://www.kaggle.com/api/v1/datasets/download/jsphyg/weather-dataset-rattle-package'

    task_rawdata = get_rawdata(url)
    task_coords_by_loc = get_coords_by_loc(s3_data_path)
    task_split = split_dataset(s3_data_path)
    task_prepro = prepo_pipeline(s3_data_path)

    task_rawdata >> task_coords_by_loc >> task_split >> task_prepro

# Inicia el DAG
dag = process_eltl_ria()