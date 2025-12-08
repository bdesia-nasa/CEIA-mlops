# ----------------------------------------------------------------
# DAG: Diametral Deformation of CNE NPP fuel channels - ETL process
# ----------------------------------------------------------------

import os
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# ----------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------

default_args = {
    'owner': 'Juan Nervi',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15),
}

md_text = """
# ETL - Diametral deformation of CNEE pressure tubes (ISI 2024)

Full Pipeline:
- Augment neutronic flux database for PT.
- Process inner diameter and thickness measurements.
- Calculate diametral change rates
- Train/Val/Test split by channel
"""

# Directories inside S3 bucket
S3_BASE_PATH = "s3://cne_fc_db/"                            # Directory where the whole data is stored.
RAW_DATA_PATH = os.path.join(S3_BASE_PATH, "raw/")          # Directory where raw data is stored.
INTERIM_PATH = os.path.join(S3_BASE_PATH, "interim/")       # Directory where intermedia data is stored. This data is partially processed.
PROCESSED_PATH = os.path.join(S3_BASE_PATH, "processed/")   # Directory where processed data is stored. This data is ready for modelling.

@dag(
    dag_id="diametral_deformation_etl",
    description="ETL process for diameter deformation of CNE NPP pressure tubes",
    doc_md=md_text,
    tags=["deformacion", "diametral", "isi2024", "nuclear","etl"],
    default_args=default_args,
    catchup=False,
    schedule_interval=None,
    start_date=days_ago(1),
    )

def etl_pipeline():
    
    @task
    def get_evaluation_time() -> float:
        import os
        import awswrangler as wr
        
        df = wr.s3.read_csv(f"{RAW_DATA_PATH}evaluation_time.csv")
        time_efph = float(df.iloc[0, 0])
        print(f"Evaluation time loaded: {time_efph} EFPH")
        return time_efph


    # ----------------------------------------------------------------
    # 1. Augment neutronic flux db for PT
    # ----------------------------------------------------------------

    @task.virtualenv(
        task_id="generate_extended_flux",
        requirements=[
            "pandas>=1.5",
            "numpy>=1.21",
            "scipy>=1.9",
            "awswrangler==3.6.0",
        ],
        system_site_packages=False,
        )
    
    def generate_extended_flux():
        import os
        import numpy as np
        import pandas as pd
        import awswrangler as wr
        from scipy.optimize import curve_fit
        import warnings

        class FluxCurve:
            def __init__(self, kparam=5, nbundles=12):
                self.kparam = kparam
                self.nbundles = nbundles
                self.A = None
                self.B = None
                self.fitted = False

            def _model(self, x, A, B):
                global_func = A * np.sin(np.pi * x)
                local_func = B * (np.abs(np.sin(self.nbundles * np.pi * x))) ** (1/self.kparam)
                return global_func + local_func

            def fit(self, xdata, ydata):
                p0 = np.max(ydata) * np.array([1.03, 0.4])
                bounds = ([0.9 * p0[0], 0.2 * p0[1]], [1.5 * p0[0], 1.1 * p0[1]])
                try:
                    best_params, _ = curve_fit(self._model, xdata, ydata, p0=p0, bounds=bounds, maxfev=5000)
                    self.A, self.B = best_params
                    self.fitted = True
                    return self.A, self.B
                except:
                    warnings.warn("Fit failed")
                    self.fitted = False
                    return None, None

            def predict(self, xdata):
                if not self.fitted:
                    raise RuntimeError("Model not fitted")
                return self._model(xdata, self.A, self.B)

        def generate_for_type(input_s3_path, output_s3_path, prefix):
            df = wr.s3.read_csv(input_s3_path)
            os.makedirs("/tmp/flux_extended", exist_ok=True)

            for idx, row in df.iterrows():
                canal = row['Canal']
                ntrozos = len(row) - 1
                coord_rel = [0] + [(i - 0.5) / ntrozos for i in range(1, ntrozos + 1)] + [1]
                flux_1mev = [0] + [row[f'Trozo_{i}'] for i in range(1, ntrozos + 1)] + [0]

                model = FluxCurve()
                model.fit(np.array(coord_rel), np.array(flux_1mev))

                ext_coord = np.linspace(0, 1, 1200)
                ext_flux = model.predict(ext_coord)

                out_df = pd.DataFrame({'coord_rel': ext_coord, 'flux_1mev': ext_flux})
                local_path = f"/tmp/flux_extended/{canal}.flx"
                out_df.to_csv(local_path, index=False, sep='\t', header=False)

                s3_path = f"{output_s3_path}{canal}.flx"
                wr.s3.upload(local_path, s3_path)

            print(f"Flux extendido generado: {prefix}")

        # PT
        generate_for_type(
            f"{RAW_DATA_PATH}flux/pt_flux_1MeV.csv",
            f"{INTERIM_PATH}flux/pressure_tubes_extended/",
            "PT"
        )

    # ----------------------------------------------------------------
    # 2. Process raw data (for diameter and thickness)
    # ----------------------------------------------------------------

    @task.virtualenv(
        task_id="process_raw_measurements",
        requirements=["pandas>=1.5", 
                      "numpy", 
                      "awswrangler==3.6.0"],
        )
    
    def process_raw_measurements():
        import pandas as pd
        import numpy as np
        import awswrangler as wr
        import os

        folder_diam = f"{RAW_DATA_PATH}isi2024/diameter/"
        folder_thick = f"{RAW_DATA_PATH}isi2024/thickness/"

        # Read all files
        diam_files = [f for f in wr.s3.list_objects(folder_diam) if f.endswith('.csv')]
        thick_files = [f for f in wr.s3.list_objects(folder_thick) if f.endswith('.csv')]

        idiam_by_ch = {}
        thick_by_ch = {}

        for file in diam_files:
            ch = os.path.basename(file)[21:24]
            df = wr.s3.read_csv(file, skiprows=15)
            idiam_by_ch[ch] = df

        for file in thick_files:
            ch = os.path.basename(file)[25:28]
            df = wr.s3.read_csv(file, skiprows=15)
            thick_by_ch[ch] = df

        # Unified axial coords based on diameter data
        rawdata_by_ch = {}
        for ch in idiam_by_ch:
            if ch not in thick_by_ch:
                continue
            df_d = idiam_by_ch[ch]
            df_t = thick_by_ch[ch]

            interpolated_thick = np.interp(df_d['Axial'], df_t['Axial'], df_t['Mean'])

            unified = pd.DataFrame({
                'Axial': df_d['Axial'],
                'MeanDiam': df_d['Mean'],
                'MeanThick': interpolated_thick
            })
            rawdata_by_ch[ch] = unified

        # Save in interim
        for ch, df in rawdata_by_ch.items():
            path = f"{INTERIM_PATH}unified_measurements/{ch}.csv"
            wr.s3.to_csv(df, path, index=False)

        return list(rawdata_by_ch.keys())

    # ----------------------------------------------------------------
    # 3. Enrich data and compute rates
    # ----------------------------------------------------------------

    @task.virtualenv(
        task_id="enrich_and_calculate_rates",
        requirements=["pandas",
                     "numpy", 
                     "awswrangler==3.6.0"],
        )
    
    def enrich_and_calculate_rates(channel_list: list, evaluation_time_efph: float):
        import pandas as pd
        import numpy as np
        import awswrangler as wr

        years = evaluation_time_efph / (24 * 365.25)  # más preciso que 365

        channel_data = wr.s3.read_csv(f"{RAW_DATA_PATH}channel_data.csv")

        results = {}

        for ch in channel_list:
            try:
                # Metadata by channel
                row = channel_data[channel_data['channel'] == ch].iloc[0]
                ch_length_mm = row['length(mm)']
                PT_idiam_0 = row['inner_diam(mm)']
                PT_thick_0 = row['thick(mm)']
                bm = row['bm_near(mm)']
                aface = row['a-face']

                PT_odiam_0 = PT_idiam_0 + 2 * PT_thick_0

                # Unified data
                df = wr.s3.read_csv(f"{INTERIM_PATH}unified_measurements/{ch}.csv")
                df['Axial_m'] = (df['Axial'] - bm) / 1000.0

                # Load P&T and flux data
                pt_path = f"{RAW_DATA_PATH}pressure_temperature/canales_BOL/{ch}.pt"
                flux_path = f"{INTERIM_PATH}flux/pressure_tubes_extended/{ch}.flx"

                pt_df = wr.s3.read_csv(pt_path, sep='\t')
                flux_df = wr.s3.read_csv(flux_path, sep='\t', header=None, names=['coord_rel', 'flux'])

                ch_length_m = ch_length_mm / 1000.0
                ct_length = pt_df['Length(m)'].max()
                dl = ch_length_m - ct_length
                pt_df['Length(m)'] += dl / 2

                flux_df['coord_real'] = flux_df['coord_rel'] * ch_length_m

                if aface == 'outlet':
                    pt_df['Pressure (Pa)'] = pt_df['Pressure (Pa)'][::-1].values
                    pt_df['Temperature (ºC)'] = pt_df['Temperature (ºC)'][::-1].values
                    flux_df = flux_df[::-1].reset_index(drop=True)

                # Interpolate P&T and neutronic flux
                P = np.interp(df['Axial_m'], pt_df['Length(m)'], pt_df['Pressure (Pa)'])
                T = np.interp(df['Axial_m'], pt_df['Length(m)'], pt_df['Temperature (ºC)'])
                F = np.interp(df['Axial_m'], flux_df['coord_real'], flux_df['flux'])

                # Tasas
                df['MeanRateDiam'] = (df['MeanDiam'] - PT_idiam_0) / years
                df['MeanRateThick'] = (df['MeanThick'] - PT_thick_0) / years
                df['MeanOutDiam'] = df['MeanDiam'] + 2 * df['MeanThick']
                df['MeanRateOutDiam'] = (df['MeanOutDiam'] - PT_odiam_0) / years
                df['MeanRateElon'] = - (df['MeanRateDiam'] / PT_idiam_0) - (df['MeanRateThick'] / PT_thick_0)

                df['Pressure'] = P
                df['Temperature'] = T
                df['Flux'] = F
                df['Channel'] = ch

                results[ch] = df

                # Guardar por canal
                wr.s3.to_csv(df, f"{PROCESSED_PATH}by_channel/{ch}.csv", index=False)

            except Exception as e:
                print(f"Error procesando canal {ch}: {e}")

        # Guardar dataset completo concatenado
        full_df = pd.concat(results.values(), ignore_index=True)
        wr.s3.to_csv(full_df, f"{PROCESSED_PATH}full_dataset.csv", index=False)

        return list(results.keys())

    # ----------------------------------------------------------------
    # 4. Train / val / test split (by channel)
    # ----------------------------------------------------------------

    @task
    def split_train_val_test(channel_list):
        import random
        random.seed(42)

        n = len(channel_list)
        test_frac = 0.30
        val_frac = 0.10

        n_test = int(n * test_frac)
        n_val = int(n * val_frac)

        test_channels = random.sample(channel_list, n_test)
        remaining = [c for c in channel_list if c not in test_channels]
        val_channels = random.sample(remaining, n_val)
        train_channels = [c for c in remaining if c not in val_channels]

        split_info = {
            "train": train_channels,
            "val": val_channels,
            "test": test_channels,
        }

        # Save lists
        import awswrangler as wr
        for name, chs in split_info.items():
            pd.DataFrame(chs, columns=["channel"]).to_csv(
                f"{PROCESSED_PATH}splits/{name}_channels.csv", index=False
            )

        return split_info

    # ----------------------------------------------------------------
    # 5. Generate final datasets (X, y) by group
    # ----------------------------------------------------------------

    @task.virtualenv(
        task_id="generate_final_datasets",
        requirements=["pandas", 
                      "awswrangler==3.6.0"],
    )
    def generate_final_datasets(split_info):
        import pandas as pd
        import awswrangler as wr

        full_df = wr.s3.read_csv(f"{PROCESSED_PATH}full_dataset.csv")

        features = ['Axial_m', 'Pressure', 'Temperature', 'Flux',
                    'MeanDiam', 'MeanThick', 'MeanOutDiam']  # ajusta según necesites

        target = 'MeanRateOutDiam'  # o MeanRateDiam, según modelo

        X = full_df[features]
        y = full_df[target]
        channels = full_df['Channel']

        for split_name, ch_list in split_info.items():
            mask = channels.isin(ch_list)
            X_split = X[mask]
            y_split = y[mask]

            wr.s3.to_csv(X_split.reset_index(drop=True),
                         f"{PROCESSED_PATH}{split_name}/X_{split_name}.csv", index=False)
            wr.s3.to_csv(pd.DataFrame(y_split).reset_index(drop=True),
                         f"{PROCESSED_PATH}{split_name}/y_{split_name}.csv", index=False)

        print("Final datasets generated: train, val, test")

    @task
    def save_run_metadata(evaluation_time_efph: float, split_info: dict):
        import json
        from datetime import datetime
        import awswrangler as wr
        
        metadata = {
            "run_date": datetime.utcnow().isoformat(),
            "evaluation_time_efph": evaluation_time_efph,
            "n_channels_total": len(split_info["train"]) + len(split_info["val"]) + len(split_info["test"]),
            "n_train": len(split_info["train"]),
            "n_val": len(split_info["val"]),
            "n_test": len(split_info["test"]),
            "dag_version": "v2025.03",
        }
    
        metadata_path = f"{PROCESSED_PATH}run_metadata/{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        wr.s3.upload(json.dumps(metadata, indent=2), metadata_path)

    # ----------------------------------------------------------------
    # Workflow
    # ----------------------------------------------------------------

    evaluation_time = get_evaluation_time()

    flux_task = generate_extended_flux()
    channels_after_raw = process_raw_measurements()
    enriched_channels = enrich_and_calculate_rates(channels_after_raw, evaluation_time)
    split = split_train_val_test(enriched_channels)
    final = generate_final_datasets(split)
    metadata = save_run_metadata(evaluation_time, split)

    flux_task >> channels_after_raw
    evaluation_time >> enriched_channels
    channels_after_raw >> enriched_channels >> split >> final >> metadata

# Ejecutar DAG
dag = etl_pipeline()