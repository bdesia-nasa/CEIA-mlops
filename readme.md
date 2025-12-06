# CEIA – Proyecto MLOps

Repositorio del TP de MLOps – Especialización en IA (CEIA - UBA).

Autor: Juan Nervi

Contiene:
- Infraestructura Docker
- Apache Airflow
- MLflow
- Servicios de inferencia
- Notebooks de experimentación

## 9. Arquitectura del proyecto

El proyecto levanta un stack de MLOps simple basado en **Docker Compose** con:

- **Airflow**: orquestador de workflows (entrenamiento, evaluación, deployment). docker pull apache/airflow:2.10.2
- **MLflow**: tracking de experimentos y modelos.
- **MinIO (S3 compatible)**: storage de artefactos (modelos, datasets, outputs).
- **Postgres**: base de datos para Airflow y backend de MLflow.
- **Rain Predictor Service**: servicio de inferencia (API) que consume el modelo registrado.

### 9.1. Diagrama lógico

```text
                 ┌───────────────────────────────┐
                 │           Airflow             │
                 │  (webserver + scheduler)      │
                 └──────────────┬────────────────┘
                                │
                                │ orquesta DAGs
                                │
        ┌───────────────────────┴───────────────────────────────┐
        │                                                       │
        v                                                       v
┌───────────────────────┐                           ┌───────────────────────┐
│        MLflow         │                           │        Postgres       │
│  - UI tracking         │<------------------------->│  - DB Airflow         │
│  - Registry modelos    │     conexión SQL          │  - DB MLflow backend  │
└─────────┬─────────────┘                           └───────────────────────┘
          │
          │ guarda artefactos (modelos, metrics files, etc.)
          v
┌───────────────────────┐
│         MinIO         │
│  (S3 compatible)      │
│  - bucket mlflow      │
│  - bucket data        │
└─────────┬─────────────┘
          │
          │ lee el modelo "en producción"
          v
┌──────────────────────────────┐
│  Fuel Channel Diameter API   │
│   (diameter-predictor)       │
│  - llama a MLflow            │
│  - descarga modelo           │
│  - sirve predicciones        │
└──────────────────────────────┘











# Miscelaneas
Para lanzar el docker compose, en un power shell lanzar: docker compose up -d
  
docker compose up postgres s3 -d 
docker compose  up airflow-init airflow-scheduler airflow-cli minio-bucket-init airflow-webserver -d


## Pipeline de instalación y ejecución

### 1. Prerrequisitos

- **Git**
- **Docker Desktop** (con al menos **4 GB de RAM** asignados al engine)
- **VS Code** (opcional pero recomendado)
- Sistema probado en **Windows 10/11 + WSL2** / Linux

---

### 2. Descargar o actualizar el repositorio

#### Primera vez (clonar)

```bash
# Ir a la carpeta donde quieras clonar el proyecto
cd C:\Users\<TU_USUARIO>\Documents

# Clonar el repo
git clone https://github.com/jern10/CEIA-mlops.git

# Entrar a la carpeta del proyecto
cd CEIA-mlops

cd C:\Users\<TU_USUARIO>\Documents\CEIA-mlops
git pull origin main

# airflow configuration
AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_PROJ_DIR=./airflow
AIRFLOW_PORT=8080
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# postgres configuration
# PG_USER=airflow
# PG_PASSWORD=airflow
# PG_DATABASE=airflow
# PG_PORT=5433

# mlflow configuration
MLFLOW_PORT=5001
# MLFLOW_BUCKET_NAME=

#levantar servicios
docker compose up -d
#ver estado de contenedores
docker compose ps

| Servicio           | URL / Host local                                       | Comentario                          |
| ------------------ | ------------------------------------------------------ | ----------------------------------- |
| **Airflow UI**     | [http://localhost:8080](http://localhost:8080)         | Usuario: `airflow`, pass: `airflow` |
| **Postgres**       | `localhost:5433`                                       | DB: `airflow` (por defecto)         |
| **MinIO (UI)**     | [http://localhost:9001](http://localhost:9001)         | Access key/secret en `.env`         |
| **MinIO (S3 API)** | [http://localhost:9000](http://localhost:9000)         | Usado por MLflow y el servicio      |
| **MLflow UI**      | [http://localhost:5000](http://localhost:5000) ó 5001* | Según cómo mapees `MLFLOW_PORT`     |

| Servicio          | Host interno        | Puerto interno |
| ----------------- | ------------------- | -------------- |
| Postgres          | `postgres`          | `5432`         |
| MinIO (S3)        | `s3`                | `9000`         |
| MLflow            | `mlflow`            | `5000`/`5001`  |
| Airflow webserver | `airflow_webserver` | `8080`         |

#Apagar servicios
docker compose down

#borrar volumenes
docker compose down -v




