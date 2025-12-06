# CEIA – Proyecto MLOps

Repositorio del TP de MLOps – Especialización en IA (CEIA - UBA).

Autor: Juan Nervi

Contiene:
- Infraestructura Docker
- Apache Airflow
- MLflow
- Servicios de inferencia
- Notebooks de experimentación


# Notas
Para lanzar el docker compose, en un power shell lanzar: docker compose up -d
  
docker compose up postgres s3 -d 
docker compose  up airflow-init airflow-scheduler airflow-cli minio-bucket-init airflow-webserver -d


