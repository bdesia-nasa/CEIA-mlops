import io
import boto3

def download_data(url):
    import requests
    import zipfile

    # Crear cliente S3 con las credenciales ambientales
    session = boto3.Session(
        aws_access_key_id=None,
        aws_secret_access_key=None,
        region_name=None
    )
    s3 = session.client('s3')

    # Operative: obtener los credenciales automáticamente si están env vars
    # (Boto3 las obtiene automáticamente si están configuradas en variables de entorno)

    status = []

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        status.append("Download completed from the URL")
        print(status[-1])

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            for file_info in zip_file.infolist():
                with zip_file.open(file_info) as file:
                    data_bytes = file.read()
                    key = f"RawData/{file_info.filename}"

                    # Subir datos en memoria usando boto3
                    s3.put_object(Bucket='data', Key=key, Body=data_bytes)

                    status.append(f"Uploaded {file_info.filename} to S3 at s3://data/{key}")
                    print(status[-1])
        status.append("All files successfully uploaded to S3")
        print(status[-1])
    else:
        status.append(f"Failed to download file. HTTP status code: {response.status_code}")
        print(status[-1])

    return status


from geopy.geocoders import Nominatim   # For GEO coords
import re                               # For string manipulation

def get_geocoord(place):
    'Función para obtener la latitud y longitud de un lugar "place" en Australia'
            
    country = 'Australia'
    # Divide las palabras (en caso de que sea necesario)
    place = re.sub(r'([a-z])([A-Z])', r'\1 \2', place)
            
    # Crear un objeto geolocalizador
    geolocator = Nominatim(user_agent="myGeocoder")
            
    location = f"{place}, {country}"
    location_info = geolocator.geocode(location)
            
    if location_info:
        return (location_info.latitude, location_info.longitude)
    else:
        return (None, None)