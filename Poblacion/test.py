import pandas as pd
from geopy.geocoders import Nominatim
import time

# ---------------------------
# 1. Leer el Excel
# ---------------------------
df = pd.read_csv("Poblacion/2025_pad_mdbas.csv")  # tu archivo
# Debe tener columnas: Nom_Districte, Nom_Barri, Valor (poblacion)
df.rename(columns={'Valor': 'poblacion'}, inplace=True)

df = df.groupby(['Nom_Districte', 'Nom_Barri'], as_index=False)['poblacion'].sum()
print(f"Total barrios únicos: {len(df)}")

# ---------------------------
# 2. Inicializar geolocator
# ---------------------------
geolocator = Nominatim(user_agent="barcelona_dataset")

# ---------------------------
# 3. Geocodificar barrios
# ---------------------------
latitudes = []
longitudes = []

count = 0
lengths = len(df)
for i, row in df.iterrows():
    barrio = row['Nom_Barri']
    distrito = row['Nom_Districte']
    
    # Construir query con barrio + distrito + Barcelona
    query = f"{barrio}, {distrito}, Barcelona, Spain"
    
    try:
        location = geolocator.geocode(query)
        if location:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)
            print(f"No se encontró {query}")
    except Exception as e:
        latitudes.append(None)
        longitudes.append(None)
        print(f"Error geocoding {query}: {e}")
    
    # Evitar bloquear por límite de peticiones
    time.sleep(0.2)  # 1 segundo entre consultas
    print(f"Extracted {count+1}/{lengths}")
    count += 1

# ---------------------------
# 4. Añadir coordenadas al DataFrame
# ---------------------------
df['lat'] = latitudes
df['lon'] = longitudes

# ---------------------------
# 5. Calcular densidad (si tienes área en km2)
# ---------------------------
# Por ejemplo, si tienes columna 'area_km2':
# df['densidad'] = df['poblacion'] / df['area_km2']

# ---------------------------
# 6. Guardar dataset final
# ---------------------------
df.to_csv("barrios_barcelona_geocoded.csv", index=False)
print("Dataset guardado en barrios_barcelona_geocoded.csv")
