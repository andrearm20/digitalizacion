import streamlit as st
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

from config import INFLUX_URL, INFLUX_TOKEN, ORG, BUCKET

# --- Funciones para cargar datos ---
def get_data(field):
    query = f'''
    from(bucket: "homeiot")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "{field}")
    '''
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df = client.query_api().query_data_frame(org=ORG, query=query)
    df = df[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": field})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# --- Detección de anomalías ---
def detectar_anomalias(df, variable):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[[variable]])
    return df

# --- Interfaz Streamlit ---
st.title("Análisis de variables ambientales con IA local")

opcion = st.selectbox("Selecciona qué variable deseas analizar:", ["Humedad", "Temperatura", "Ambas"])

if st.button("Cargar y analizar datos"):
    if opcion in ["Humedad", "Ambas"]:
        df_hum = get_data("humidity")
        st.subheader("Datos de humedad:")
        st.dataframe(df_hum)
        st.write(df_hum["humidity"].describe())

        df_hum = detectar_anomalias(df_hum, "humidity")
        outliers_hum = df_hum[df_hum["anomaly"] == -1]

        st.subheader("Gráfico de humedad con anomalías:")
        fig, ax = plt.subplots()
        sns.lineplot(x="timestamp", y="humidity", data=df_hum, label="Humedad", ax=ax)
        ax.scatter(outliers_hum["timestamp"], outliers_hum["humidity"], color="red", label="Anomalía", zorder=5)
        ax.legend()
        st.pyplot(fig)
        st.subheader("Anomalías detectadas en humedad:")
        st.dataframe(outliers_hum)

    if opcion in ["Temperatura", "Ambas"]:
        df_temp = get_data("temperature")
        st.subheader("Datos de temperatura:")
        st.dataframe(df_temp)
        st.write(df_temp["temperature"].describe())

        df_temp = detectar_anomalias(df_temp, "temperature")
        outliers_temp = df_temp[df_temp["anomaly"] == -1]

        st.subheader("Gráfico de temperatura con anomalías:")
        fig, ax = plt.subplots()
        sns.lineplot(x="timestamp", y="temperature", data=df_temp, label="Temperatura", ax=ax)
        ax.scatter(outliers_temp["timestamp"], outliers_temp["temperature"], color="red", label="Anomalía", zorder=5)
        ax.legend()
        st.pyplot(fig)
        st.subheader("Anomalías detectadas en temperatura:")
        st.dataframe(outliers_temp)
