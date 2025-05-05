import streamlit as st
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración desde archivo local
from config import INFLUX_URL, INFLUX_TOKEN, ORG, BUCKET

# --- Cargar datos desde InfluxDB ---
def get_temperature_data():
    query = '''
    from(bucket: "homeiot")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "temperature")
    '''
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df = client.query_api().query_data_frame(org=ORG, query=query)
    df = df[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": "temperatura"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# --- Detección de anomalías con Isolation Forest ---
def detectar_anomalias(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["temperatura"]])
    return df

# --- Streamlit UI ---
st.title("Análisis de Temperatura con IA local")

if st.button("Cargar y analizar datos"):
    df = get_temperature_data()
    st.subheader("Datos crudos:")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas:")
    st.write(df["temperatura"].describe())

    df = detectar_anomalias(df)
    outliers = df[df["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="temperatura", data=df, label="Temperatura", ax=ax)
    ax.scatter(outliers["timestamp"], outliers["temperatura"], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)
    
def get_humidity_data():
    query2 = '''
    from(bucket: "homeiot")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "humidity")
    '''
    client2 = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df2 = client.query_api().query_data_frame(org=ORG, query2=query2)
    df2 = df2[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": "humedad"})
    df2["timestamp"] = pd.to_datetime(df["timestamp"])
    return df2
# --- Detección de anomalías con Isolation Forest ---
def detectar_anomalias(df2):
    model2 = IsolationForest(contamination=0.1, random_state=40)
    df2["anomaly"] = model2.fit_predict(df2[["humedad"]])
    return df2

# --- Streamlit UI ---
st.title("Análisis de humedad con IA local")
if st.button("Cargar y analizar datos"):
    df2 = get_humidity_data()
    st.subheader("Datos crudos:")
    st.dataframe(df2)

    st.subheader("Estadísticas descriptivas:")
    st.write(df2["humedad"].describe())

    df2 = detectar_anomalias(df2)
    outliers2 = df2[df2["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig2, ax = plt.subplots()
    sns.lineplot(x="timestamp", y="humedad", data=df, label="humedad", ax=ax)
    ax.scatter(outliers["timestamp"], outliers["humedad"], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig2)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)
