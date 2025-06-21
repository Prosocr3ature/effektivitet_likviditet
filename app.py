import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import openai
from openai.error import OpenAIError

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="📈 Försäljningslogg & Affärer", layout="wide")
st.markdown("# 📈 Försäljningslogg & Affärer")

# --- DATABASE SETUP ---
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS logg (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  datum TEXT UNIQUE,
  samtal INTEGER,
  tid_min INTEGER,
  tb REAL,
  tb_per_samtal REAL,
  tb_per_timme REAL,
  snitt_min_per_samtal REAL,
  lon REAL,
  kommentar TEXT,
  energi INTEGER,
  humor INTEGER
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS affarer (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  datum TEXT,
  affar_namn TEXT,
  skickad_tid TEXT,
  stangd_tid TEXT,
  minuter_till_stangning REAL,
  tb REAL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS mal (
  datum TEXT PRIMARY KEY,
  tb_mal INTEGER,
  samtal_mal INTEGER,
  lon_mal INTEGER,
  specifikt TEXT,
  realistiskt TEXT,
  tidsbundet TEXT
)
""")
conn.commit()

# --- OPENAI CLIENT SETUP ---
openai_key = st.secrets.get("OPENAI_KEY")
if openai_key:
    client = openai.OpenAI(api_key=openai_key)
else:
    client = None
    st.warning("Ingen OPENAI_KEY satt i secrets – GPT-summering inaktiverad")

# --- MOTIVATION MESSAGE ---
msgs = [
  "🔥 Disciplin slår motivation – varje dag!",
  "🚀 Varje samtal är en ny chans till rekord.",
  "💪 Fokusera på process, inte bara resultat.",
  "🏆 Du tävlar mot dig själv – slå gårdagens du.",
  "🌱 Små förbättringar bygger stora framgångar."
]
st.info(random.choice(msgs))

# --- SIDEBAR KPI METRICS ---
df_all = pd.read_sql_query("SELECT * FROM logg", conn)
if not df_all.empty:
    total_calls = df_all["samtal"].sum()
    total_tb    = df_all["tb"].sum()
    total_time  = df_all["tid_min"].sum()
    total_aff   = len(pd.read_sql_query("SELECT * FROM affarer", conn))
    conv_rate   = total_calls and (total_aff / total_calls) or 0
    tb_per_min  = total_time and (total_tb / total_time) or 0
    st.sidebar.metric("Konverteringsgrad", f"{conv_rate:.1%}")
    st.sidebar.metric("TB per minut", f"{tb_per_min:.2f} kr")

# --- ML SETUP: Recommendation & Clustering ---
df_model = df_all[["samtal","tid_min","tb"]].dropna()
if len(df_model) >= 5:
    X = df_model[["samtal","tid_min"]]; y = df_model["tb"]
    rec_model = RandomForestRegressor(n_estimators=50, random_state=0)
    rec_model.fit(X, y)
else:
    rec_model = None

df_a = pd.read_sql_query("SELECT minuter_till_stangning,tb FROM affarer", conn)
if len(df_a) >= 5:
    scaler = StandardScaler()
    Xa = scaler.fit_transform(df_a)
    kmeans = KMeans(n_clusters=2, random_state=0)
    df_a["cluster"] = kmeans.fit_predict(Xa)
else:
    scaler = None; kmeans = None

# --- INPUT FORM ---
col1, col2, col3 = st.columns([1.5,1,1])

with col1:
    st.subheader("🗓️ Dagslogg")
    datum   = st.date_input("Datum", datetime.today())
    samtal  = st.number_input("Antal samtal", min_value=0)
    tid_h   = st.number_input("Tid (timmar)", min_value=0)
    tid_m   = st.number_input("Tid (minuter)", min_value=0)
    tid_min = tid_h*60 + tid_m
    tb      = st.number_input("TB (kr)", min_value=0.0, step=100.0)
    kommentar = st.text_input("Kommentar")
    energi    = st.slider("Energinivå (1–5)", 1,5,3)
    humor     = st.slider("Humör (1–5)", 1,5,3)

    if st.button("💾 Spara dagslogg"):
        tb_ps = tb/samtal if samtal else 0
        tb_pt = tb/(tid_min/60) if tid_min else 0
        snitt = tid_min/samtal if samtal else 0
        lon   = tb*0.45
        cursor.execute("""
          INSERT OR REPLACE INTO logg
            (datum, samtal, tid_min, tb, tb_per_samtal, tb_per_timme,
             snitt_min_per_samtal, lon, kommentar, energi, humor)
          VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
          datum.strftime("%Y-%m-%d"),
          samtal, tid_min, tb,
          tb_ps, tb_pt, snitt, lon,
          kommentar, energi, humor
        ))
        conn.commit()
        st.success("Dagslogg sparad!")

with col2:
    st.subheader("🎯 Sätt SMART-mål")
    g_tb       = st.number_input("TB-mål", min_value=0, step=100)
    g_samtal   = st.number_input("Samtalsmål", min_value=0)
    g_lon      = st.number_input("Lönemål", min_value=0, step=100)
    specifikt  = st.text_input("Specifikt mål")
    realistiskt= st.text_input("Realistiskt mål")
    tidsbundet = st.text_input("Tidsbundet (YYYY-MM-DD)")

    if st.button("💾 Spara mål"):
        cursor.execute("""
          INSERT OR REPLACE INTO mal
            (datum,tb_mal,samtal_mal,lon_mal,
             specifikt,realistiskt,tidsbundet)
          VALUES (?,?,?,?,?,?,?)
        """, (
          datum.strftime("%Y-%m-%d"),
          g_tb, g_samtal, g_lon,
          specifikt, realistiskt, tidsbundet
        ))
        conn.commit()
        st.success("SMART-mål sparade!")

with col3:
    st.subheader("📤 Lägg till affär")
    aff_n = st.text_input("Affärsnamn")
    sent  = st.time_input("Skickad tid")
    closed= st.time_input("Stängd tid")
    tb_a  = st.number_input("TB för affären", min_value=0.0, step=100.0)

    if st.button("📌 Spara affär"):
        diff = (datetime.combine(datum, closed)
                - datetime.combine(datum, sent)).seconds/60
        cursor.execute("""
          INSERT INTO affarer
            (datum, affar_namn, skickad_tid, stangd_tid,
             minuter_till_stangning, tb)
          VALUES (?,?,?,?,?,?)
        """, (
          datum.strftime("%Y-%m-%d"),
          aff_n, sent.strftime("%H:%M"),
          closed.strftime("%H:%M"),
          diff, tb_a
        ))
        conn.commit()
        st.success("Affär sparad!")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
  "📊 Dag", "📋 Affärer", "🏆 Analys", "🎯 Målhistorik"
])

# 📊 Dag
with tab1:
    df = pd.read_sql_query("SELECT * FROM logg ORDER BY datum DESC", conn)
    if not df.empty:
        df["datum"] = pd.to_datetime(df["datum"])
        st.dataframe(df, use_container_width=True)
        fig, ax = plt.subplots()
        df.plot(x="datum", y=["tb","lon"], ax=ax, marker="o")
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)

        if rec_model:
            inc = st.slider("Öka samtal med (%)", -50,100,0)
            new_calls = df.iloc[0]["samtal"]*(1+inc/100)
            pred = rec_model.predict([[new_calls, df.iloc[0]["tid_min"]]])[0]
            st.write(f"➡️ Om du ökar samtalen med {inc}% → TB ≈ {pred:.0f} kr")

# 📋 AFFÄRER
with tab2:
    fr = st.date_input("Från", datetime.today()-timedelta(days=30))
    till = st.date_input("Till", datetime.today())
    df2 = pd.read_sql_query(
      "SELECT * FROM affarer WHERE datum BETWEEN ? AND ? ORDER BY datum",
      conn, params=(fr.strftime("%Y-%m-%d"), till.strftime("%Y-%m-%d"))
    )
    st.dataframe(df2, use_container_width=True)
    if kmeans and scaler and not df2.empty:
        Xc = scaler.transform(df2[["minuter_till_stangning","tb"]])
        df2["cluster"] = kmeans.predict(Xc)
        fig, ax = plt.subplots()
        ax.scatter(df2["minuter_till_stangning"], df2["tb"], c=df2["cluster"], cmap="viridis")
        ax.set_xlabel("Tid (min)") 
        ax.set_ylabel("TB")      
        st.pyplot(fig)

# 🏆 ANALYS
with tab3:
    df3 = pd.read_sql_query("SELECT * FROM logg", conn)
    if not df3.empty:
        df3["datum"] = pd.to_datetime(df3["datum"])
        df3["vecka"] = df3["datum"].dt.isocalendar().week
        weekly = df3.groupby("vecka")[["tb","samtal","lon"]].sum()
        st.subheader("Veckosammanställning")
        st.dataframe(weekly)
        st.line_chart(weekly[["tb","lon"]])

        # GPT‐summering via nya API
        if client:
            prompt = (
              f"Veckorapport vecka {weekly.index[-1]}, "
              f"totalt TB {weekly['tb'].iloc[-1]:.0f} kr. "
              "Ge en kort, peppande sammanfattning på svenska."
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}]
                )
                st.markdown(resp.choices[0].message.content)
            except OpenAIError as e:
                st.warning(f"GPT‐fel: {e}")

# 🎯 MÅLHISTORIK
with tab4:
    dfg = pd.read_sql_query("SELECT * FROM mal ORDER BY datum DESC", conn)
    dfl = pd.read_sql_query("SELECT * FROM logg ORDER BY datum DESC", conn)
    if not dfg.empty and not dfl.empty:
        dfg["datum"] = pd.to_datetime(dfg["datum"])
        dfl["datum"] = pd.to_datetime(dfl["datum"])
        st.dataframe(dfg, use_container_width=True)
        day = st.selectbox("Välj dag", dfg["datum"].dt.strftime("%Y-%m-%d"))
        m = dfg[dfg["datum"]==pd.to_datetime(day)].iloc[0]
        l = dfl[dfl["datum"]==pd.to_datetime(day)].iloc[0]
        pct = l["tb"]/m["tb_mal"]*100 if m["tb_mal"] else 0
        st.write(f"TB-uppfyllelse: {pct:.0f}%")
        # Simulerings‐heatmap
        calls0, tb0 = l["samtal"], l["tb"]
        avg0 = tb0/calls0 if calls0 else 0
        sr = np.arange(max(1,int(calls0*0.7)), int(calls0*1.5)+1)
        tr = np.linspace(avg0*0.8, avg0*1.3, 10)
        mat = np.outer(sr, tr)
        fig, ax = plt.subplots()
        c = ax.imshow(mat, origin="lower", aspect="auto",
                      extent=[tr[0],tr[-1],sr[0],sr[-1]])
        fig.colorbar(c, ax=ax, label="TB")
        ax.set_xlabel("Snitt‐TB")
        ax.set_ylabel("Samtal")
        st.pyplot(fig)

# --- EXCEL EXPORT ---
buf = io.BytesIO()
pd.read_sql_query("SELECT * FROM logg", conn).to_excel(buf, index=False)
st.download_button("📥 Ladda ner logg.xlsx", data=buf.getvalue(), file_name="logg.xlsx")
