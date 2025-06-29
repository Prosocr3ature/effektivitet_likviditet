# app.py

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import numpy as np
import matplotlib.pyplot as plt
import openai
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# — PAGE SETTINGS —
st.set_page_config(page_title="DaVinci’s Duk", layout="wide")
st.title("🎨 DaVinci’s Duk")
st.markdown("💪 Fokusera på process, inte bara resultat.")

# — API-Key —
openai.api_key = st.secrets.get("OPENAI_KEY", "")
if not openai.api_key:
    st.error("❌ Saknar OpenAI-nyckel! Lägg till den i Streamlit Secrets.")

# — DATABASE — (cached init)
@st.cache_resource
def get_conn():
    conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
    return conn, conn.cursor()

conn, c = get_conn()

# Skapa tabeller (första gången)
c.execute("""CREATE TABLE IF NOT EXISTS logg (
    datum TEXT PRIMARY KEY,
    samtal INTEGER,
    tid_min INTEGER,
    tb REAL,
    energi INTEGER,
    humor INTEGER
)""")
c.execute("""CREATE TABLE IF NOT EXISTS affarer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datum TEXT,
    bolagstyp TEXT,
    foretagsnamn TEXT,
    abonnemang INTEGER,
    dealtyp TEXT,
    tb REAL,
    cashback REAL,
    margin REAL,
    minuter_till_stangning REAL,
    hw_count INTEGER,
    hw_type TEXT,
    hw_model TEXT,
    hw_cost REAL,
    hw_tb REAL
)""")
c.execute("""CREATE TABLE IF NOT EXISTS mal (
    datum TEXT PRIMARY KEY,
    daily_tb_goal REAL,
    daily_calls_goal INTEGER,
    monthly_tb_goal REAL
)""")
c.execute("""CREATE TABLE IF NOT EXISTS guldkunder (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orgnummer TEXT,
    kontaktperson TEXT,
    bindningstid TEXT,
    abonnemangsform TEXT,
    pris REAL,
    operatoerforsok INTEGER,
    har_kund_svarat TEXT,
    ovriga_abb_ja_nej TEXT,
    noteringar TEXT
)""")
c.execute("""CREATE TABLE IF NOT EXISTS aterkomster (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orgnummer TEXT,
    kontaktperson TEXT,
    aterkomstdatum TEXT,
    tema TEXT,
    noteringar TEXT
)""")
c.execute("""CREATE TABLE IF NOT EXISTS klara_kunder (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    orgnummer TEXT,
    kontaktperson TEXT,
    avslutsdatum TEXT,
    status TEXT,
    noteringar TEXT
)""")
conn.commit()

# — DAGLIG LOGG —
col1, col2 = st.columns([2,3])
idag = datetime.today().date()

with col1:
    st.subheader("🗓️ Dagslogg")
    samtal = st.number_input("Samtal", 0, step=1)
    tid_h = st.number_input("Tid (h)", 0, step=1)
    tid_m = st.number_input("Tid (min)", 0, step=1)
    tid_tot = tid_h * 60 + tid_m
    tb = st.number_input("TB", 0.0, step=100.0)
    energi = st.slider("Energi (1–5)", 1, 5, 3)
    humor = st.slider("Humör (1–5)", 1, 5, 3)

    if st.button("💾 Spara dagslogg"):
        c.execute("""INSERT OR REPLACE INTO logg VALUES (?,?,?,?,?,?)""",
                  (idag.strftime("%Y-%m-%d"), samtal, tid_tot, tb, energi, humor))
        conn.commit()
        st.success("Sparad!")

with col2:
    st.subheader("🎯 Mål")
    d_tb = st.number_input("TB-mål idag", 0.0, step=100.0)
    d_calls = st.number_input("Samtalsmål idag", 0, step=1)
    m_tb = st.number_input("Månads-TB-mål", 0.0, step=100.0)

    if st.button("💾 Spara mål"):
        c.execute("""INSERT OR REPLACE INTO mal VALUES (?,?,?,?)""",
                  (idag.strftime("%Y-%m-%d"), d_tb, d_calls, m_tb))
        conn.commit()
        st.success("Mål sparade!")

    row = c.execute("SELECT * FROM logg WHERE datum=?", (idag.strftime("%Y-%m-%d"),)).fetchone()
    goal = c.execute("SELECT * FROM mal WHERE datum=?", (idag.strftime("%Y-%m-%d"),)).fetchone()
    if row and goal:
        _, calls, tmin, tbt, _, _ = row
        _, g_tb, g_calls, g_month = goal
        st.write("**Dagsmål TB**"); st.progress(min(1.0, tbt / (g_tb or 1)))
        st.write("**Dagsmål samtal**"); st.progress(min(1.0, calls / (g_calls or 1)))
        month = idag.strftime("%Y-%m")
        total_tb = pd.read_sql_query("SELECT SUM(tb) FROM logg WHERE substr(datum,1,7)=?", conn, params=(month,)).iloc[0,0] or 0
        st.write("**Månads-TB**"); st.progress(min(1.0, total_tb / (g_month or 1)))

# — AFFÄRER MED HÅRDVARA —
st.markdown("---")
st.subheader("📤 Affärer + Hårdvara")
skickad = st.time_input("Skickad tid")
stangd = st.time_input("Stängd tid")
bolagstyp = st.selectbox("Bolagstyp", ["Enskild firma", "Aktiebolag"])
foretagsnamn = st.text_input("Företagsnamn")
abonnemang = st.number_input("Abonnemang", 0, step=1)
dealtyp = st.selectbox("Typ", ["Nyteckning", "Förlängning"])
tb_affar = st.number_input("TB", 0.0, step=100.0)
cashback = st.number_input("Cashback", 0.0, step=10.0)
margin = tb_affar - cashback
minuter = (datetime.combine(idag, stangd) - datetime.combine(idag, skickad)).seconds / 60
hw_count = st.number_input("Antal hårdvaror", 0, step=1)
hw_type = st.text_input("Typ av hårdvara")
hw_model = st.text_input("Modell")
hw_cost = st.number_input("Inköpspris", 0.0, step=10.0)
hw_tb = st.number_input("TB från hårdvara", 0.0, step=10.0)

if st.button("➕ Lägg till affär"):
    c.execute("""INSERT INTO affarer
        (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,tb,cashback,margin,
        minuter_till_stangning,hw_count,hw_type,hw_model,hw_cost,hw_tb)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (idag.strftime("%Y-%m-%d"), bolagstyp, foretagsnamn, abonnemang,
         dealtyp, tb_affar, cashback, margin, minuter,
         hw_count, hw_type, hw_model, hw_cost, hw_tb))
    conn.commit()
    st.success("Affär tillagd!")

# — FLIKAR —
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📋 Affärer", "📈 Översikt", "👑 Kunder", "📥 Export"])

# Tab1
with tab1:
    df = pd.read_sql_query("SELECT * FROM affarer WHERE datum=?", conn, params=(idag.strftime("%Y-%m-%d"),))
    st.dataframe(df, use_container_width=True)

# Tab2 – översikt + GPT
with tab2:
    df_log = pd.read_sql_query("SELECT * FROM logg", conn)
    if not df_log.empty:
        df_log['datum'] = pd.to_datetime(df_log['datum'])
        st.line_chart(df_log.set_index("datum")[["tb", "humor"]])
        latest = df_log[df_log["datum"] == pd.Timestamp(idag)]
        if not latest.empty:
            d = latest.iloc[0]
            prompt = f"Dagens försäljning {idag}: samtal={d['samtal']}, tb={d['tb']}. Ge 3 konkreta förbättringsförslag på svenska."
            try:
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
                st.markdown("**GPT-insikter:**  " + response.choices[0].message.content)
            except Exception as e:
                st.warning("GPT-analys misslyckades.")

# Tab3 – kunder
def editable(name, table, columns):
    st.subheader(name)
    df = pd.read_sql_query("SELECT * FROM " + table, conn)
    ed = st.data_editor(df[columns], num_rows="dynamic", use_container_width=True)
    if st.button(f"💾 Spara {name}"):
        c.execute(f"DELETE FROM {table}")
        for _, row in ed.iterrows():
            vals = tuple(row[col] for col in columns)
            c.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['?']*len(columns))})", vals)
        conn.commit()
        st.success(f"{name} sparad!")

with tab3:
    editable("⭐ Guldkunder", "guldkunder", [
        "orgnummer","kontaktperson","bindningstid","abonnemangsform",
        "pris","operatoerforsok","har_kund_svarat","ovriga_abb_ja_nej","noteringar"
    ])
    editable("🔁 Återkomster", "aterkomster", [
        "orgnummer","kontaktperson","aterkomstdatum","tema","noteringar"
    ])
    editable("✅ Klara kunder", "klara_kunder", [
        "orgnummer","kontaktperson","avslutsdatum","status","noteringar"
    ])

# Tab4 – export
with tab4:
    st.subheader("📥 Exportera rapport")
    if st.button("📦 Skapa rapport"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.read_sql_query("SELECT * FROM logg", conn).to_excel(writer, sheet_name="Logg", index=False)
            pd.read_sql_query("SELECT * FROM affarer", conn).to_excel(writer, sheet_name="Affärer", index=False)
            pd.read_sql_query("SELECT * FROM mal", conn).to_excel(writer, sheet_name="Mål", index=False)
            pd.read_sql_query("SELECT * FROM guldkunder", conn).to_excel(writer, sheet_name="Guldkunder", index=False)
            pd.read_sql_query("SELECT * FROM aterkomster", conn).to_excel(writer, sheet_name="Återkomster", index=False)
            pd.read_sql_query("SELECT * FROM klara_kunder", conn).to_excel(writer, sheet_name="Klara kunder", index=False)
        st.download_button("📥 Ladda ner rapport", output.getvalue(), f"rapport_{idag}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
