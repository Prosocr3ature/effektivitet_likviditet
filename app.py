import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import io

# Initiera databas
conn = sqlite3.connect('forsaljning.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
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
        kommentar TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS affarer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datum TEXT,
        affar_namn TEXT,
        skickad_tid TEXT,
        stangd_tid TEXT,
        minuter_till_stangning REAL,
        tb REAL
    )
''')
conn.commit()

st.title("📈 Försäljningslogg & Affärer")

# ---------------------- DAGSBLOCK ----------------------

st.header("🗓️ Fyll i dagens siffror")
datum = st.date_input("Datum", datetime.today())

# Hämta TB från affärer
cursor.execute("SELECT SUM(tb) FROM affarer WHERE datum = ?", (datum.strftime("%Y-%m-%d"),))
tb_summa = cursor.fetchone()[0] or 0.0

# Kontroll om dagslogg finns
cursor.execute("SELECT * FROM logg WHERE datum = ?", (datum.strftime("%Y-%m-%d"),))
existing_log = cursor.fetchone()

if existing_log:
    current_tb = existing_log[4]
    if abs(current_tb - tb_summa) > 1:
        st.warning(f"⚠️ TB i loggen ({current_tb} kr) skiljer sig från affärssumman ({tb_summa} kr)")
    if st.button("🔄 Synka TB från affärer"):
        samtal, tid_min, kommentar = existing_log[2], existing_log[3], existing_log[10]
        tb = tb_summa
        tb_per_samtal = tb / samtal if samtal > 0 else 0
        tb_per_timme = tb / (tid_min / 60) if tid_min > 0 else 0
        snitt_min_per_samtal = tid_min / samtal if samtal > 0 else 0
        lon = tb * 0.45

        cursor.execute('''
            UPDATE logg
            SET tb = ?, tb_per_samtal = ?, tb_per_timme = ?, snitt_min_per_samtal = ?, lon = ?
            WHERE datum = ?
        ''', (tb, tb_per_samtal, tb_per_timme, snitt_min_per_samtal, lon, datum.strftime("%Y-%m-%d")))
        conn.commit()
        st.success("TB synkad!")

# Dagsinput
samtal = st.number_input("Antal samtal", min_value=0, step=1)
tid_h = st.number_input("Tid (timmar)", min_value=0, step=1)
tid_m = st.number_input("Tid (minuter)", min_value=0, step=1)
tid_min = tid_h * 60 + tid_m
tb = st.number_input("TB (kr) – förifyllt från affärer", value=tb_summa, step=100.0)
kommentar = st.text_input("Kommentar/reflektion")

tb_per_samtal = tb / samtal if samtal > 0 else 0
tb_per_timme = tb / (tid_min / 60) if tid_min > 0 else 0
snitt_min_per_samtal = tid_min / samtal if samtal > 0 else 0
lon = tb * 0.45

if st.button("💾 Spara dagslogg"):
    cursor.execute('''
        INSERT OR REPLACE INTO logg
        (datum, samtal, tid_min, tb, tb_per_samtal, tb_per_timme, snitt_min_per_samtal, lon, kommentar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datum.strftime("%Y-%m-%d"), samtal, tid_min, tb, tb_per_samtal, tb_per_timme, snitt_min_per_samtal, lon, kommentar))
    conn.commit()
    st.success("Dagslogg sparad!")

# ---------------------- AFFÄRSBLOCK ----------------------

st.header("📤 Lägg till affär")
affar_namn = st.text_input("Affärsnamn")
skickad = st.time_input("Skickad tid")
stangd = st.time_input("Stängd tid")
tb_affar = st.number_input("TB för affären", min_value=0.0, step=100.0)

if st.button("📌 Spara affär"):
    skickad_dt = datetime.combine(datum, skickad)
    stangd_dt = datetime.combine(datum, stangd)
    minuter_diff = (stangd_dt - skickad_dt).total_seconds() / 60

    cursor.execute('''
        INSERT INTO affarer
        (datum, affar_namn, skickad_tid, stangd_tid, minuter_till_stangning, tb)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datum.strftime("%Y-%m-%d"), affar_namn, skickad.strftime("%H:%M"), stangd.strftime("%H:%M"), minuter_diff, tb_affar))
    conn.commit()
    st.success("Affär sparad!")

    # Om dagslogg finns → uppdatera TB automatiskt
    cursor.execute("SELECT * FROM logg WHERE datum = ?", (datum.strftime("%Y-%m-%d"),))
    row = cursor.fetchone()
    if row:
        samtal, tid_min, kommentar = row[2], row[3], row[10]
        cursor.execute("SELECT SUM(tb) FROM affarer WHERE datum = ?", (datum.strftime("%Y-%m-%d"),))
        tb_updated = cursor.fetchone()[0] or 0
        tb_per_samtal = tb_updated / samtal if samtal > 0 else 0
        tb_per_timme = tb_updated / (tid_min / 60) if tid_min > 0 else 0
        snitt_min_per_samtal = tid_min / samtal if samtal > 0 else 0
        lon = tb_updated * 0.45

        cursor.execute('''
            UPDATE logg
            SET tb = ?, tb_per_samtal = ?, tb_per_timme = ?, snitt_min_per_samtal = ?, lon = ?
            WHERE datum = ?
        ''', (tb_updated, tb_per_samtal, tb_per_timme, snitt_min_per_samtal, lon, datum.strftime("%Y-%m-%d")))
        conn.commit()
        st.info("TB uppdaterad automatiskt i dagsloggen!")

# ---------------------- ÖVERSIKTER ----------------------

st.header("📊 Dagslogg – Översikt")
df = pd.read_sql_query("SELECT * FROM logg ORDER BY datum", conn)
if not df.empty:
    df['datum'] = pd.to_datetime(df['datum'])
    st.dataframe(df.drop(columns=['id']), use_container_width=True)
    st.line_chart(df.set_index("datum")["snitt_min_per_samtal"])

st.header("📋 Affärer – Översikt")
df_affar = pd.read_sql_query("SELECT * FROM affarer ORDER BY datum", conn)
if not df_affar.empty:
    df_affar['datum'] = pd.to_datetime(df_affar['datum'])
    st.dataframe(df_affar.drop(columns=['id']), use_container_width=True)
    st.line_chart(df_affar.set_index("datum")["minuter_till_stangning"])
