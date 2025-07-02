import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import io
import numpy as np
import openai

# --- SETUP PAGE ---
st.set_page_config(page_title="🎨 DaVinci’s Duk", layout="wide")
st.title("🎨 DaVinci’s Duk")
st.markdown("💪 Fokusera på process, inte bara resultat.")

# --- SETUP OPENAI ---
openai.api_key = st.secrets.get("OPENAI_KEY", "")

# --- SETUP DATABASE ---
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
c = conn.cursor()

# Skapa alla tabeller (körs automatiskt)
def init_db():
    c.execute("""CREATE TABLE IF NOT EXISTS logg (
        datum TEXT PRIMARY KEY, samtal INTEGER, tid_min INTEGER, tb REAL, energi INTEGER, humor INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS affarer (
        id INTEGER PRIMARY KEY AUTOINCREMENT, datum TEXT, bolagstyp TEXT, foretagsnamn TEXT, abonnemang INTEGER,
        dealtyp TEXT, tb REAL, cashback REAL, margin REAL, minuter_till_stangning REAL,
        hw_count INTEGER, hw_type TEXT, hw_model TEXT, hw_cost REAL, hw_tb REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS mal (
        datum TEXT PRIMARY KEY, daily_tb_goal REAL, daily_calls_goal INTEGER, monthly_tb_goal REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS guldkunder (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT, bindningstid TEXT,
        abonnemangsform TEXT, pris REAL, operatoerforsok INTEGER, har_kund_svarat TEXT, ovriga_abb_ja_nej TEXT, noteringar TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS aterkomster (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT, aterkomstdatum TEXT,
        tema TEXT, noteringar TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS klara_kunder (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT, avslutsdatum TEXT,
        status TEXT, noteringar TEXT)""")
    conn.commit()

init_db()  # Kör initialisering

# --- DAGLOGG OCH MÅL ---
col1, col2 = st.columns(2)
idag = datetime.today().date()

with col1:
    st.subheader("🗓️ Dagslogg")
    samtal = st.number_input("Antal samtal", 0, step=1)
    tid_tot = st.number_input("Total tid (min)", 0, step=5)
    tb = st.number_input("Total TB", 0.0, step=100.0)
    energi = st.slider("Energi (1-5)", 1, 5, 3)
    humor = st.slider("Humör (1-5)", 1, 5, 3)

    if st.button("💾 Spara logg"):
        c.execute("INSERT OR REPLACE INTO logg VALUES (?,?,?,?,?,?)",
                  (idag.strftime("%Y-%m-%d"), samtal, tid_tot, tb, energi, humor))
        conn.commit()
        st.success("Logg sparad!")

with col2:
    st.subheader("🎯 Sätt mål")
    daily_tb_goal = st.number_input("TB-mål (dag)", 0.0, step=100.0)
    daily_calls_goal = st.number_input("Samtalsmål (dag)", 0, step=1)
    monthly_tb_goal = st.number_input("TB-mål (månad)", 0.0, step=500.0)

    if st.button("💾 Spara mål"):
        c.execute("INSERT OR REPLACE INTO mal VALUES (?,?,?,?)",
                  (idag.strftime("%Y-%m-%d"), daily_tb_goal, daily_calls_goal, monthly_tb_goal))
        conn.commit()
        st.success("Mål sparat!")

# --- AFFÄRER OCH HÅRDVARA ---
st.subheader("📤 Lägg till affär + hårdvara")
skickad = st.time_input("Skickad tid")
stangd = st.time_input("Stängd tid")
minuter = (datetime.combine(idag, stangd) - datetime.combine(idag, skickad)).seconds / 60

with st.form("affars_form"):
    bolagstyp = st.selectbox("Bolagstyp", ["Enskild firma", "Aktiebolag"])
    foretagsnamn = st.text_input("Företagsnamn")
    abonnemang = st.number_input("Abonnemang antal", 0, step=1)
    dealtyp = st.selectbox("Typ", ["Nyteckning", "Förlängning"])
    tb_affar = st.number_input("Affär TB", 0.0, step=100.0)
    cashback = st.number_input("Cashback", 0.0, step=10.0)
    hw_count = st.number_input("Antal hårdvaror", 0, step=1)
    hw_type = st.text_input("Typ av hårdvara")
    hw_model = st.text_input("Modell")
    hw_cost = st.number_input("Inköpspris", 0.0, step=10.0)
    hw_tb = st.number_input("TB hårdvara", 0.0, step=10.0)

    if st.form_submit_button("➕ Lägg till affär"):
        c.execute("""INSERT INTO affarer (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,tb,cashback,margin,
                    minuter_till_stangning,hw_count,hw_type,hw_model,hw_cost,hw_tb)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (idag.strftime("%Y-%m-%d"), bolagstyp, foretagsnamn, abonnemang, dealtyp, tb_affar, cashback,
                   tb_affar - cashback, minuter, hw_count, hw_type, hw_model, hw_cost, hw_tb))
        conn.commit()
        st.success("Affär tillagd!")

# --- VISA DAGENS AFFÄRER ---
df_today = pd.read_sql("SELECT * FROM affarer WHERE datum=?", conn, params=(idag.strftime("%Y-%m-%d"),))
st.subheader("📋 Dagens affärer")
st.dataframe(df_today)

# --- GPT ANALYS ---
st.subheader("🤖 GPT Analys av dagen")
if not df_today.empty:
    prompt = f"Analysera dagens affärer: {df_today.to_dict('records')}, ge förbättringsförslag."
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": prompt}])
        st.info(response.choices[0].message.content)
    except Exception as e:
        st.warning("GPT-analys misslyckades.")

# --- EXCEL EXPORT ---
st.subheader("📥 Exportera data")
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    pd.read_sql("SELECT * FROM logg", conn).to_excel(writer, sheet_name="Logg", index=False)
    df_today.to_excel(writer, sheet_name="Affärer", index=False)
    pd.read_sql("SELECT * FROM mal", conn).to_excel(writer, sheet_name="Mål", index=False)
st.download_button("Ladda ner Excel", output.getvalue(), f"rapport_{idag}.xlsx")

# --- CLOSE CONNECTION ---
conn.close()
