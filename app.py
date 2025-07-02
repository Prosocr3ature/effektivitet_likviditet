import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import numpy as np
import matplotlib.pyplot as plt
import openai

# — SETTINGS —
st.set_page_config(page_title="🎨 DaVinci’s Duk", layout="wide")
st.title("🎨 DaVinci’s Duk")
st.markdown("💪 Fokusera på process, inte bara resultat.")

openai.api_key = st.secrets.get("OPENAI_KEY", "")

# — DATABASE —
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
c = conn.cursor()

# Tables
c.execute("""CREATE TABLE IF NOT EXISTS logg (
    datum TEXT PRIMARY KEY, start_tid TEXT, slut_tid TEXT, samtal INTEGER,
    tb REAL, energi INTEGER, humor INTEGER)""")

c.execute("""CREATE TABLE IF NOT EXISTS affarer (
    id INTEGER PRIMARY KEY AUTOINCREMENT, datum TEXT, bolagstyp TEXT,
    foretagsnamn TEXT, abonnemang INTEGER, dealtyp TEXT, tb REAL, cashback REAL,
    margin REAL, hw_count INTEGER, hw_type TEXT, hw_model TEXT, hw_cost REAL, hw_tb REAL)""")

c.execute("""CREATE TABLE IF NOT EXISTS mal (
    datum TEXT PRIMARY KEY, daily_tb REAL, daily_calls INTEGER, monthly_tb REAL)""")

for kund_tabell in ["guldkunder", "aterkomster", "klara_kunder"]:
    c.execute(f"""CREATE TABLE IF NOT EXISTS {kund_tabell} (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT,
        datum TEXT, detaljer TEXT, noteringar TEXT)""")
conn.commit()

# — DAGLOGG —
col1, col2 = st.columns([2,3])
idag = datetime.today().date()

with col1:
    st.subheader("🗓️ Dagslogg")
    start_tid = st.time_input("Starttid (logg)", value=datetime.strptime("09:00", "%H:%M").time())
    slut_tid = st.time_input("Sluttid (logg)", value=datetime.strptime("17:00", "%H:%M").time())
    samtal = st.number_input("Antal samtal", min_value=0)
    tb = st.number_input("Dagens TB", min_value=0.0, step=100.0)
    energi = st.slider("Energi (1–5)", 1, 5, 3)
    humor = st.slider("Humör (1–5)", 1, 5, 3)

    if st.button("💾 Spara dagslogg"):
        c.execute("INSERT OR REPLACE INTO logg VALUES (?,?,?,?,?,?,?)",
            (idag, start_tid.strftime("%H:%M"), slut_tid.strftime("%H:%M"),
             samtal, tb, energi, humor))
        conn.commit()
        st.success("Dagslogg sparad!")

with col2:
    st.subheader("🎯 Mål")
    daily_tb = st.number_input("Dagens TB-mål", min_value=0.0, step=100.0)
    daily_calls = st.number_input("Dagens samtalsmål", min_value=0)
    monthly_tb = st.number_input("Månads-TB-mål", min_value=0.0, step=100.0)

    if st.button("💾 Spara mål"):
        c.execute("INSERT OR REPLACE INTO mal VALUES (?,?,?,?)",
            (idag, daily_tb, daily_calls, monthly_tb))
        conn.commit()
        st.success("Mål sparade!")

# Progress bars
logg = c.execute("SELECT samtal, tb FROM logg WHERE datum=?", (str(idag),)).fetchone()
mal = c.execute("SELECT daily_tb, daily_calls, monthly_tb FROM mal WHERE datum=?", (str(idag),)).fetchone()

if logg and mal:
    st.write("**Dagsmål TB**"); st.progress(min(1, logg[1]/(mal[0] or 1)))
    st.write("**Dagsmål samtal**"); st.progress(min(1, logg[0]/(mal[1] or 1)))

    month_tb = c.execute("SELECT SUM(tb) FROM logg WHERE substr(datum,1,7)=?",
                         (idag.strftime("%Y-%m"),)).fetchone()[0] or 0
    st.write("**Månads-TB**"); st.progress(min(1, month_tb/(mal[2] or 1)))

# — AFFÄRER —
st.subheader("📤 Affärer + Hårdvara")
with st.expander("Lägg till ny affär"):
    cols = st.columns(4)
    datum_affar = cols[0].date_input("Datum", idag)
    bolagstyp = cols[1].selectbox("Bolagstyp", ["Enskild firma","Aktiebolag"])
    foretagsnamn = cols[2].text_input("Företagsnamn")
    abonnemang = cols[3].number_input("Abonnemang",0)

    cols2 = st.columns(4)
    dealtyp = cols2[0].selectbox("Typ",["Nyteckning","Förlängning"])
    tb_affar = cols2[1].number_input("TB affär",0.0,step=100.0)
    cashback = cols2[2].number_input("Cashback",0.0)
    margin = tb_affar-cashback

    hw_count = cols2[3].number_input("Antal HW",0)
    hw_type = st.text_input("HW Typ")
    hw_model = st.text_input("HW Modell")
    hw_cost = st.number_input("HW Inköpspris")
    hw_tb = st.number_input("HW TB")

    if st.button("➕ Lägg till affär"):
        c.execute("""INSERT INTO affarer (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,
        tb,cashback,margin,hw_count,hw_type,hw_model,hw_cost,hw_tb) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datum_affar,bolagstyp,foretagsnamn,abonnemang,dealtyp,tb_affar,cashback,margin,
        hw_count,hw_type,hw_model,hw_cost,hw_tb))
        conn.commit()
        st.success("Affär sparad!")

# — GPT ANALYS —
st.subheader("🤖 AI Försäljningsanalys")
if st.button("Generera AI-analys"):
    prompt = f"Dagens TB={tb}, Samtal={samtal}. Ge 3 förbättringsförslag."
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
            {"role":"user","content":prompt}])
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error("GPT misslyckades: " + str(e))

# — EXPORT EXCEL —
st.subheader("📥 Exportera data")
if st.button("Exportera till Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for table in ["logg","affarer","mal","guldkunder","aterkomster","klara_kunder"]:
            df=pd.read_sql_query(f"SELECT * FROM {table}",conn)
            df.to_excel(writer, sheet_name=table, index=False)
    st.download_button("Ladda ner Excel", output.getvalue(), "rapport.xlsx")

# — KUNDHANTERING —
st.subheader("👥 Kunder")
kund_tabs = st.tabs(["Guldkunder","Återkomster","Klara kunder"])
for i,tabell in enumerate(["guldkunder","aterkomster","klara_kunder"]):
    with kund_tabs[i]:
        df=pd.read_sql_query(f"SELECT * FROM {tabell}",conn)
        st.dataframe(df)

conn.close()
