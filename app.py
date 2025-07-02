import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import numpy as np
import openai

# Page settings
st.set_page_config(page_title="🎨 DaVinci’s Duk", layout="wide")
st.title("🎨 DaVinci’s Duk")
st.markdown("💪 Fokusera på process, inte bara resultat.")

openai.api_key = st.secrets.get("OPENAI_KEY", "")

# Database
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
c = conn.cursor()

# Create tables
tables = {
    "logg": """CREATE TABLE IF NOT EXISTS logg (
        datum TEXT PRIMARY KEY, start_tid TEXT, slut_tid TEXT, samtal INTEGER,
        tb REAL, energi INTEGER, humor INTEGER)""",
    "affarer": """CREATE TABLE IF NOT EXISTS affarer (
        id INTEGER PRIMARY KEY AUTOINCREMENT, datum TEXT, bolagstyp TEXT,
        foretagsnamn TEXT, abonnemang INTEGER, dealtyp TEXT, tb REAL, cashback REAL,
        margin REAL, hw_count INTEGER, hw_type TEXT, hw_model TEXT, hw_cost REAL, hw_tb REAL)""",
    "mal": """CREATE TABLE IF NOT EXISTS mal (
        datum TEXT PRIMARY KEY, daily_tb REAL, daily_calls INTEGER, monthly_tb REAL)""",
    "guldkunder": """CREATE TABLE IF NOT EXISTS guldkunder (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT,
        telefon TEXT, email TEXT, bindningstid TEXT, abonnemangsform TEXT, 
        pris REAL, operatoer TEXT, noteringar TEXT)""",
    "aterkomster": """CREATE TABLE IF NOT EXISTS aterkomster (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT,
        datum TEXT, detaljer TEXT, noteringar TEXT)""",
    "klara_kunder": """CREATE TABLE IF NOT EXISTS klara_kunder (
        id INTEGER PRIMARY KEY AUTOINCREMENT, orgnummer TEXT, kontaktperson TEXT,
        datum TEXT, status TEXT, noteringar TEXT)"""
}
for t in tables.values():
    c.execute(t)
conn.commit()

# DAGLIG LOGG
with st.expander("🗓️ Dagslogg & mål"):
    cols = st.columns(4)
    datum = cols[0].date_input("Datum", datetime.today())
    start_tid = cols[1].time_input("Starttid", datetime.strptime("09:00", "%H:%M"))
    slut_tid = cols[2].time_input("Sluttid", datetime.strptime("17:00", "%H:%M"))
    samtal = cols[3].number_input("Antal samtal", 0)

    cols2 = st.columns(4)
    tb = cols2[0].number_input("Dagens TB", 0.0, step=100.0)
    energi = cols2[1].slider("Energi", 1, 5, 3)
    humor = cols2[2].slider("Humör", 1, 5, 3)

    # Mål
    daily_tb = cols2[3].number_input("Dagens TB-mål", 0.0, step=100.0)
    daily_calls = st.number_input("Dagens samtalsmål", 0)
    monthly_tb = st.number_input("Månads-TB-mål", 0.0, step=100.0)

    if st.button("💾 Spara Logg & Mål"):
        c.execute("REPLACE INTO logg VALUES (?,?,?,?,?,?,?)",
            (datum, start_tid.strftime("%H:%M"), slut_tid.strftime("%H:%M"),
             samtal, tb, energi, humor))
        c.execute("REPLACE INTO mal VALUES (?,?,?,?)",
            (datum, daily_tb, daily_calls, monthly_tb))
        conn.commit()
        st.success("Logg och mål sparade!")

# AFFÄRER MED HÅRDVARA
with st.expander("📤 Lägg till affär"):
    cols = st.columns(3)
    aff_datum = cols[0].date_input("Affärsdatum", datetime.today(), key="aff_datum")
    bolagstyp = cols[1].selectbox("Bolagstyp", ["Enskild firma", "Aktiebolag"])
    foretagsnamn = cols[2].text_input("Företagsnamn")

    abonnemang = st.number_input("Abonnemang", 0)
    dealtyp = st.selectbox("Dealtyp", ["Nyteckning", "Förlängning"])
    tb_affar = st.number_input("TB Affär", 0.0, step=100.0)
    cashback = st.number_input("Cashback", 0.0, step=10.0)
    margin = tb_affar - cashback

    hw_count = st.number_input("Antal HW", 0)
    hw_type = st.text_input("HW Typ")
    hw_model = st.text_input("HW Modell")
    hw_cost = st.number_input("HW Inköpspris", 0.0)
    hw_tb = st.number_input("HW TB", 0.0)

    if st.button("➕ Spara affär"):
        c.execute("""INSERT INTO affarer 
        (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,tb,cashback,margin,hw_count,hw_type,hw_model,hw_cost,hw_tb)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (aff_datum, bolagstyp, foretagsnamn, abonnemang, dealtyp, tb_affar, cashback,
        margin, hw_count, hw_type, hw_model, hw_cost, hw_tb))
        conn.commit()
        st.success("Affär tillagd!")

# AI Analys
if st.button("🤖 AI-Analys Dagsresultat"):
    prompt = f"Dagens försäljning: Antal samtal: {samtal}, TB: {tb}. Ge tre konkreta säljtips på svenska."
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        ai_res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        tips = ai_res.choices[0].message.content
        st.info(tips)
    except Exception as e:
        st.error(f"AI-analys misslyckades: {e}") 

# KUNDHANTERING (Excel-liknande)
st.header("👥 Kundregister (Redigerbara Excel-liknande tabeller)")
kund_tabs = st.tabs(["Guldkunder", "Återkomster", "Klara kunder"])

def editable_df(table, cols):
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button(f"💾 Spara {table}"):
        c.execute(f"DELETE FROM {table}")
        for _, row in edited_df.iterrows():
            vals = tuple(row[col] for col in cols)
            c.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES({','.join('?'*len(cols))})", vals)
        conn.commit()
        st.success(f"{table} sparad!")

with kund_tabs[0]:
    editable_df("guldkunder", ["orgnummer", "kontaktperson", "telefon", "email", "bindningstid",
                               "abonnemangsform", "pris", "operatoer", "noteringar"])
with kund_tabs[1]:
    editable_df("aterkomster", ["orgnummer","kontaktperson","datum","detaljer","noteringar"])
with kund_tabs[2]:
    editable_df("klara_kunder", ["orgnummer","kontaktperson","datum","status","noteringar"])

# EXPORT EXCEL
if st.button("📥 Exportera all data till Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output) as w:
        for table in tables:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            df.to_excel(w, sheet_name=table, index=False)
    st.download_button("Ladda ner Excel", output.getvalue(), "rapport.xlsx")

conn.close()
