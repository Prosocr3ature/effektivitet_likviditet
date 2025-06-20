import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
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

cursor.execute('''
    CREATE TABLE IF NOT EXISTS mal (
        datum TEXT PRIMARY KEY,
        tb_mal INTEGER,
        samtal_mal INTEGER,
        lon_mal INTEGER
    )
''')
conn.commit()


st.title("📈 Försäljningslogg & Affärer")

# Visa dagens mål
st.subheader("📌 Dagens sparade mål")
today_str = datetime.today().strftime('%Y-%m-%d')
goal_row = cursor.execute("SELECT * FROM mal WHERE datum = ?", (today_str,)).fetchone()
if goal_row:
    st.markdown(f"- 🎯 TB-mål: **{goal_row[1]} kr**")
    st.markdown(f"- 📞 Samtalsmål: **{goal_row[2]}**")
    st.markdown(f"- 💰 Lönemål: **{goal_row[3]} kr**")
else:
    st.info("Inget sparat mål för idag ännu.")

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
        samtal, tid_min, kommentar = existing_log[2], existing_log[3], existing_log[9]
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
        samtal, tid_min = row[2], row[3]
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


# ---------------------- MÅL ----------------------
st.header("🏁 Sätt mål för idag")
tb_mal = st.number_input("🎯 TB-mål", min_value=0, step=100)
samtal_mal = st.number_input("📞 Samtalsmål", min_value=0, step=1)
lon_mal = st.number_input("💰 Lönemål", min_value=0, step=100)

if st.button("💾 Spara mål"):
    cursor.execute('''
        INSERT OR REPLACE INTO mal (datum, tb_mal, samtal_mal, lon_mal)
        VALUES (?, ?, ?, ?)
    ''', (datum.strftime("%Y-%m-%d"), tb_mal, samtal_mal, lon_mal))
    conn.commit()
    st.success("Mål sparade!")

# Jämför med mål
cursor.execute("SELECT * FROM mal WHERE datum = ?", (datum.strftime("%Y-%m-%d"),))
mål = cursor.fetchone()
if mål and existing_log:
    tb_result, samtal_result, lon_result = existing_log[4], existing_log[2], existing_log[8]
    st.subheader("📈 Målstatus")
    st.markdown(f"TB: {'✅' if tb_result >= mål[1] else '❌'} {tb_result} / {mål[1]}")
    st.markdown(f"Samtal: {'✅' if samtal_result >= mål[2] else '❌'} {samtal_result} / {mål[2]}")
    st.markdown(f"Lön: {'✅' if lon_result >= mål[3] else '❌'} {lon_result} / {mål[3]}")


# ---------------------- VECKOANALYS ----------------------
st.header("📅 Veckoanalys")
df['vecka'] = df['datum'].dt.isocalendar().week
weekly_summary = df.groupby('vecka').agg({
    'tb': 'sum',
    'samtal': 'sum',
    'lon': 'sum'
}).reset_index()
st.dataframe(weekly_summary)

st.line_chart(weekly_summary.set_index("vecka")[["tb", "lon"]])

# ---------------------- MÅNADSVY ----------------------
st.header("🗓️ Månadsvy")
df['manad'] = df['datum'].dt.to_period('M')
monthly_summary = df.groupby('manad').agg({
    'tb': 'sum',
    'samtal': 'sum',
    'lon': 'sum'
}).reset_index()
st.dataframe(monthly_summary)
st.bar_chart(monthly_summary.set_index("manad")[["tb", "lon"]])

# ---------------------- AUTOMATISKA MÅLFÖRSLAG ----------------------
st.header("🤖 Automatiskt målförslag")
if not df.empty:
    senaste_dag = df[df['datum'] == df['datum'].max()]
    if not senaste_dag.empty:
        snitt_tb = senaste_dag["tb"].values[0]
        snitt_samtal = senaste_dag["samtal"].values[0]
        snitt_lon = senaste_dag["lon"].values[0]
        st.markdown(f"🔁 Förslag inför nästa dag:")
        st.markdown(f"- TB-mål: **{int(snitt_tb * 1.1)} kr**")
        st.markdown(f"- Samtalsmål: **{int(snitt_samtal * 1.05)}**")
        st.markdown(f"- Lönemål: **{int(snitt_lon * 1.1)} kr**")


# ---------------------- GAMIFICATION ----------------------
st.header("🏆 Gamification – Utmana dig själv!")

if not df.empty:
    total_tb = df["tb"].sum()
    total_samtal = df["samtal"].sum()
    total_lon = df["lon"].sum()
    dagar = df["datum"].nunique()

    tb_per_dag = total_tb / dagar if dagar else 0
    samtal_per_dag = total_samtal / dagar if dagar else 0
    lon_per_dag = total_lon / dagar if dagar else 0

    # Nivåer (kan anpassas)
    if tb_per_dag >= 10000:
        nivå = "💎 Elite Closer"
    elif tb_per_dag >= 7000:
        nivå = "🔥 High Performer"
    elif tb_per_dag >= 4000:
        nivå = "🚀 On Fire"
    elif tb_per_dag >= 2000:
        nivå = "💼 Ambitious"
    else:
        nivå = "📈 Rookie in Training"

    st.markdown(f"**Nivå:** {nivå}")
    st.markdown(f"- 📊 Snitt TB/dag: **{int(tb_per_dag)} kr**")
    st.markdown(f"- 📞 Snitt samtal/dag: **{int(samtal_per_dag)}**")
    st.markdown(f"- 💰 Snitt lön/dag: **{int(lon_per_dag)} kr**")

    # Trendanalys: hur gick du jämfört med förra veckan?
    df['vecka'] = df['datum'].dt.isocalendar().week
    senaste_veckor = df['vecka'].unique()[-2:] if len(df['vecka'].unique()) >= 2 else None
    if senaste_veckor is not None and len(senaste_veckor) == 2:
        vecka1, vecka2 = senaste_veckor
        tb_v1 = df[df['vecka'] == vecka1]['tb'].sum()
        tb_v2 = df[df['vecka'] == vecka2]['tb'].sum()
        skillnad = tb_v2 - tb_v1
        trend = "📈 Uppåt!" if skillnad > 0 else "📉 Neråt!" if skillnad < 0 else "➖ Samma nivå"

        st.subheader("📊 Veckotrend")
        st.markdown(f"TB förra veckan: **{tb_v1} kr**")
        st.markdown(f"TB denna vecka: **{tb_v2} kr**")
        st.markdown(f"🔁 Förändring: **{int(skillnad)} kr** — {trend}")

        # Motivation
        if trend == "📈 Uppåt!":
            st.success("Du är på väg upp! Fortsätt så 💪")
        elif trend == "📉 Neråt!":
            st.warning("Lite tapp – men du vet vad du är kapabel till 🔥")
        else:
            st.info("Stabilt! Nu pushar vi nästa nivå 💥")
