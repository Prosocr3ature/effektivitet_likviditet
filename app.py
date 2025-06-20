import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import numpy as np
import random

# ----------- DATABASE -----------
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

# ----------- MOTIVATIONS-TIPS/CITAT -----------
motivations = [
    "🔥 \"Disciplin slår motivation – varje dag!\"",
    "🚀 \"Varje samtal är en ny chans till rekord.\"",
    "💪 \"Fokusera på process, inte bara resultat.\"",
    "🏆 \"Du tävlar bara mot dig själv – slå gårdagens du.\"",
    "🦁 \"Gör det som känns svårt först, så vinner du senare.\"",
    "🌱 \"Varje liten förbättring gör stor skillnad över tid.\"",
    "🎯 \"Tydligt mål? Dubbel chans till träff!\"",
    "👊 \"Allt du gör idag – bygger morgondagens framgång.\""
]

st.set_page_config(page_title="Säljlogg", layout="wide")
st.markdown("<h1 style='color:#083759;'>📈 Försäljningslogg & Affärer</h1>", unsafe_allow_html=True)

# ----------- DAGLIG MOTIVATION ----------
st.info(random.choice(motivations))

# ----------- UI: LOGG, MÅL, AFFÄR -----------
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    st.subheader("🗓️ Dagslogg")
    datum = st.date_input("Datum", datetime.today())
    samtal = st.number_input("Antal samtal", min_value=0, step=1)
    tid_h = st.number_input("Tid (timmar)", min_value=0, step=1)
    tid_m = st.number_input("Tid (minuter)", min_value=0, step=1)
    tid_min = tid_h * 60 + tid_m
    tb = st.number_input("TB (kr)", min_value=0.0, step=100.0)
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
        # Automatisk feedback direkt
        goal_row = cursor.execute("SELECT * FROM mal WHERE datum = ?", (datum.strftime('%Y-%m-%d'),)).fetchone()
        if goal_row:
            tb_pct = int(100*tb/(goal_row[1] or 1))
            samtal_pct = int(100*samtal/(goal_row[2] or 1))
            lon_pct = int(100*lon/(goal_row[3] or 1))
            st.markdown("---")
            if tb_pct>=100 and samtal_pct>=100 and lon_pct>=100:
                st.success("🔥 Fullträff! Alla mål uppfyllda idag!")
            else:
                feedback = []
                if tb_pct<100: feedback.append("TB under mål")
                if samtal_pct<100: feedback.append("Samtal under mål")
                if lon_pct<100: feedback.append("Lön under mål")
                st.warning(" ".join(feedback) + " – du är nära, nästa gång tar du det!")

with col2:
    st.subheader("🎯 Sätt mål")
    tb_mal = st.number_input("TB-mål", min_value=0, step=100)
    samtal_mal = st.number_input("Samtalsmål", min_value=0, step=1)
    lon_mal = st.number_input("Lönemål", min_value=0, step=100)
    if st.button("💾 Spara mål"):
        cursor.execute('''
            INSERT OR REPLACE INTO mal (datum, tb_mal, samtal_mal, lon_mal)
            VALUES (?, ?, ?, ?)
        ''', (datum.strftime("%Y-%m-%d"), tb_mal, samtal_mal, lon_mal))
        conn.commit()
        st.success("Mål sparade!")
        # Direkt feedback om mål för idag
        last_log = cursor.execute("SELECT * FROM logg WHERE datum = ?", (datum.strftime('%Y-%m-%d'),)).fetchone()
        if last_log:
            tb_pct = int(100*(last_log[4]/(tb_mal or 1)))
            samtal_pct = int(100*(last_log[2]/(samtal_mal or 1)))
            lon_pct = int(100*(last_log[8]/(lon_mal or 1)))
            st.markdown("---")
            if tb_pct>=100 and samtal_pct>=100 and lon_pct>=100:
                st.success("🔥 Fullträff! Alla mål för idag redan uppfyllda!")
            else:
                feedback = []
                if tb_pct<100: feedback.append("TB under mål")
                if samtal_pct<100: feedback.append("Samtal under mål")
                if lon_pct<100: feedback.append("Lön under mål")
                st.warning(" ".join(feedback))

with col3:
    st.subheader("📤 Lägg till affär")
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

st.divider()
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dagslogg & Analys",
    "📋 Affärer",
    "🏆 Vecko/Månadsanalys",
    "🎯 Målhistorik, Analys & Rapport"
])

# --- DAGSVY & TOPPLISTA ---
with tab1:
    df_logg = pd.read_sql_query("SELECT * FROM logg ORDER BY datum", conn)
    if not df_logg.empty:
        df_logg['datum'] = pd.to_datetime(df_logg['datum'])
        st.subheader("Logg (alla dagar)")
        st.dataframe(df_logg.drop(columns=['id']), use_container_width=True)
        # Grafer: TB och lön över tid
        fig, ax = plt.subplots()
        df_logg.plot(x="datum", y=["tb", "lon"], ax=ax, marker='o')
        ax.set_ylabel("Belopp (kr)")
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        # Rekord/topplista
        st.markdown("#### 🏅 Ditt rekord:")
        top = df_logg.loc[df_logg['tb'].idxmax()]
        st.write(f"TB-rekord: {int(top['tb'])} kr ({top['datum'].date()})")
        st.write(f"Fler samtal än någonsin: {int(df_logg['samtal'].max())}")
        st.write(f"Högsta lönedag: {int(df_logg['lon'].max())} kr")
        # Ladda ner all historik
        excel_buffer = io.BytesIO()
        df_logg.drop(columns=['id']).to_excel(excel_buffer, index=False, engine="openpyxl")
        st.download_button("Ladda ner all logg som Excel", data=excel_buffer.getvalue(), file_name="daglogg_historik.xlsx")

# --- AFFÄRER ---
with tab2:
    date_start = st.date_input("Affärer från", datetime.today() - timedelta(days=30), key="affar_start")
    date_end = st.date_input("Affärer till", datetime.today(), key="affar_end")
    df_affar = pd.read_sql_query(
        "SELECT * FROM affarer WHERE datum BETWEEN ? AND ? ORDER BY datum",
        conn, params=(date_start.strftime("%Y-%m-%d"), date_end.strftime("%Y-%m-%d"))
    )
    if not df_affar.empty:
        st.dataframe(df_affar.drop(columns=['id']), use_container_width=True)
        excel_buffer_affar = io.BytesIO()
        df_affar.drop(columns=['id']).to_excel(excel_buffer_affar, index=False, engine="openpyxl")
        st.download_button("Ladda ner affärer som Excel", data=excel_buffer_affar.getvalue(), file_name="affarer.xlsx")
    else:
        st.info("Inga affärer i detta intervall.")

# --- VECKO/MÅNADSVY, GAMIFICATION & PROGNOS ---
with tab3:
    st.subheader("Vecko-/Månadsanalys & Nivåer")
    if not df_logg.empty:
        df_logg['vecka'] = df_logg['datum'].dt.isocalendar().week
        df_logg['manad'] = df_logg['datum'].dt.to_period('M')
        weekly = df_logg.groupby('vecka').agg(tb=('tb','sum'), samtal=('samtal','sum'), lon=('lon','sum')).reset_index()
        st.dataframe(weekly, use_container_width=True)
        st.line_chart(weekly.set_index('vecka')[['tb','lon']])
        monthly = df_logg.groupby('manad').agg(tb=('tb','sum'), samtal=('samtal','sum'), lon=('lon','sum')).reset_index()
        st.dataframe(monthly, use_container_width=True)
        st.bar_chart(monthly.set_index('manad')[['tb','lon']])
        dagar = df_logg['datum'].nunique()
        tot_tb = df_logg['tb'].sum()
        tb_per_dag = tot_tb / dagar if dagar else 0
        if tb_per_dag >= 10000:
            niva = "💎 Elite Closer"
        elif tb_per_dag >= 7000:
            niva = "🔥 High Performer"
        elif tb_per_dag >= 4000:
            niva = "🚀 On Fire"
        elif tb_per_dag >= 2000:
            niva = "💼 Ambitious"
        else:
            niva = "📈 Rookie in Training"
        st.markdown(f"**Nivå:** <span style='color:#1976d2;font-weight:bold;'>{niva}</span> &nbsp; (Snitt TB/dag: <b>{int(tb_per_dag)}</b> kr)", unsafe_allow_html=True)
        if len(weekly) >= 2:
            diff = weekly['tb'].iloc[-1] - weekly['tb'].iloc[-2]
            trend = "📈 Uppåt!" if diff > 0 else "📉 Neråt!" if diff < 0 else "➖ Samma nivå"
            st.info(f"Senaste veckotrend: {trend} ({int(diff)} kr)")
        # Prognos till månadsslut
        this_month = df_logg[df_logg['datum'].dt.to_period('M') == pd.to_datetime(datetime.today()).to_period('M')]
        dagar_gjorda = this_month['datum'].nunique()
        dagar_tot = pd.Period(datetime.today(), 'M').days_in_month
        if dagar_gjorda:
            tb_per_dag = this_month['tb'].sum() / dagar_gjorda
            prog_tb = int(tb_per_dag * dagar_tot)
            st.success(f"🔮 Prognos: Om du håller snittet når du **{prog_tb} kr** TB denna månad!")

# --- MÅLHISTORIK, ANALYS, SIMULERING & RAPPORT/EXPORT ---
with tab4:
    st.header("🎯 Målhistorik, Analys & Vad-händer-om")

    # Ladda mål och loggar
    df_mal = pd.read_sql_query("SELECT * FROM mal ORDER BY datum DESC", conn)
    df_logg = pd.read_sql_query("SELECT * FROM logg ORDER BY datum DESC", conn)
    if not df_mal.empty and not df_logg.empty:
        df_mal['datum'] = pd.to_datetime(df_mal['datum'])
        df_logg['datum'] = pd.to_datetime(df_logg['datum'])

        st.dataframe(df_mal, use_container_width=True)

        dagar = list(df_mal['datum'].dt.strftime("%Y-%m-%d"))
        valdag = st.selectbox("Välj dag för analys/simulering", dagar)
        mal_row = df_mal[df_mal['datum'] == pd.to_datetime(valdag)].iloc[0]
        logg_row = df_logg[df_logg['datum'] == pd.to_datetime(valdag)].squeeze()

        st.subheader(f"Detaljerad analys {valdag}")

        tb_pct = round(100 * logg_row['tb'] / mal_row['tb_mal']) if mal_row['tb_mal'] else 0
        samtal_pct = round(100 * logg_row['samtal'] / mal_row['samtal_mal']) if mal_row['samtal_mal'] else 0
        lon_pct = round(100 * logg_row['lon'] / mal_row['lon_mal']) if mal_row['lon_mal'] else 0

        def emoji(pct):
            if pct >= 120: return "💚"
            elif pct >= 100: return "🟩"
            elif pct >= 80: return "🟨"
            else: return "🟥"

        st.write(f"TB: {logg_row['tb']} / {mal_row['tb_mal']} kr  ({tb_pct}%) {emoji(tb_pct)}")
        st.write(f"Samtal: {logg_row['samtal']} / {mal_row['samtal_mal']}  ({samtal_pct}%) {emoji(samtal_pct)}")
        st.write(f"Lön: {int(logg_row['lon'])} / {mal_row['lon_mal']} kr  ({lon_pct}%) {emoji(lon_pct)}")
        st.write(f"Kommentar: {logg_row['kommentar']}")

        st.markdown("---")
        feedback = []
        if tb_pct >= 100 and samtal_pct >= 100 and lon_pct >= 100:
            feedback.append("🌟 **Fullträff! Alla mål uppnådda – grym prestation!**")
        else:
            if tb_pct < 100:
                feedback.append(f"TB under mål – fundera på hur du kan få upp snittaffären eller antalet större affärer.")
            if samtal_pct < 100:
                feedback.append(f"Samtal under mål – går det att öka samtalstakten eller kvaliteten?")
            if lon_pct < 100:
                feedback.append(f"Lön under mål – hänger ofta ihop med TB, men håll koll på konvertering/fördelning mellan affärer.")
        st.write('\n'.join(feedback))

        st.markdown("---")
        senaste7 = df_mal.head(7)
        if not senaste7.empty:
            st.subheader("📊 7-dagars trend")
            tb_succ = (df_logg[df_logg['datum'].isin(senaste7['datum'])]['tb'].sum() /
                       (senaste7['tb_mal'].sum() or 1)) * 100
            samtal_succ = (df_logg[df_logg['datum'].isin(senaste7['datum'])]['samtal'].sum() /
                           (senaste7['samtal_mal'].sum() or 1)) * 100
            lon_succ = (df_logg[df_logg['datum'].isin(senaste7['datum'])]['lon'].sum() /
                        (senaste7['lon_mal'].sum() or 1)) * 100
            st.write(f"TB snittmåluppfyl
