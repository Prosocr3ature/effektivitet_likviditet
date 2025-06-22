import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE SETUP ---
st.set_page_config(page_title="📈 Säljdashboard", layout="wide")
st.title("📈 Försäljningslogg & Affärer")
st.markdown("💪 Fokusera på process, inte bara resultat.")

# --- DATABASE INITIALIZATION ---
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
c = conn.cursor()

# Logg-tabell
c.execute("""
CREATE TABLE IF NOT EXISTS logg (
  datum TEXT PRIMARY KEY,
  samtal INTEGER,
  tid_min INTEGER,
  tb REAL,
  energi INTEGER,
  humor INTEGER
)
""")

# Affärer-tabell med minuter_till_stangning
c.execute("""
CREATE TABLE IF NOT EXISTS affarer (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  datum TEXT,
  bolagstyp TEXT,
  foretagsnamn TEXT,
  abonnemang INTEGER,
  dealtyp TEXT,
  tb REAL,
  cashback REAL,
  margin REAL,
  minuter_till_stangning REAL
)
""")
conn.commit()

# --- SESSION STATE FOR EDITING ---
if "selected_affar" not in st.session_state:
    st.session_state.selected_affar = None

# --- INPUT: DAGSLOGG & AFFÄRER ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🗓️ Dagslogg")
    idag   = st.date_input("Datum", datetime.today())
    samtal = st.number_input("Antal samtal", min_value=0, step=1)
    tid_h  = st.number_input("Tid (timmar)", min_value=0, step=1)
    tid_m  = st.number_input("Tid (minuter)", min_value=0, step=1)
    tid_min= tid_h * 60 + tid_m
    tb_tot = st.number_input("Total TB (kr)", min_value=0.0, step=100.0)
    energi = st.slider("Energinivå (1–5)", 1, 5, 3)
    humor  = st.slider("Humör (1–5)",      1, 5, 3)

    if st.button("💾 Spara dagslogg"):
        c.execute("""
          INSERT OR REPLACE INTO logg
            (datum,samtal,tid_min,tb,energi,humor)
          VALUES (?,?,?,?,?,?)
        """, (
          idag.strftime("%Y-%m-%d"),
          samtal, tid_min, tb_tot,
          energi, humor
        ))
        conn.commit()
        st.success("Dagslogg sparad!")

with col2:
    st.subheader("📤 Lägg till / Redigera affär")
    skickad = st.time_input("Skickad tid")
    stangd  = st.time_input("Stängd tid")

    bolagstyp    = st.selectbox("Bolagstyp", ["Enskild firma","Aktiebolag"])
    foretagsnamn = st.text_input("Företagsnamn")
    abonnemang   = st.number_input("Abonnemang sålda", min_value=0, step=1)
    dealtyp      = st.selectbox("Nyteckning eller Förlängning", ["Nyteckning","Förlängning"])
    tb_affar     = st.number_input("TB för affären", min_value=0.0, step=100.0)
    cashback     = st.number_input("Cashback till kund", min_value=0.0, step=10.0)
    margin       = tb_affar - cashback
    minuter_diff = (datetime.combine(idag, stangd) - datetime.combine(idag, skickad)).seconds / 60

    aff_df = pd.read_sql_query(
        "SELECT * FROM affarer WHERE datum=? ORDER BY id",
        conn, params=(idag.strftime("%Y-%m-%d"),)
    )
    ids = aff_df["id"].tolist()
    st.session_state.selected_affar = st.selectbox(
        "Välj affär att redigera", [None] + ids
    )

    if st.session_state.selected_affar:
        row = aff_df[aff_df["id"] == st.session_state.selected_affar].iloc[0]
        bolagstyp    = st.selectbox(
            "Bolagstyp",
            ["Enskild firma","Aktiebolag"],
            index=["Enskild firma","Aktiebolag"].index(row["bolagstyp"])
        )
        foretagsnamn = st.text_input("Företagsnamn", value=row["foretagsnamn"])
        abonnemang   = st.number_input("Abonnemang sålda", value=int(row["abonnemang"]))
        dealtyp      = st.selectbox(
            "Nyteckning eller Förlängning",
            ["Nyteckning","Förlängning"],
            index=["Nyteckning","Förlängning"].index(row["dealtyp"])
        )
        tb_affar     = st.number_input("TB för affären", value=row["tb"])
        cashback     = st.number_input("Cashback till kund", value=row["cashback"])
        margin       = tb_affar - cashback

        if st.button("🔄 Uppdatera affär"):
            c.execute("""
              UPDATE affarer SET
                bolagstyp=?, foretagsnamn=?, abonnemang=?, dealtyp=?,
                tb=?, cashback=?, margin=?, minuter_till_stangning=?
              WHERE id=?
            """, (
              bolagstyp, foretagsnamn, abonnemang, dealtyp,
              tb_affar, cashback, margin, minuter_diff,
              st.session_state.selected_affar
            ))
            conn.commit()
            st.success("Affär uppdaterad!")

        if st.button("🗑️ Radera affär"):
            c.execute("DELETE FROM affarer WHERE id=?", (st.session_state.selected_affar,))
            conn.commit()
            st.success("Affär raderad!")
    else:
        if st.button("➕ Lägg till affär"):
            c.execute("""
              INSERT INTO affarer
                (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,
                 tb,cashback,margin,minuter_till_stangning)
              VALUES (?,?,?,?,?,?,?,?,?)
            """, (
              idag.strftime("%Y-%m-%d"),
              bolagstyp, foretagsnamn, abonnemang, dealtyp,
              tb_affar, cashback, margin, minuter_diff
            ))
            conn.commit()
            st.success("Affär tillagd!")

# --- DAGENS AFFÄRER I TABELL ---
st.markdown("---")
st.subheader("📋 Dagens affärer")
aff_df = pd.read_sql_query(
    "SELECT * FROM affarer WHERE datum=?",
    conn, params=(idag.strftime("%Y-%m-%d"),)
)
st.dataframe(aff_df.drop(columns=["datum"]), use_container_width=True)

# --- KPI & RECOMMENDATION ---
st.markdown("---")
st.subheader("🤖 Dagens KPI & rekommendationer")
log = pd.read_sql_query(
    "SELECT * FROM logg WHERE datum=?",
    conn, params=(idag.strftime("%Y-%m-%d"),)
)
if not log.empty:
    log = log.iloc[0]
    tot_tb     = log["tb"]
    tot_calls  = log["samtal"]
    tot_time   = log["tid_min"]
    tot_affs   = len(aff_df)
    avg_tb_aff = tot_affs and tot_tb/tot_affs or 0
    conv_rate  = tot_calls and tot_affs/tot_calls or 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total TB",      f"{tot_tb:.0f} kr")
    m2.metric("Antal affärer", f"{tot_affs}")
    m3.metric("TB per affär",  f"{avg_tb_aff:.0f} kr")
    m4.metric("Konv.grad",     f"{conv_rate:.1%}")

    df_model = pd.read_sql_query("SELECT samtal,tid_min,tb FROM logg", conn)
    if len(df_model) >= 5:
        X = df_model[["samtal","tid_min"]]; y = df_model["tb"]
        model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
        inc = st.slider("Test: öka samtal med (%)", -50, 100, 0)
        new_calls = tot_calls * (1 + inc/100)
        pred_tb   = model.predict([[new_calls, tot_time]])[0]
        st.write(f"➡️ Om du ökar samtalen med {inc}% → förväntad TB ≈ {pred_tb:.0f} kr")
else:
    st.info("Mata in dagens logg för KPI‐analys…")

# --- AFFÄRSSEGMENTERING (utan minsta‐gräns) ---
st.markdown("---")
st.subheader("🗺️ Segmentering av affärer")
full_aff = pd.read_sql_query("SELECT minuter_till_stangning,tb FROM affarer", conn)
if not full_aff.empty:
    n_clusters = min(2, len(full_aff))
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(full_aff)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xs)
    clusters = kmeans.predict(Xs)
    full_aff["cluster"] = clusters

    fig, ax = plt.subplots()
    ax.scatter(
        full_aff["minuter_till_stangning"],
        full_aff["tb"],
        c=full_aff["cluster"],
        cmap="tab10",
        s=50
    )
    ax.set_xlabel("Tid till stängning (min)")
    ax.set_ylabel("TB")
    ax.set_title("Kluster av affärer")
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Inga affärer att segmentera.")

# --- AUTOMATISKA SMART‐MÅL (ingen minimi-gräns) ---
st.markdown("---")
st.subheader("🎯 Automatiska målförslag")
df7 = pd.read_sql_query(
    "SELECT * FROM logg WHERE datum >= date('now','-7 days')",
    conn
)
if not df7.empty:
    avg_calls = df7["samtal"].mean()
    avg_tb    = df7["tb"].mean()
    st.write(f"- Samtalsmål imorgon: **{int(avg_calls * 1.05)}** (≈+5%)")
    st.write(f"- TB‐mål imorgon: **{int(avg_tb * 1.10)} kr** (≈+10%)")
    st.write("- Fokusera på nyteckningar för högre snitt‐TB.")
else:
    st.info("Mata in minst en dags logg för att få målförslag…")

# --- EXCEL EXPORT AV HELA LOGGEN ---
st.markdown("---")
st.subheader("📥 Ladda ner hela loggen som Excel")
buf = io.BytesIO()
pd.read_sql_query('SELECT * FROM logg', conn).to_excel(buf, index=False)
st.download_button(
    "Ladda ner som Excel",
    data=buf.getvalue(),
    file_name="forsaljningslogg.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
