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
st.set_page_config(page_title="🎨 DaVinci’s Duk", layout="wide")
st.title("🎨 DaVinci’s Duk")
st.markdown("💪 Fokusera på process, inte bara resultat.")

# --- DATABASE INITIALIZATION ---
conn = sqlite3.connect("forsaljning.db", check_same_thread=False)
c = conn.cursor()

# Logg-table
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
# Affärer-table (inklusive hårdvara)
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
  minuter_till_stangning REAL,
  hw_count INTEGER,
  hw_type TEXT,
  hw_model TEXT,
  hw_cost REAL,
  hw_tb REAL
)
""")
# Guldkunder
c.execute("""
CREATE TABLE IF NOT EXISTS guldkunder (
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
)
""")
# Återkomster
c.execute("""
CREATE TABLE IF NOT EXISTS aterkomster (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  orgnummer TEXT,
  kontaktperson TEXT,
  aterkomstdatum TEXT,
  tema TEXT,
  noteringar TEXT
)
""")
# Klara kunder
c.execute("""
CREATE TABLE IF NOT EXISTS klara_kunder (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  orgnummer TEXT,
  kontaktperson TEXT,
  avslutsdatum TEXT,
  status TEXT,
  noteringar TEXT
)
""")
conn.commit()

# --- INPUT SECTIONS ---
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
    humor  = st.slider("Humör (1–5)", 1, 5, 3)
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
    bolagstyp    = st.selectbox("Bolagstyp", ["Enskild firma", "Aktiebolag"])
    foretagsnamn = st.text_input("Företagsnamn")
    abonnemang   = st.number_input("Abonnemang sålda", min_value=0, step=1)
    dealtyp      = st.selectbox("Nyteckning eller Förlängning", ["Nyteckning", "Förlängning"])
    skickad      = st.time_input("Skickad tid")
    stangd       = st.time_input("Stängd tid")
    tb_affar     = st.number_input("TB för affären", min_value=0.0, step=100.0)
    cashback     = st.number_input("Cashback till kund", min_value=0.0, step=10.0)
    margin       = tb_affar - cashback
    minuter_diff = (datetime.combine(idag, stangd) - datetime.combine(idag, skickad)).seconds / 60
    st.markdown("**Hårdvara**")
    hw_count = st.number_input("Antal hårdvaror sålda", min_value=0, step=1)
    hw_type  = st.text_input("Typ av hårdvara")
    hw_model = st.text_input("Modell")
    hw_cost  = st.number_input("Inköpspris/avgift", min_value=0.0, step=10.0)
    hw_tb    = st.number_input("TB för hårdvara", min_value=0.0, step=10.0)

    df_aff = pd.read_sql_query(
        "SELECT * FROM affarer WHERE datum=? ORDER BY id", conn,
        params=(idag.strftime("%Y-%m-%d"),)
    )
    sel = st.selectbox("Välj affär att redigera", [None] + df_aff["id"].tolist())
    if sel:
        row = df_aff[df_aff["id"] == sel].iloc[0]
        # Förifyll
        bolagstyp    = row["bolagstyp"]
        foretagsnamn = row["foretagsnamn"]
        abonnemang   = int(row["abonnemang"])
        dealtyp      = row["dealtyp"]
        tb_affar     = row["tb"]
        cashback     = row["cashback"]
        hw_count     = int(row["hw_count"])
        hw_type      = row["hw_type"]
        hw_model     = row["hw_model"]
        hw_cost      = row["hw_cost"]
        hw_tb        = row["hw_tb"]
        if st.button("🔄 Uppdatera affär"):
            c.execute("""
              UPDATE affarer SET
                bolagstyp=?, foretagsnamn=?, abonnemang=?, dealtyp=?,
                tb=?, cashback=?, margin=?, minuter_till_stangning=?,
                hw_count=?, hw_type=?, hw_model=?, hw_cost=?, hw_tb=?
              WHERE id=?
            """, (
              bolagstyp, foretagsnamn, abonnemang, dealtyp,
              tb_affar, cashback, margin, minuter_diff,
              hw_count, hw_type, hw_model, hw_cost, hw_tb,
              sel
            ))
            conn.commit()
            st.success("Affär uppdaterad!")
        if st.button("🗑️ Radera affär"):
            c.execute("DELETE FROM affarer WHERE id=?", (sel,))
            conn.commit()
            st.success("Affär raderad!")
    else:
        if st.button("➕ Lägg till affär"):
            c.execute("""
              INSERT INTO affarer
                (datum,bolagstyp,foretagsnamn,abonnemang,dealtyp,
                 tb,cashback,margin,minuter_till_stangning,
                 hw_count,hw_type,hw_model,hw_cost,hw_tb)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
              idag.strftime("%Y-%m-%d"),
              bolagstyp, foretagsnamn, abonnemang, dealtyp,
              tb_affar, cashback, margin, minuter_diff,
              hw_count, hw_type, hw_model, hw_cost, hw_tb
            ))
            conn.commit()
            st.success("Affär tillagd!")

st.markdown("---")

# --- Dagens affärer ---
st.subheader("📋 Dagens affärer")
df_today = pd.read_sql_query(
    "SELECT * FROM affarer WHERE datum=?", conn,
    params=(idig.strftime("%Y-%m-%d"),)
)
st.dataframe(df_today.drop(columns=["datum"]), use_container_width=True)

# --- KPI & REKOMMENDATION ---
st.markdown("---")
st.subheader("🤖 Dagens KPI & rekommendationer")
log = pd.read_sql_query(
    "SELECT * FROM logg WHERE datum=?", conn,
    params=(idag.strftime("%Y-%m-%d"),)
)
if not log.empty:
    row = log.iloc[0]
    tot_tb, tot_calls, tot_time = row["tb"], row["samtal"], row["tid_min"]
    tot_affs = len(df_today)
    avg_tb   = tot_affs and tot_tb/tot_affs or 0
    conv     = tot_calls and tot_affs/tot_calls or 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total TB",      f"{tot_tb:.0f} kr")
    m2.metric("Antal affärer", str(tot_affs))
    m3.metric("TB per affär",  f"{avg_tb:.0f} kr")
    m4.metric("Konv.grad",     f"{conv:.1%}")
    df_m = pd.read_sql_query("SELECT samtal,tid_min,tb FROM logg", conn)
    if len(df_m) >= 2:
        X, y = df_m[["samtal","tid_min"]], df_m["tb"]
        mdl = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
        inc = st.slider("Öka samtal med (%)", -50, 100, 0)
        pred_tb = mdl.predict([[tot_calls*(1+inc/100), tot_time]])[0]
        st.write(f"➡️ Öka samtalen med {inc}% → förväntad TB ≈ {pred_tb:.0f} kr")
else:
    st.info("Mata in dagens logg för KPI‐analys…")

# --- AFFÄRS-SEGMENTERING ---
st.markdown("---")
st.subheader("🗺️ Segmentering av affärer")
full = pd.read_sql_query("SELECT minuter_till_stangning,tb FROM affarer", conn)
if not full.empty:
    clusters = min(2, len(full))
    Xs = StandardScaler().fit_transform(full)
    lbl=KMeans(n_clusters=clusters, random_state=0).fit_predict(Xs)
    full["cluster"] = lbl
    fig, ax = plt.subplots()
    ax.scatter(full["minuter_till_stangning"], full["tb"], c=lbl, cmap="tab10")
    ax.set_xlabel("Tid till stängning (min)"); ax.set_ylabel("TB")
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Inga affärer att segmentera.")

# --- AUTOMATISKA MÅLFÖRSLAG ---
st.markdown("---")
st.subheader("🎯 Automatiska målförslag")
last7 = pd.read_sql_query(
    "SELECT * FROM logg WHERE datum>=date('now','-7 days')", conn
)
if not last7.empty:
    ac, at = last7["samtal"].mean(), last7["tb"].mean()
    st.write(f"- Samtalsmål imorgon: **{int(ac*1.05)}**")
    st.write(f"- TB‐mål imorgon: **{int(at*1.10)} kr**")
else:
    st.info("Mata in minst en dags logg…")

# --- REDIGERBARA FLIKAR ---
st.markdown("---")
tabs = st.tabs(["⭐ Guldkunder","⏰ Återkomster","✅ Klara kunder"])

def highlight_yes_no(val):
    if str(val).lower()=="ja": return "background-color: lightgreen"
    if str(val).lower()=="nej": return "background-color: salmon"
    return ""

def highlight_date(val):
    try:
        d=pd.to_datetime(val)
        if d <= pd.Timestamp.today()+pd.Timedelta(days=3):
            return "background-color: lightyellow"
    except: pass
    return ""

def highlight_status(val):
    v=str(val).lower()
    if "klar" in v or "avslutad" in v:
        return "background-color: lightgreen"
    return ""

with tabs[0]:
    st.subheader("⭐ Guldkunder")
    dfg = pd.read_sql("SELECT * FROM guldkunder", conn)
    search = st.text_input("Sök (org.nr eller kontaktperson)", key="g1")
    if search:
        dfg = dfg[dfg["orgnummer"].str.contains(search, case=False, na=False) |
                  dfg["kontaktperson"].str.contains(search, case=False, na=False)]
    ed1 = st.data_editor(dfg, num_rows="dynamic", use_container_width=True)
    st.dataframe(ed1.style.applymap(highlight_yes_no, subset=["har_kund_svarat"]), use_container_width=True)
    if st.button("💾 Spara Guldkunder"):
        c.execute("DELETE FROM guldkunder")
        for _, r in ed1.iterrows():
            c.execute("""
              INSERT INTO guldkunder
                (orgnummer,kontaktperson,bindningstid,abonnemangsform,
                 pris,operatoerforsok,har_kund_svarat,ovriga_abb_ja_nej,noteringar)
              VALUES (?,?,?,?,?,?,?,?,?)
            """, tuple(r[col] for col in ed1.columns))
        conn.commit(); st.success("Guldkunder sparade!")

with tabs[1]:
    st.subheader("⏰ Återkomster")
    dfa = pd.read_sql("SELECT * FROM aterkomster", conn)
    search = st.text_input("Sök (org.nr eller kontaktperson)", key="a2")
    date   = st.date_input("Datum", key="d2")
    if search:
        dfa = dfa[dfa["orgnummer"].str.contains(search, case=False, na=False) |
                  dfa["kontaktperson"].str.contains(search, case=False, na=False)]
    if date:
        dfa = dfa[dfa["aterkomstdatum"] == date.strftime("%Y-%m-%d")]
    ed2 = st.data_editor(dfa, num_rows="dynamic", use_container_width=True)
    st.dataframe(ed2.style.applymap(highlight_date, subset=["aterkomstdatum"]), use_container_width=True)
    if st.button("💾 Spara Återkomster"):
        c.execute("DELETE FROM aterkomster")
        for _, r in ed2.iterrows():
            c.execute("""
              INSERT INTO aterkomster
                (orgnummer,kontaktperson,aterkomstdatum,tema,noteringar)
              VALUES (?,?,?,?,?)
            """, tuple(r[col] for col in ed2.columns))
        conn.commit(); st.success("Återkomster sparade!")

with tabs[2]:
    st.subheader("✅ Klara kunder")
    dfk = pd.read_sql("SELECT * FROM klara_kunder", conn)
    search = st.text_input("Sök (org.nr eller kontaktperson)", key="k3")
    date   = st.date_input("Avslutsdatum", key="d3")
    if search:
        dfk = dfk[dfk["orgnummer"].str.contains(search, case=False, na=False) |
                  dfk["kontaktperson"].str.contains(search, case=False, na=False)]
    if date:
        dfk = dfk[dfk["avslutsdatum"] == date.strftime("%Y-%m-%d")]
    ed3 = st.data_editor(dfk, num_rows="dynamic", use_container_width=True)
    st.dataframe(ed3.style.applymap(highlight_status, subset=["status"]), use_container_width=True)
    if st.button("💾 Spara Klara kunder"):
        c.execute("DELETE FROM klara_kunder")
        for _, r in ed3.iterrows():
            c.execute("""
              INSERT INTO klara_kunder
                (orgnummer,kontaktperson,avslutsdatum,status,noteringar)
              VALUES (?,?,?,?,?)
            """, tuple(r[col] for col in ed3.columns))
        conn.commit(); st.success("Klara kunder sparade!")

# --- EXCEL EXPORT AV HELA LOGGEN ---
st.markdown("---")
st.subheader("📥 Ladda ner loggen som Excel")
buf = io.BytesIO()
pd.read_sql_query("SELECT * FROM logg", conn).to_excel(buf, index=False)
st.download_button(
    "Ladda ner logg.xlsx",
    data=buf.getvalue(),
    file_name="logg.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
