import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

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

# ----------- APP UI -----------
st.set_page_config(page_title="Säljlogg", layout="wide")
st.markdown("<h1 style='color:#083759;'>📈 Försäljningslogg & Affärer</h1>", unsafe_allow_html=True)

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
tab1, tab2, tab3 = st.tabs(["📊 Dagslogg & Analys", "📋 Affärer", "🏆 Vecko/Månadsanalys"])

# --- DAGSVY ---
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

        # Dagens målstatus
        goal_row = cursor.execute("SELECT * FROM mal WHERE datum = ?", (datetime.today().strftime('%Y-%m-%d'),)).fetchone()
        if goal_row:
            st.markdown(f"<b>TB-mål:</b> {goal_row[1]} kr &nbsp; <b>Samtalsmål:</b> {goal_row[2]} &nbsp; <b>Lönemål:</b> {goal_row[3]} kr", unsafe_allow_html=True)
            last_log = df_logg[df_logg['datum'] == datetime.today().strftime('%Y-%m-%d')]
            if not last_log.empty:
                tb_res, samtal_res, lon_res = last_log['tb'].values[0], last_log['samtal'].values[0], last_log['lon'].values[0]
                st.markdown(
                    f"<b>Dagens resultat:</b> TB: {'✅' if tb_res >= goal_row[1] else '❌'} {int(tb_res)} / {goal_row[1]}, "
                    f"Samtal: {'✅' if samtal_res >= goal_row[2] else '❌'} {int(samtal_res)} / {goal_row[2]}, "
                    f"Lön: {'✅' if lon_res >= goal_row[3] else '❌'} {int(lon_res)} / {goal_row[3]}",
                    unsafe_allow_html=True
                )
        # Export
        excel_buffer = io.BytesIO()
        df_logg.drop(columns=['id']).to_excel(excel_buffer, index=False, engine="openpyxl")
        st.download_button("Ladda ner logg som Excel", data=excel_buffer.getvalue(), file_name="daglogg.xlsx")

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
        # Export Excel & PDF
        excel_buffer_affar = io.BytesIO()
        df_affar.drop(columns=['id']).to_excel(excel_buffer_affar, index=False, engine="openpyxl")
        st.download_button("Ladda ner affärer som Excel", data=excel_buffer_affar.getvalue(), file_name="affarer.xlsx")
        if st.button("Exportera affärer till PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Affärsrapport", ln=True, align='C')
            pdf.ln(10)
            for _, row in df_affar.iterrows():
                rad = f"{row['datum']} | {row['affar_namn']} | TB: {row['tb']} kr | Tid: {row['minuter_till_stangning']} min"
                pdf.cell(0, 10, rad, ln=True)
            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            st.download_button(label="Ladda ner affärer som PDF", data=pdf_buffer, file_name="affarer.pdf", mime="application/pdf")
    else:
        st.info("Inga affärer i detta intervall.")

# --- VECKO/MÅNADSVY & GAMIFICATION ---
with tab3:
    st.subheader("Vecko-/Månadsanalys & Nivåer")
    if not df_logg.empty:
        df_logg['vecka'] = df_logg['datum'].dt.isocalendar().week
        df_logg['manad'] = df_logg['datum'].dt.to_period('M')
        # Veckosammanställning
        weekly = df_logg.groupby('vecka').agg(tb=('tb','sum'), samtal=('samtal','sum'), lon=('lon','sum')).reset_index()
        st.dataframe(weekly, use_container_width=True)
        st.line_chart(weekly.set_index('vecka')[['tb','lon']])
        # Månadssammanställning
        monthly = df_logg.groupby('manad').agg(tb=('tb','sum'), samtal=('samtal','sum'), lon=('lon','sum')).reset_index()
        st.dataframe(monthly, use_container_width=True)
        st.bar_chart(monthly.set_index('manad')[['tb','lon']])
        # Gamification / nivåer
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
        # Trendanalys
        if len(weekly) >= 2:
            diff = weekly['tb'].iloc[-1] - weekly['tb'].iloc[-2]
            trend = "📈 Uppåt!" if diff > 0 else "📉 Neråt!" if diff < 0 else "➖ Samma nivå"
            st.info(f"Senaste veckotrend: {trend} ({int(diff)} kr)")

# --- MAILUTSKICK AV RAPPORT ---
st.divider()
st.header("📧 Skicka rapport som e-post")
recipient = st.text_input("Mottagarens e-post")
smtp_user = st.text_input("Din Gmail-adress", type="default")
smtp_pass = st.text_input("App-lösenord (Gmail)", type="password")
if st.button("📤 Skicka rapport (Excel & PDF)"):
    if not smtp_user or not smtp_pass or not recipient:
        st.error("Fyll i alla fält.")
    else:
        try:
            msg = EmailMessage()
            msg['Subject'] = "Försäljningsrapport"
            msg['From'] = smtp_user
            msg['To'] = recipient
            msg.set_content("Här är försäljningsrapporten som PDF och Excel.")

            # Excel
            df_send = df_logg.drop(columns=['id']) if not df_logg.empty else pd.DataFrame()
            excel_buf = io.BytesIO()
            df_send.to_excel(excel_buf, index=False, engine="openpyxl")
            msg.add_attachment(excel_buf.getvalue(), maintype='application',
                              subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename="daglogg.xlsx")
            # PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Försäljningsrapport", ln=True, align='C')
            pdf.ln(10)
            if not df_send.empty:
                for _, row in df_send.iterrows():
                    rad = f"{row['datum'].strftime('%Y-%m-%d')} - TB: {row['tb']} kr, Samtal: {row['samtal']}, Lön: {row['lon']} kr"
                    pdf.cell(0, 10, rad, ln=True)
            else:
                pdf.cell(0, 10, "Ingen data.", ln=True)
            pdf_buf = io.BytesIO()
            pdf.output(pdf_buf)
            pdf_buf.seek(0)
            msg.add_attachment(pdf_buf.getvalue(), maintype='application', subtype='pdf', filename="forsaljningsrapport.pdf")

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)
            st.success("E-post skickad!")
        except Exception as e:
            st.error(f"Fel vid utskick: {e}")
