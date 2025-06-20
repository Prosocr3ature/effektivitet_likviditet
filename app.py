
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
        datum TEXT,
        samtal INTEGER,
        tid_min INTEGER,
        tb REAL,
        tb_per_samtal REAL,
        tb_per_timme REAL,
        lon REAL,
        kommentar TEXT
    )
''')
conn.commit()

st.title("📈 Daglig Försäljningslogg (med redigering & export)")

# Inmatning
st.header("Fyll i dagens siffror")
datum = st.date_input("Datum", datetime.today())
samtal = st.number_input("Antal samtal", min_value=0, step=1)
tid_min = st.number_input("Tid (minuter)", min_value=0, step=1)
tb = st.number_input("TB (kr)", min_value=0.0, step=100.0)
kommentar = st.text_input("Kommentar/reflektion")

# Beräkningar
tb_per_samtal = tb / samtal if samtal > 0 else 0
tb_per_timme = tb / (tid_min / 60) if tid_min > 0 else 0
lon = tb * 0.45

# Spara till databas
if st.button("Spara till logg"):
    cursor.execute('''
        INSERT INTO logg (datum, samtal, tid_min, tb, tb_per_samtal, tb_per_timme, lon, kommentar)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datum.strftime("%Y-%m-%d"), samtal, tid_min, tb, tb_per_samtal, tb_per_timme, lon, kommentar))
    conn.commit()
    st.success("Dagens siffror sparade permanent!")

# Visa historik
df = pd.read_sql_query("SELECT * FROM logg ORDER BY datum", conn)
if not df.empty:
    st.subheader("✨ Din historik")
    st.dataframe(df.drop(columns=['id']), use_container_width=True)

    # Export till Excel
    st.download_button(
        label="📥 Ladda ner som Excel",
        data=io.BytesIO(df.to_excel(index=False, engine='openpyxl')),
        file_name="forsaljning_logg.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Redigera/radera
    st.subheader("✏️ Redigera eller 🗑️ Radera")
    selected_id = st.selectbox("Välj post att redigera/radera", df["id"])

    selected_row = df[df["id"] == selected_id].iloc[0]
    new_tb = st.number_input("Uppdatera TB", value=float(selected_row["tb"]), key="edit_tb")
    new_kommentar = st.text_input("Uppdatera kommentar", value=selected_row["kommentar"], key="edit_comment")

    if st.button("Uppdatera post"):
        new_tb_per_samtal = new_tb / selected_row["samtal"] if selected_row["samtal"] > 0 else 0
        new_tb_per_timme = new_tb / (selected_row["tid_min"] / 60) if selected_row["tid_min"] > 0 else 0
        new_lon = new_tb * 0.45
        cursor.execute("""
            UPDATE logg
            SET tb = ?, tb_per_samtal = ?, tb_per_timme = ?, lon = ?, kommentar = ?
            WHERE id = ?
        """, (new_tb, new_tb_per_samtal, new_tb_per_timme, new_lon, new_kommentar, selected_id))
        conn.commit()
        st.success("Post uppdaterad!")

    if st.button("Radera post"):
        cursor.execute("DELETE FROM logg WHERE id = ?", (selected_id,))
        conn.commit()
        st.warning("Post raderad!")

    # Diagram
    st.subheader("📊 Diagram")
    df['datum'] = pd.to_datetime(df['datum'])
    df.set_index("datum", inplace=True)
    st.line_chart(df["tb"])
    st.line_chart(df["lon"])
