import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import numpy as np
import random

# --- CONFIGURE PAGE ---
st.set_page_config(page_title="📈 Försäljningslogg & Affärer", layout="wide")
st.markdown("# 📈 Försäljningslogg & Affärer")

# --- DATABASE SETUP ---
conn = sqlite3.connect('forsaljning.db', check_same_thread=False)
cursor = conn.cursor()
for table_sql in [
    '''CREATE TABLE IF NOT EXISTS logg (
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
    )''',
    '''CREATE TABLE IF NOT EXISTS affarer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datum TEXT,
        affar_namn TEXT,
        skickad_tid TEXT,
        stangd_tid TEXT,
        minuter_till_stangning REAL,
        tb REAL
    )''',
    '''CREATE TABLE IF NOT EXISTS mal (
        datum TEXT PRIMARY KEY,
        tb_mal INTEGER,
        samtal_mal INTEGER,
        lon_mal INTEGER
    )''']:
    cursor.execute(table_sql)
conn.commit()

# --- MOTIVATION ---
msgs = [
    "🔥 Disciplin slår motivation – varje dag!",
    "🚀 Varje samtal är en ny chans till rekord.",
    "💪 Fokusera på process, inte bara resultat.",
    "🏆 Du tävlar bara mot dig själv – slå gårdagens du.",
    "🌱 Små förbättringar bygger stora framgångar."
]
st.info(random.choice(msgs))

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns([1.5,1,1])

# Daily Log
with col1:
    st.subheader("🗓️ Dagslogg")
    datum = st.date_input('Datum', datetime.today())
    samtal = st.number_input('Antal samtal', 0)
    tid_h = st.number_input('Tid (timmar)', 0)
    tid_m = st.number_input('Tid (minuter)', 0)
    tid_min = tid_h*60 + tid_m
    tb = st.number_input('TB (kr)', 0.0, step=100.0)
    kommentar = st.text_input('Kommentar')
    if st.button('💾 Spara dagslogg'):
        tb_ps = tb/samtal if samtal else 0
        tb_pt = tb/(tid_min/60) if tid_min else 0
        snitt = tid_min/samtal if samtal else 0
        lon = tb*0.45
        cursor.execute('''
            INSERT OR REPLACE INTO logg
            (datum,samtal,tid_min,tb,tb_per_samtal,tb_per_timme,snitt_min_per_samtal,lon,kommentar)
            VALUES(?,?,?,?,?,?,?,?,?)
        ''', (datum.strftime('%Y-%m-%d'),samtal,tid_min,tb,tb_ps,tb_pt,snitt,lon,kommentar))
        conn.commit()
        st.success('Dagslogg sparad!')

# Goals
with col2:
    st.subheader('🎯 Sätt mål')
    g_tb = st.number_input('TB-mål',0,step=100)
    g_samtal = st.number_input('Samtalsmål',0)
    g_lon = st.number_input('Lönemål',0,step=100)
    if st.button('💾 Spara mål'):
        cursor.execute('''
            INSERT OR REPLACE INTO mal (datum,tb_mal,samtal_mal,lon_mal)
            VALUES(?,?,?,?)
        ''',(datum.strftime('%Y-%m-%d'),g_tb,g_samtal,g_lon))
        conn.commit()
        st.success('Mål sparade!')

# Business Entries
with col3:
    st.subheader('📤 Lägg till affär')
    name = st.text_input('Affärsnamn')
    sent = st.time_input('Skickad tid')
    closed = st.time_input('Stängd tid')
    tb_a = st.number_input('TB affär',0.0,step=100.0)
    if st.button('📌 Spara affär'):
        diff = (datetime.combine(datum,closed)-datetime.combine(datum,sent)).seconds/60
        cursor.execute('''
            INSERT INTO affarer (datum,affar_namn,skickad_tid,stangd_tid,minuter_till_stangning,tb)
            VALUES(?,?,?,?,?,?)
        ''',(datum.strftime('%Y-%m-%d'),name,sent.strftime('%H:%M'),closed.strftime('%H:%M'),diff,tb_a))
        conn.commit()
        st.success('Affär sparad!')

st.divider()
# Tabs
tab1,tab2,tab3,tab4 = st.tabs(['📊 Dag','📋 Affärer','🏆 Analys','🎯 Målhistorik'])

# Tab1: Day log
with tab1:
    df = pd.read_sql_query('SELECT * FROM logg ORDER BY datum DESC',conn)
    if not df.empty:
        df['datum']=pd.to_datetime(df['datum'])
        st.dataframe(df,use_container_width=True)
        fig,ax=plt.subplots()
        df.plot(x='datum',y=['tb','lon'],ax=ax,marker='o')
        ax.grid(True)
        st.pyplot(fig,use_container_width=True)

# Tab2: Business
with tab2:
    start=st.date_input('Från',datetime.today()-timedelta(days=30),key='s')
    end=st.date_input('Till',datetime.today(),key='e')
    df2=pd.read_sql_query(
        'SELECT * FROM affarer WHERE datum BETWEEN ? AND ? ORDER BY datum',
        conn,params=(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
    )
    st.dataframe(df2,use_container_width=True)

# Tab3: Analysis
with tab3:
    df=pd.read_sql_query('SELECT * FROM logg',conn)
    df['datum']=pd.to_datetime(df['datum'])
    df['vecka']=df['datum'].dt.isocalendar().week
    df['manad']=df['datum'].dt.to_period('M')
    weekly = df.groupby('vecka')[['tb','samtal','lon']].sum()
    monthly = df.groupby('manad')[['tb','samtal','lon']].sum()
    st.dataframe(weekly)
    st.line_chart(weekly[['tb','lon']])
    st.dataframe(monthly)
    st.bar_chart(monthly[['tb','lon']])

# Tab4: Goals history & simulation
with tab4:
    dfg=pd.read_sql_query('SELECT * FROM mal ORDER BY datum DESC',conn)
    dfl=pd.read_sql_query('SELECT * FROM logg ORDER BY datum DESC',conn)
    if not dfg.empty and not dfl.empty:
        dfg['datum']=pd.to_datetime(dfg['datum'])
        dfl['datum']=pd.to_datetime(dfl['datum'])
        st.dataframe(dfg)
        day=st.selectbox('Välj dag',dfg['datum'].dt.strftime('%Y-%m-%d'))
        gr=dfg[dfg['datum']==day].iloc[0]
        lr=dfl[dfl['datum']==day].iloc[0]
        pct_tb=lr['tb']/gr['tb_mal']*100 if gr['tb_mal'] else 0
        pct_s=lr['samtal']/gr['samtal_mal']*100 if gr['samtal_mal'] else 0
        pct_l=lr['lon']/gr['lon_mal']*100 if gr['lon_mal'] else 0
        st.write(f"TB: {lr['tb']} / {gr['tb_mal']} ({pct_tb:.0f}%)")
        st.write(f"Samtal: {lr['samtal']} / {gr['samtal_mal']} ({pct_s:.0f}%)")
        st.write(f"Lön: {lr['lon']} / {gr['lon_mal']} ({pct_l:.0f}%)")
        # Simulation
        ds=st.slider('+Samtal %',-20,100,0)
        dt=st.slider('+TB %',-20,100,0)
        ns=int(lr['samtal']*(1+ds/100))
        nts=lr['tb']/lr['samtal']*(1+dt/100) if lr['samtal'] else 0
        nt=ns*nts
        nl=nt*0.45
        st.write(f"Sim: {ns} samtal, {nts:.0f} snitt, TB={nt:.0f}, Lön={nl:.0f}")
        # Heatmap
        sr=np.arange(max(1,int(lr['samtal']*0.7)),int(lr['samtal']*1.5)+1)
        tr=np.linspace((lr['tb']/lr['samtal'] if lr['samtal'] else 0)*0.8,(lr['tb']/lr['samtal']if lr['samtal'] else 0)*1.3,10)
        mat=np.outer(sr,tr)
        fig,ax=plt.subplots()
        c=ax.imshow(mat,origin='lower',aspect='auto',extent=[tr[0],tr[-1],sr[0],sr[-1]])
        fig.colorbar(c,ax=ax,label='TB')
        st.pyplot(fig)

# Excel export of full log
buf=io.BytesIO()
pd.read_sql_query('SELECT * FROM logg',conn).to_excel(buf,index=False)
st.download_button('📥 Ladda ner Excel','buf.getvalue()',file_name='logg.xlsx')
