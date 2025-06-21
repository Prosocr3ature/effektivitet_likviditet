import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="📈 Försäljningslogg & Affärer", layout="wide")

# --- DATABASE SETUP ---
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
        kommentar TEXT,
        energi INTEGER,
        humor INTEGER
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
        lon_mal INTEGER,
        specifikt TEXT,
        realistiskt TEXT,
        tidsbundet TEXT
    )
''')
conn.commit()

# --- OPENAI CLIENT SETUP ---
openai_client = OpenAI(api_key=st.secrets.get("OPENAI_KEY", ""))

# --- MOTIVATION MESSAGE ---
msgs = [
    "🔥 Disciplin slår motivation – varje dag!",
    "🚀 Varje samtal är en ny chans till rekord.",
    "💪 Fokusera på process, inte bara resultat.",
    "🏆 Du tävlar mot dig själv – slå gårdagens du.",
    "🌱 Små förbättringar bygger stora framgångar."
]
st.info(random.choice(msgs))

# --- SIDEBAR KPI METRICS ---
df_all = pd.read_sql_query("SELECT * FROM logg", conn)
if not df_all.empty:
    total_calls = df_all['samtal'].sum()
    total_tb = df_all['tb'].sum()
    total_time = df_all['tid_min'].sum()
    total_aff = len(pd.read_sql_query("SELECT * FROM affarer", conn))
    conv_rate = total_calls and (total_aff / total_calls) or 0
    tb_per_min = total_time and (total_tb / total_time) or 0
    st.sidebar.metric("Konverteringsgrad", f"{conv_rate:.1%}")
    st.sidebar.metric("TB per minut", f"{tb_per_min:.2f} kr")

# --- TRAIN RECOMMENDATION MODEL ---
df_model = df_all[['samtal','tid_min','tb']].dropna()
if len(df_model) >= 5:
    X = df_model[['samtal','tid_min']]
    y = df_model['tb']
    rec_model = RandomForestRegressor(n_estimators=50, random_state=0)
    rec_model.fit(X, y)
else:
    rec_model = None

# --- TRAIN AFFÄRER CLUSTERING ---
df_a = pd.read_sql_query("SELECT minuter_till_stangning, tb FROM affarer", conn)
if len(df_a) >= 5:
    scaler = StandardScaler()
    Xa = scaler.fit_transform(df_a)
    kmeans = KMeans(n_clusters=2, random_state=0)
    df_a['cluster'] = kmeans.fit_predict(Xa)
else:
    kmeans = None
    df_a['cluster'] = []

# --- INPUT LAYOUT ---
col1, col2, col3 = st.columns([1.5,1,1])

# Daily inputs
with col1:
    st.subheader("🗓️ Dagslogg")
    datum = st.date_input('Datum', datetime.today())
    samtal = st.number_input('Antal samtal', 0)
    tid_h = st.number_input('Tid (timmar)', 0)
    tid_m = st.number_input('Tid (minuter)', 0)
    tid_min = tid_h*60 + tid_m
    tb = st.number_input('TB (kr)', 0.0, step=100.0)
    kommentar = st.text_input('Kommentar')
    energi = st.slider('Energinivå (1–5)', 1,5,3)
    humor = st.slider('Humör (1–5)',1,5,3)
    if st.button('💾 Spara dagslogg'):
        tb_ps = tb/samtal if samtal else 0
        tb_pt = tb/(tid_min/60) if tid_min else 0
        snitt = tid_min/samtal if samtal else 0
        lon = tb*0.45
        cursor.execute('''
            INSERT OR REPLACE INTO logg
            (datum,samtal,tid_min,tb,tb_per_samtal,tb_per_timme,snitt_min_per_samtal,lon,kommentar,energi,humor)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ''',(
            datum.strftime('%Y-%m-%d'),samtal,tid_min,tb,tb_ps,tb_pt,snitt,lon,kommentar,energi,humor
        ))
        conn.commit()
        st.success('Dagslogg sparad!')

# SMART goals with micro-tasks
with col2:
    st.subheader('🎯 Sätt SMART-mål')
    g_tb = st.number_input('TB-mål',0,step=100)
    g_samtal = st.number_input('Samtalsmål',0)
    g_lon = st.number_input('Lönemål',0,step=100)
    spec = st.text_input('Specifikt mål')
    real = st.text_input('Realistiskt mål')
    tids = st.text_input('Tidsbundet (YYYY-MM-DD)')
    if st.button('💾 Spara mål'):
        cursor.execute('''
            INSERT OR REPLACE INTO mal
            (datum,tb_mal,samtal_mal,lon_mal,specifikt,realistiskt,tidsbundet)
            VALUES(?,?,?,?,?,?,?)''',(
            datum.strftime('%Y-%m-%d'),g_tb,g_samtal,g_lon,spec,real,tids
        ))
        conn.commit()
        st.success('SMART-mål sparade!')
        if tids:
            days_left = max(1,(pd.to_datetime(tids) - pd.Timestamp(datum)).days)
        else:
            days_left = 1
        per_day_tb = g_tb/days_left
        st.write(f"⭐ Gör minst {per_day_tb:.0f} kr TB per dag för att nå målet.")

# Business entries
with col3:
    st.subheader('📤 Lägg till affär')
    aff_n = st.text_input('Affärsnamn')
    sent = st.time_input('Skickad tid')
    closed = st.time_input('Stängd tid')
    tb_a = st.number_input('TB affär',0.0,step=100.0)
    if st.button('📌 Spara affär'):
        diff = (datetime.combine(datum,closed)-datetime.combine(datum,sent)).seconds/60
        cursor.execute('''
            INSERT INTO affarer
            (datum,affar_namn,skickad_tid,stangd_tid,minuter_till_stangning,tb)
            VALUES(?,?,?,?,?,?)''',(
            datum.strftime('%Y-%m-%d'),aff_n,sent.strftime('%H:%M'),
            closed.strftime('%H:%M'),diff,tb_a
        ))
        conn.commit()
        st.success('Affär sparad!')

st.divider()
tab1,tab2,tab3,tab4 = st.tabs(['📊 Dag','📋 Affärer','🏆 Analys','🎯 Målhistorik'])

# Tab1: Day log & recommendation
with tab1:
    df = pd.read_sql_query('SELECT * FROM logg ORDER BY datum DESC',conn)
    if not df.empty:
        df['datum']=pd.to_datetime(df['datum'])
        st.dataframe(df,use_container_width=True)
        if rec_model:
            inc = st.slider('Öka samtal med (%)',-50,100,0)
            new_calls = df.iloc[0]['samtal']*(1+inc/100)
            pred = rec_model.predict([[new_calls, df.iloc[0]['tid_min']]])[0]
            st.write(f"➡️ Om du ökar samtalen med {inc}% → TB ≈ {pred:.0f} kr")

# Tab2: Business + clustering
with tab2:
    df2 = pd.read_sql_query('SELECT * FROM affarer',conn)
    if not df2.empty:
        df2['datum']=pd.to_datetime(df2['datum'])
        st.dataframe(df2,use_container_width=True)
        if kmeans:
            Xc = scaler.transform(df2[['minuter_till_stangning','tb']])
            df2['cluster']=kmeans.predict(Xc)
            st.subheader('Segmentering av affärer')
            fig,ax=plt.subplots()
            ax.scatter(df2['minuter_till_stangning'],df2['tb'],c=df2['cluster'],cmap='viridis')
            ax.set_xlabel('Tid (min)')
            ax.set_ylabel('TB')
            st.pyplot(fig)

# Tab3: Analysis & GPT summary
with tab3:
    df3 = pd.read_sql_query('SELECT * FROM logg',conn)
    if not df3.empty:
        df3['datum']=pd.to_datetime(df3['datum'])
        df3['vecka']=df3['datum'].dt.isocalendar().week
        weekly = df3.groupby('vecka')[['tb','samtal','lon']].sum()
        st.subheader('Veckosammanställning')
        st.dataframe(weekly)
        st.line_chart(weekly[['tb','lon']])
        # GPT summary
        week_num = weekly.index[-1]
        tot_tb = weekly['tb'].iloc[-1]
        days = df3[df3['vecka']==week_num]['datum'].nunique()
        prompt = f"Veckorapport: vecka {week_num}, totalt TB {tot_tb:.0f} kr över {days} dagar. Ge en peppande sammanfattning på svenska."
        if openai_client.api_key:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}]
            )
            st.markdown(resp.choices[0].message.content)
        else:
            st.info('Ange OPENAI_KEY i secrets för GPT-summering')

# Tab4: Goal history & simulation heatmap
with tab4:
    dfg = pd.read_sql_query('SELECT * FROM mal ORDER BY datum DESC',conn)
    if not dfg.empty:
        dfg['datum']=pd.to_datetime(dfg['datum'])
        st.dataframe(dfg,use_container_width=True)
        day = st.selectbox('Välj dag',dfg['datum'].dt.strftime('%Y-%m-%d'))
        gr = dfg[dfg['datum']==pd.to_datetime(day)].iloc[0]
        lr = pd.read_sql_query(f"SELECT * FROM logg WHERE datum='{day}'",conn).iloc[0]
        pct = lr['tb']/gr['tb_mal']*100 if gr['tb_mal'] else 0
        st.write(f"Måluppfyllelse TB: {pct:.0f}%")
        calls = lr['samtal']; tb_avg = lr['tb']/calls if calls else 0
        calls_range = np.arange(max(1,int(calls*0.7)),int(calls*1.5)+1)
        tb_range = np.linspace(tb_avg*0.8,tb_avg*1.3,10)
        mat = np.outer(calls_range,tb_range)
        fig,ax=plt.subplots()
        c=ax.imshow(mat,origin='lower',extent=[tb_range[0],tb_range[-1],calls_range[0],calls_range[-1]])
        fig.colorbar(c,ax=ax,label='TB')
        ax.set_xlabel('Snitt-TB'); ax.set_ylabel('Samtal')
        st.pyplot(fig)

# Export full log to Excel
buf = io.BytesIO()
pd.read_sql_query('SELECT * FROM logg',conn).to_excel(buf,index=False)
st.download_button('📥 Ladda ner logg.xlsx',buf.getvalue(),file_name='logg.xlsx')
