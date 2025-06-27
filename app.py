# app.py

import streamlit as st
import pandas as pd
import openai
from datetime import datetime, date
from supabase import create_client, Client
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain
from streamlit_extras.app_logo import add_logo
import plotly.express as px
import plotly.graph_objects as go
import calendar

# --- CONFIG ---
st.set_page_config(page_title="DaVinci's Duk", layout="wide")
st.title("🧠 DaVinci's Duk – Sälj & Kundöversikt")

# --- SUPABASE ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- GPT ANALYS ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
GPT_MODEL = "gpt-4"

def gpt_analysera_affarer(affarer_df):
    if affarer_df.empty:
        return "Inga affärer att analysera."
    prompt = f"""
    Här är dagens affärer:

    {affarer_df.to_string(index=False)}

    Analysera vilka typer av affärer som varit mest lönsamma, eventuella mönster, vad som borde optimeras, och ge ett konkret förslag på samtalsmål och TB-mål för nästa dag.
    """
    try:
        res = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Du är en analytisk säljspecialist."},
                {"role": "user", "content": prompt}
            ]
        )
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Fel vid GPT-anrop: {e}"

# --- DATAFETCH UT ---
def fetch_data(tabell):
    return supabase.table(tabell).select("*").execute().data

def save_data(tabell, data):
    supabase.table(tabell).insert(data).execute()

# --- SIDOMENY ---
flik = st.sidebar.selectbox("Navigera", [
    "Dagslogg & Affärer",
    "Guldkunder",
    "Återkomster",
    "Klara kunder",
    "TB-mål per månad"
])

# --- GEMENSAM FILTER ---
df_all = pd.DataFrame(fetch_data("affarer"))
df_all["datum"] = pd.to_datetime(df_all["datum"], errors='coerce')

# --- DAGLOGG & AFFÄRER ---
if flik == "Dagslogg & Affärer":
    with st.form("dagform"):
        st.subheader("🗓️ Logga affär")
        col1, col2, col3 = st.columns(3)
        with col1:
            datum = st.date_input("Datum", value=date.today())
            kundnamn = st.text_input("Kundnamn")
            foretagsnamn = st.text_input("Företagsnamn")
            orgnr = st.text_input("Org-/Pers.nr")
            abonnemang = st.number_input("Abonnemang sålda", step=1)
        with col2:
            tb = st.number_input("TB", step=100.0)
            cashback = st.number_input("Cashback", step=100.0)
            dealtyp = st.selectbox("Nyteckning / Förlängning", ["Nyteckning", "Förlängning"])
            skickad = st.time_input("Skickad tid")
            stangd = st.time_input("Stängd tid")
        with col3:
            hw_typ = st.text_input("Hårdvara typ")
            hw_model = st.text_input("Hårdvara modell")
            hw_pris = st.number_input("Inköpspris/förhöjd avgift", step=100.0)
            hw_tb = st.number_input("TB från hårdvara", step=100.0)

        if st.form_submit_button("💾 Spara affär"):
            total_tb = tb + hw_tb
            minut_diff = (datetime.combine(date.today(), stangd) - datetime.combine(date.today(), skickad)).seconds / 60
            save_data("affarer", {
                "datum": str(datum),
                "kundnamn": kundnamn,
                "foretagsnamn": foretagsnamn,
                "orgnr": orgnr,
                "abonnemang": abonnemang,
                "tb": total_tb,
                "cashback": cashback,
                "dealtyp": dealtyp,
                "skickad": str(skickad),
                "stangd": str(stangd),
                "minuter_till_stangning": minut_diff,
                "hw_typ": hw_typ,
                "hw_model": hw_model,
                "hw_pris": hw_pris,
                "hw_tb": hw_tb
            })
            st.success("Affär sparad!")

    st.subheader("📊 Dagens affärer")
    idag_df = df_all[df_all["datum"] == pd.to_datetime(date.today())]
    st.dataframe(idag_df, use_container_width=True)

    st.subheader("🤖 GPT-analys")
    analys = gpt_analysera_affarer(idag_df)
    st.write(analys)

# --- GULDKUNDER MFL ---
def visa_flik(tabell_namn, rubrik):
    st.subheader(f"📁 {rubrik}")
    df = pd.DataFrame(fetch_data(tabell_namn))
    if df.empty:
        st.info("Inga kunder än.")
        return

    df["datum"] = pd.to_datetime(df["datum"], errors='coerce')

    med1, med2 = st.columns([2,2])
    with med1:
        valdatum = st.date_input("Filtrera på datum")
        df = df[df["datum"] == pd.to_datetime(valdatum)]
    with med2:
        sok = st.text_input("Sök på kund, företag eller org.nr")
        if sok:
            df = df[df.apply(lambda r: sok.lower() in str(r).lower(), axis=1)]

    st.dataframe(df, use_container_width=True)

if flik == "Guldkunder":
    visa_flik("guldkunder", "Guldkunder")
elif flik == "Återkomster":
    visa_flik("aterkomster", "Återkomster")
elif flik == "Klara kunder":
    visa_flik("klarakunder", "Klara kunder")

# --- TB-MÅL MÅNAD FÖR MÅNAD ---
if flik == "TB-mål per månad":
    st.subheader("🎯 Månadsmål & Progress")
    current_month = datetime.today().strftime("%Y-%m")
    all_logg = pd.DataFrame(fetch_data("affarer"))
    all_logg["datum"] = pd.to_datetime(all_logg["datum"], errors='coerce')
    all_logg["manad"] = all_logg["datum"].dt.strftime("%Y-%m")
    tb_mtd = all_logg[all_logg["manad"] == current_month]["tb"].sum()

    st.write(f"TB hittills denna månad ({current_month}): **{int(tb_mtd)} kr**")
    mal_input = st.number_input("Sätt/uppdatera TB-mål för denna månad", step=1000.0)

    if st.button("💾 Spara mål"):
        supabase.table("manadsmal").upsert({"manad": current_month, "tb_mal": mal_input}).execute()
        st.success("Mål uppdaterat")

    mal_df = pd.DataFrame(fetch_data("manadsmal"))
    if current_month in mal_df["manad"].values:
        mal = mal_df[mal_df["manad"] == current_month]["tb_mal"].iloc[0]
        prog = min(tb_mtd / mal, 1.0) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prog,
            title={'text': "Uppnått % av målet"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sätt ett mål först.")
