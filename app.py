import streamlit as st
import pandas as pd
from datetime import datetime
import openai
from supabase import create_client, Client

# --- SETTINGS ---
st.set_page_config(page_title="📈 DaVinci's Duk", layout="wide")
st.title("📈 DaVinci's Duk – Försäljningssystem")
st.markdown("💡 Fokusera på process, insikt och förbättring varje dag.")

# --- SUPABASE SETUP ---
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(supabase_url, supabase_key)

# --- OPENAI SETUP ---
openai.api_key = st.secrets["openai"]["api_key"]
openai_model = st.secrets["openai"].get("model", "gpt-4")

# --- LAYOUT ---
tabs = st.tabs(["📅 Dagslogg & Affärer", "⭐ Guldkunder", "🔁 Återkomster", "✅ Klara Kunder", "🎯 Mål"])

# --- FUNCTIONS ---
def fetch_table(table):
    return supabase.table(table).select("*").execute().data

def insert_row(table, data):
    supabase.table(table).insert(data).execute()

def update_row(table, match_column, match_value, data):
    supabase.table(table).update(data).eq(match_column, match_value).execute()

def delete_row(table, match_column, match_value):
    supabase.table(table).delete().eq(match_column, match_value).execute()

def gpt_analyze(affarer_df):
    if affarer_df.empty:
        return "Inga affärer att analysera idag."
    prompt = f"Analysera följande affärer (i dataframe-format):\n{affarer_df.to_string(index=False)}\n\nVad fungerade bäst idag? Vad bör prioriteras imorgon? Ge även ett rekommenderat TB-mål och antal samtal."
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 1. DAGSLOGG & AFFÄRER ---
with tabs[0]:
    st.header("🗓️ Dagslogg")
    datum = st.date_input("Datum", datetime.today())
    samtal = st.number_input("Antal samtal", min_value=0, step=1)
    tb_dag = st.number_input("TB (kr) för dagen", min_value=0.0, step=100.0)
    energi = st.slider("Energi (1–5)", 1, 5, 3)
    humor = st.slider("Humör (1–5)", 1, 5, 3)

    if st.button("💾 Spara dagslogg"):
        insert_row("logg", {
            "datum": str(datum), "samtal": samtal,
            "tb": tb_dag, "energi": energi, "humor": humor
        })
        st.success("Dagslogg sparad!")

    st.header("📤 Lägg till affär")
    kol1, kol2 = st.columns(2)
    with kol1:
        foretagsnamn = st.text_input("Företagsnamn")
        bolagstyp = st.selectbox("Bolagstyp", ["Enskild firma", "Aktiebolag"])
        dealtyp = st.selectbox("Typ av affär", ["Nyteckning", "Förlängning"])
        abonnemang = st.number_input("Antal abonnemang", min_value=0)
        tb = st.number_input("TB för affären", min_value=0.0)
        cashback = st.number_input("Cashback", min_value=0.0)
    with kol2:
        skickad = st.time_input("Skickad tid")
        stangd = st.time_input("Stängd tid")
        diff = (datetime.combine(datetime.today(), stangd) - datetime.combine(datetime.today(), skickad)).seconds / 60
        modell = st.text_input("Modell på hårdvara")
        typ = st.text_input("Typ av hårdvara")
        antal = st.number_input("Antal hårdvaror", min_value=0)
        inkopspris = st.number_input("Förhöjd avgift / inköpspris", min_value=0.0)

    if st.button("➕ Lägg till affär"):
        insert_row("affarer", {
            "datum": str(datum), "foretagsnamn": foretagsnamn,
            "bolagstyp": bolagstyp, "dealtyp": dealtyp, "abonnemang": abonnemang,
            "tb": tb, "cashback": cashback, "minuter_till_stangning": diff,
            "modell": modell, "typ": typ, "antal": antal, "inkopspris": inkopspris
        })
        st.success("Affär tillagd!")

    st.subheader("📊 Dagens affärer")
    dagens_aff = supabase.table("affarer").select("*").eq("datum", str(datum)).execute().data
    df_aff = pd.DataFrame(dagens_aff)
    st.dataframe(df_aff)

    st.subheader("🧠 GPT-analys av dagens affärer")
    if st.button("🔍 Kör analys"):
        with st.spinner("Analyserar med GPT..."):
            analys = gpt_analyze(pd.DataFrame(dagens_aff))
            st.markdown(analys)

# --- 2. GULDKUNDER / ÅTERKOMSTER / KLARA KUNDER ---
def customer_tab(tab_index, table_name):
    with tabs[tab_index]:
        st.header(f"🗂️ {table_name.capitalize()}")
        data = fetch_table(table_name)
        df = pd.DataFrame(data)
        search = st.text_input("🔍 Sök kund (namn, orgnr/personnr, företagsnamn)")
        if search:
            df = df[df.apply(lambda row: search.lower() in str(row.values).lower(), axis=1)]
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        with st.expander("➕ Lägg till ny kund"):
            namn = st.text_input("Namn")
            org = st.text_input("Person-/Organisationsnummer")
            foretag = st.text_input("Företagsnamn")
            kommentar = st.text_area("Kommentar")
            if st.button(f"Spara till {table_name}"):
                insert_row(table_name, {
                    "namn": namn,
                    "orgnr": org,
                    "foretagsnamn": foretag,
                    "kommentar": kommentar,
                    "datum": str(datetime.today().date())
                })
                st.success("Kund sparad!")

customer_tab(1, "guldkunder")
customer_tab(2, "aterkomster")
customer_tab(3, "klarakunder")

# --- 3. MÅLFUNKTION ---
with tabs[4]:
    st.header("🎯 TB-mål per månad")
    month = st.selectbox("Välj månad", pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS").strftime("%Y-%m"))
    current = supabase.table("mal").select("*").eq("manad", month).execute().data
    nuvarande_tb = current[0]["tb"] if current else 0

    nytt_tb = st.number_input("Sätt/ändra TB-mål (kr)", value=nuvarande_tb, step=1000)
    if st.button("💾 Spara mål"):
        if current:
            update_row("mal", "manad", month, {"tb": nytt_tb})
        else:
            insert_row("mal", {"manad": month, "tb": nytt_tb})
        st.success("Mål uppdaterat!")

    dagloggar = pd.DataFrame(fetch_table("logg"))
    dagloggar["datum"] = pd.to_datetime(dagloggar["datum"])
    totalt = dagloggar[dagloggar["datum"].dt.strftime("%Y-%m") == month]["tb"].sum()
    st.progress(min(1.0, totalt / nytt_tb))
    st.write(f"Uppnått TB: {int(totalt)} kr / {int(nytt_tb)} kr")
