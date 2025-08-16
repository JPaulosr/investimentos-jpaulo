# APP_Proventos.py
# Lê a aba APP_Proventos por GID, corrige parse pt-BR e soma o Total Líquido ORIGINAL da planilha.

import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="APP | Proventos", page_icon="💸", layout="wide")
st.title("💸 Proventos & Calendário — APP_Proventos (planilha)")

# =========================
# CONFIG
# =========================
SHEET_ID = "1TQBzbueeBTgNmXwZPg04GFOwNL4vh_1ZbKlDAGQJ09o"
GID_APP_PROVENTOS = "1322179207"   # aba APP_Proventos

CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID_APP_PROVENTOS}"

# =========================
# HELPERS
# =========================
def parse_brl(x):
    """Converte strings em pt-BR para float (R$ 1.234,56 -> 1234.56).
       Aceita numérico, vazio, 'R$ ', etc."""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("R$", "").replace(" ", "")
    # vírgula decimal: troca vírgula por ponto e remove separador de milhar
    if re.search(r",\d{1,4}$", s):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def parse_int(x):
    try:
        return int(float(str(x).replace(",", ".").strip()))
    except:
        return 0

def normalizar_colunas(df):
    """Padroniza nomes mesmo que venham com variações."""
    def norm(c):
        c = (
            c.lower()
             .replace("(", " ").replace(")", " ")
             .replace("/", " ").replace("-", " ").replace(".", " ")
             .replace("  ", " ").strip()
        )
        mapa_acentos = (("á","a"),("à","a"),("ã","a"),("â","a"),
                        ("é","e"),("ê","e"),("í","i"),
                        ("ó","o"),("ô","o"),("õ","o"),
                        ("ú","u"),("ç","c"))
        for a,b in mapa_acentos:
            c = c.replace(a,b)
        return c.replace(" ", "_")

    df = df.copy()
    df.columns = [norm(c) for c in df.columns]

    mapa = {
        "ticker":"ticker", "ativo":"ticker", "codigo":"ticker",
        "data":"data",
        "tipo":"tipo","evento":"tipo",
        "quantidade":"quantidade","qtd":"quantidade","qtd_liquida":"quantidade",
        "unitario_r$":"unitario_rs","unitario":"unitario_rs","valor_unitario":"unitario_rs",
        "provento_unitario":"unitario_rs","valor_por_cota":"unitario_rs",
        "irrf":"irrf","imposto":"irrf",
        # valor líquido original da planilha (o que deve ser somado)
        "total_liquido_r$":"total_liquido_orig_rs","total_final_r$":"total_liquido_orig_rs",
        "valor_liquido":"total_liquido_orig_rs","total_liquido":"total_liquido_orig_rs",
        "liquido_r$":"total_liquido_orig_rs"
    }
    df = df.rename(columns={c: mapa.get(c,c) for c in df.columns})
    return df

def preparar_proventos(df):
    df = normalizar_colunas(df)

    # datas
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)

    # numéricos
    if "quantidade" in df.columns: df["quantidade"] = df["quantidade"].apply(parse_int)
    else: df["quantidade"] = 0

    if "unitario_rs" in df.columns: df["unitario_rs"] = df["unitario_rs"].apply(parse_brl)
    else: df["unitario_rs"] = 0.0

    if "irrf" in df.columns: df["irrf"] = df["irrf"].apply(parse_brl)
    else: df["irrf"] = 0.0

    if "total_liquido_orig_rs" in df.columns:
        df["total_liquido_orig_rs"] = df["total_liquido_orig_rs"].apply(parse_brl)
    else:
        df["total_liquido_orig_rs"] = 0.0

    # cálculo paralelo (auditoria)
    df["total_liquido_calc_rs"] = (df["quantidade"] * df["unitario_rs"]) - df["irrf"]

    # guard-rail: se unitário mediano está alto, provavelmente veio em centavos -> divide por 100
    med = df["unitario_rs"].median(skipna=True)
    if pd.notna(med) and med > 20:
        df["unitario_rs"] = df["unitario_rs"] / 100.0
        df["total_liquido_calc_rs"] = (df["quantidade"] * df["unitario_rs"]) - df["irrf"]

    df["diff_final_calc"] = df["total_liquido_calc_rs"] - df["total_liquido_orig_rs"]
    return df

@st.cache_data(ttl=300)
def carregar_df():
    # lê a aba APP_Proventos por gid (CSV export)
    df = pd.read_csv(CSV_URL, dtype=str)  # não deixa o pandas adivinhar número/moeda
    return preparar_proventos(df)

# =========================
# LOAD
# =========================
df = carregar_df()
if df.empty:
    st.warning("Não encontrei dados na aba APP_Proventos.")
    st.stop()

# =========================
# FILTROS
# =========================
col_f1, col_f2, col_f3, col_f4 = st.columns([1,1,1,2])

with col_f1:
    tickers = ["(Todos)"] + sorted([t for t in df["ticker"].dropna().astype(str).unique() if t.strip() != ""])
    tk = st.selectbox("Ticker", tickers, index=0)

with col_f2:
    anos = ["(Todos)"] + sorted(df["data"].dt.year.dropna().astype(int).unique().tolist())
    ano = st.selectbox("Ano", anos, index=0)

with col_f3:
    tipos = ["(Todos)"] + sorted([t for t in df.get("tipo", pd.Series(dtype=str)).fillna("").unique() if str(t).strip()!=""])
    tipo = st.selectbox("Tipo", tipos, index=0)

with col_f4:
    busca = st.text_input("Busca livre (descrição/observações, se houver)", "")

df_f = df.copy()
if tk != "(Todos)":
    df_f = df_f[df_f["ticker"].astype(str).str.upper() == tk.upper()]
if ano != "(Todos)":
    df_f = df_f[df_f["data"].dt.year == int(ano)]
if tipo != "(Todos)":
    df_f = df_f[df_f["tipo"].astype(str).str.upper() == str(tipo).upper()]
if busca.strip():
    # procura em todas as colunas de texto
    mask = pd.Series(False, index=df_f.index)
    for c in df_f.columns:
        if df_f[c].dtype == "O":
            mask = mask | df_f[c].astype(str).str.contains(busca, case=False, na=False)
    df_f = df_f[mask]

# =========================
# MÉTRICAS
# =========================
# Soma OFICIAL sempre do valor líquido ORIGINAL da planilha.
soma_oficial = df_f["total_liquido_orig_rs"].sum()
# fallback se a planilha não tiver essa coluna preenchida
if soma_oficial == 0:
    soma_oficial = df_f["total_liquido_calc_rs"].sum()

soma_calc = df_f["total_liquido_calc_rs"].sum()
dif = soma_calc - df_f["total_liquido_orig_rs"].sum()

def fmt_brl(v):
    return f"R$ {v:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

m1, m2, m3 = st.columns(3)
m1.metric("Soma usada (FINAL / planilha)", fmt_brl(soma_oficial))
m2.metric("Soma calculada (auditoria)", fmt_brl(soma_calc))
m3.metric("Diferença (calc - planilha)", fmt_brl(dif))

# =========================
# TABELAS
# =========================
st.subheader("🔎 Auditoria linha a linha (planilha × cálculo)")
cols_show = [
    "ticker","data","tipo","quantidade","unitario_rs","irrf",
    "total_liquido_orig_rs","total_liquido_calc_rs","diff_final_calc"
]
df_show = df_f[cols_show].sort_values("total_liquido_calc_rs", ascending=False)
st.dataframe(df_show, use_container_width=True)

st.subheader("📊 Soma por Ticker (valor oficial da planilha)")
grp = df_f.groupby("ticker", dropna=True, as_index=False)["total_liquido_orig_rs"].sum()
grp = grp.sort_values("total_liquido_orig_rs", ascending=False)
st.dataframe(grp.rename(columns={"ticker":"Ticker","total_liquido_orig_rs":"Total Líquido (planilha)"}),
             use_container_width=True)
