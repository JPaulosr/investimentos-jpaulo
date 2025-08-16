# 101_Resumo_Investimentos.py
# Resumo de Proventos (somando SEMPRE o valor líquido original da planilha)
# — Isolado desta página, não altera sua página atual de proventos.

import io
import re
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Resumo Investimentos", page_icon="📈", layout="wide")
st.title("📈 Resumo de Proventos — (somando o Líquido da planilha)")

# =========================
# CONFIG — sua planilha/aba
# =========================
SHEET_ID = "1TQBzbueeBTgNmXwZPg04GFOwNL4vh_1ZbKlDAGQJ09o"  # <- SUA planilha
GID_APP_PROVENTOS = "1322179207"                          # <- aba APP_Proventos

# =========================
# URLs auxiliares
# =========================
def _csv_url_export(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

def _csv_url_gviz(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={gid}"

def _fetch_csv(url, timeout=25):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
        "Referer": "https://docs.google.com/"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), dtype=str)

# =========================
# Carregamento robusto
# =========================
@st.cache_data(ttl=300, show_spinner=True)
def carregar_bruto():
    e1 = e2 = e3 = None
    # 1) export?format=csv
    try:
        return _fetch_csv(_csv_url_export(GID_APP_PROVENTOS))
    except Exception as ex:
        e1 = ex
        st.info(f"Rota 1 (export CSV) falhou: {ex}. Tentando rota 2…")

    # 2) gviz/tq
    try:
        return _fetch_csv(_csv_url_gviz(GID_APP_PROVENTOS))
    except Exception as ex:
        e2 = ex
        st.info(f"Rota 2 (gviz CSV) falhou: {ex}. Tentando Service Account…")

    # 3) gspread
    try:
        from google.oauth2.service_account import Credentials
        import gspread
        info = st.secrets.get("gcp_service_account") or st.secrets.get("GCP_SERVICE_ACCOUNT")
        if not info:
            raise RuntimeError("Secrets gcp_service_account ausentes.")
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = None
        for w in sh.worksheets():
            if str(w.id) == str(GID_APP_PROVENTOS):
                ws = w; break
        if ws is None:
            raise RuntimeError(f"Aba com gid={GID_APP_PROVENTOS} não encontrada.")
        data = ws.get_all_records()
        return pd.DataFrame(data, dtype=str)
    except Exception as ex:
        e3 = ex
        st.error(
            "❌ Não foi possível ler a aba `APP_Proventos`.\n\n"
            f"- Erro rota 1 (export CSV): {e1}\n"
            f"- Erro rota 2 (gviz CSV): {e2}\n"
            f"- Erro rota 3 (gspread): {e3}\n\n"
            "👉 Soluções: (a) deixe a planilha **pública com link (Leitor)**, ou "
            "(b) compartilhe com o e-mail da **Service Account** presente em `st.secrets`."
        )
        return pd.DataFrame()

# =========================
# Helpers de normalização
# =========================
def parse_brl(x):
    """Converte 'R$ 1.234,56' -> 1234.56; aceita numérico ou vazio."""
    if pd.isna(x): return 0.0
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "": return 0.0
    s = s.replace("R$", "").replace(" ", "")
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

def _norm_cols(df):
    def norm(c):
        c = (
            c.lower()
             .replace("(", " ").replace(")", " ")
             .replace("/", " ").replace("-", " ").replace(".", " ")
             .replace("  ", " ").strip()
        )
        for a,b in (("á","a"),("à","a"),("ã","a"),("â","a"),
                    ("é","e"),("ê","e"),("í","i"),
                    ("ó","o"),("ô","o"),("õ","o"),
                    ("ú","u"),("ç","c")):
            c = c.replace(a,b)
        return c.replace(" ", "_")

    df = df.copy()
    df.columns = [_norm if ( _norm:=norm(c) ) else c for c in df.columns]

    mapa = {
        "ticker":"ticker", "ativo":"ticker", "codigo":"ticker",
        "data":"data",
        "tipo":"tipo", "evento":"tipo",
        "quantidade":"quantidade", "qtd":"quantidade", "qtd_liquida":"quantidade",
        "unitario_r$":"unitario_rs","unitario":"unitario_rs","valor_unitario":"unitario_rs",
        "provento_unitario":"unitario_rs","valor_por_cota":"unitario_rs",
        "irrf":"irrf","imposto":"irrf",
        "total_liquido_r$":"total_liquido_orig_rs","total_final_r$":"total_liquido_orig_rs",
        "valor_liquido":"total_liquido_orig_rs","total_liquido":"total_liquido_orig_rs",
        "liquido_r$":"total_liquido_orig_rs"
    }
    return df.rename(columns={c: mapa.get(c,c) for c in df.columns})

def preparar(df):
    df = _norm_cols(df)

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

    # cálculo paralelo (AUDITORIA)
    df["total_liquido_calc_rs"] = (df["quantidade"] * df["unitario_rs"]) - df["irrf"]

    # guard-rail: se unitário mediano está alto, provavelmente veio em centavos
    med = df["unitario_rs"].median(skipna=True)
    if pd.notna(med) and med > 20:
        df["unitario_rs"] /= 100.0
        df["total_liquido_calc_rs"] = (df["quantidade"] * df["unitario_rs"]) - df["irrf"]

    df["diff_final_calc"] = df["total_liquido_calc_rs"] - df["total_liquido_orig_rs"]
    return df

def fmt_brl(v):
    return f"R$ {v:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

# =========================
# Load
# =========================
df_raw = carregar_bruto()
if df_raw.empty:
    st.stop()

df = preparar(df_raw)

# =========================
# Filtros
# =========================
c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    tickers = ["(Todos)"] + sorted([t for t in df["ticker"].dropna().astype(str).unique() if t.strip()!=""])
    tk = st.selectbox("Ticker", tickers, index=0)
with c2:
    anos = ["(Todos)"] + sorted(df["data"].dt.year.dropna().astype(int).unique().tolist())
    ano = st.selectbox("Ano", anos, index=0)
with c3:
    tipos = ["(Todos)"] + sorted([t for t in df.get("tipo", pd.Series(dtype=str)).fillna("").unique() if str(t).strip()!=""])
    tipo = st.selectbox("Tipo", tipos, index=0)
with c4:
    busca = st.text_input("Busca livre (se existir coluna de descrição/obs.)", "")

df_f = df.copy()
if tk != "(Todos)":
    df_f = df_f[df_f["ticker"].astype(str).str.upper() == tk.upper()]
if ano != "(Todos)":
    df_f = df_f[df_f["data"].dt.year == int(ano)]
if tipo != "(Todos)":
    df_f = df_f[df_f["tipo"].astype(str).str.upper() == str(tipo).upper()]
if busca.strip():
    mask = pd.Series(False, index=df_f.index)
    for c in df_f.columns:
        if df_f[c].dtype == "O":
            mask |= df_f[c].astype(str).str.contains(busca, case=False, na=False)
    df_f = df_f[mask]

# =========================
# Métricas (OFICIAL = planilha)
# =========================
soma_oficial = df_f["total_liquido_orig_rs"].sum()
if soma_oficial == 0:  # fallback se a planilha não tiver essa coluna
    soma_oficial = df_f["total_liquido_calc_rs"].sum()

soma_calc = df_f["total_liquido_calc_rs"].sum()
dif = soma_calc - df_f["total_liquido_orig_rs"].sum()

m1,m2,m3 = st.columns(3)
m1.metric("Soma usada (FINAL / planilha)", fmt_brl(soma_oficial))
m2.metric("Soma calculada (auditoria)", fmt_brl(soma_calc))
m3.metric("Diferença (calc - planilha)", fmt_brl(dif))

# =========================
# Tabelas
# =========================
st.subheader("🔎 Auditoria (linha a linha)")
cols_show = [
    "ticker","data","tipo","quantidade","unitario_rs","irrf",
    "total_liquido_orig_rs","total_liquido_calc_rs","diff_final_calc"
]
st.dataframe(df_f[cols_show].sort_values("total_liquido_calc_rs", ascending=False),
             use_container_width=True)

st.subheader("📊 Soma por Ticker (valor oficial da planilha)")
grp_t = (df_f.groupby("ticker", as_index=False)["total_liquido_orig_rs"]
              .sum().sort_values("total_liquido_orig_rs", ascending=False))
st.dataframe(grp_t.rename(columns={"ticker":"Ticker","total_liquido_orig_rs":"Total Líquido (planilha)"}),
             use_container_width=True)

st.subheader("📅 Soma mensal (valor oficial da planilha)")
if "data" in df_f.columns:
    dm = df_f.copy()
    dm["ano_mes"] = dm["data"].dt.to_period("M").astype(str)
    grp_m = (dm.groupby("ano_mes", as_index=False)["total_liquido_orig_rs"]
               .sum().sort_values("ano_mes"))
    st.dataframe(grp_m.rename(columns={"ano_mes":"Ano-Mês","total_liquido_orig_rs":"Total Líquido (planilha)"}),
                 use_container_width=True)
else:
    st.info("Coluna de data não disponível para agregar por mês.")
