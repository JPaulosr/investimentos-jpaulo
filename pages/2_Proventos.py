# pages/2_💰_Proventos.py
# Página dedicada a Proventos (multipage)
# - Lê a aba "3. Proventos" via Service Account (gspread) como prioridade
# - Detecta cabeçalho por palavras‑chave (resolve header com linhas acima)
# - Filtros (período, classe, ticker) e KPIs (Total, YTD, últimos 12m)
# - Gráficos: Proventos por mês, por ticker e por classe
# - Botão para download do CSV filtrado

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="💰 Proventos", page_icon="💰", layout="wide")
st.title("💰 Proventos")

PLOTLY_TEMPLATE = "plotly_dark"

# ==============================
# Secrets / Config
# ==============================
SHEET_ID = st.secrets.get("SHEET_ID", "").strip()
ABA_PROVENTOS = st.secrets.get("ABA_PROVENTOS", "3. Proventos")
ABA_PROVENTOS_GID = str(st.secrets.get("ABA_PROVENTOS_GID", "")).strip()

# ==============================
# Helpers numéricos / datas
# ==============================
def br_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-", "--"}:
        return None
    s = (s.replace("R$", "").replace("US$", "").replace("$", "")
           .replace("%", "").replace(" ", ""))
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def to_datetime_br(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def moeda_br(v):
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==============================
# Service Account + leitores
# ==============================
def _has_sa():
    return bool(st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account"))

def _get_sa_info():
    return st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account") or {}

def _find_header_row(values, expect_cols):
    exp = [e.strip().lower() for e in expect_cols]
    best = None
    best_hits = 0
    for i, row in enumerate(values):
        row_low = [str(c).strip().lower() for c in row]
        hits = sum(1 for e in exp if e in row_low)
        if hits > best_hits:
            best_hits, best = hits, i
        if hits >= 2:
            return i
    return best if best is not None else 0

def _read_ws_values(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials
    info = _get_sa_info()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(aba_nome)
    except Exception:
        titles = [w.title for w in sh.worksheets()]
        match = next((t for t in titles if t.casefold() == aba_nome.casefold()), None)
        ws = sh.worksheet(match if match else aba_nome)

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    expect = ["Ticker", "Tipo Provento", "Data"]
    header_idx = _find_header_row(values, expect)
    headers_raw = [h.strip() for h in values[header_idx]]
    seen, headers = {}, []
    for h in headers_raw:
        base = h if h else "col"
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df = pd.DataFrame(values[header_idx + 1:], columns=headers)
    df = df.replace({"": None}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df

def _read_csv_by_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    import urllib.error
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

def _read_csv_by_name(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    import urllib.error
    from urllib.parse import quote
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(aba_nome, safe='')}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def ler_proventos(sheet_id: str, aba_nome: str, gid: str="") -> pd.DataFrame:
    # 1) Service Account
    if sheet_id and _has_sa():
        try:
            df = _read_ws_values(sheet_id, aba_nome)
            if not df.empty:
                st.session_state["proventos_modo"] = "service_account"
                return df
        except Exception as e:
            st.info(f"[SA] leitura de '{aba_nome}' falhou: {e}")
    # 2) CSV por GID
    if sheet_id and gid:
        df = _read_csv_by_gid(sheet_id, gid)
        if not df.empty:
            st.session_state["proventos_modo"] = "csv_gid"
            return df
    # 3) CSV por NOME
    if sheet_id and aba_nome:
        df = _read_csv_by_name(sheet_id, aba_nome)
        if not df.empty:
            st.session_state["proventos_modo"] = "csv_nome"
            return df
    st.session_state["proventos_modo"] = "falhou"
    return pd.DataFrame()

# ==============================
# Padronização Proventos
# ==============================
def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo", "Tipo Provento"],
        "ValorUnitario": ["Unitário R$", "Unitario R$", "Unitário", "Unitario"],
        "Valor": ["Total Líquido R$", "Total Liquido R$", "Valor", "Total"],
        "Classe": ["Classe do Ativo", "Classe"],
        "Quantidade": ["Quantidade", "Qtd"],
        "PTAX": ["PTAX"],
        "ValorBruto": ["Total Bruto R$", "Total Bruto"],
        "IRRF": ["IRRF"],
        "Mes": ["Mês","Mes"],
        "Ano": ["Ano"],
    }
    out = pd.DataFrame()
    for novo, cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None

    out["Data"] = to_datetime_br(out["Data"])
    for col in ["Quantidade","ValorUnitario","Valor","ValorBruto","IRRF"]:
        if col in out.columns:
            out[col] = out[col].map(br_to_float)

    if "Valor" not in out.columns or out["Valor"].isna().all():
        if {"Quantidade","ValorUnitario"}.issubset(out.columns):
            out["Valor"] = out["Quantidade"].fillna(0).astype(float) * out["ValorUnitario"].fillna(0).astype(float)

    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()

    return out

# ==============================
# Carregar dados
# ==============================
if not SHEET_ID:
    st.error("❌ `SHEET_ID` não definido nos secrets.")
    st.stop()

with st.spinner("Carregando Proventos..."):
    df_raw = ler_proventos(SHEET_ID, ABA_PROVENTOS, ABA_PROVENTOS_GID)
PV = padronizar_proventos(df_raw)

# ==============================
# Filtros (período / classe / ticker)
# ==============================
with st.sidebar:
    st.header("Filtros")
    if PV.empty or "Data" not in PV.columns:
        st.info("Sem dados de Proventos.")
    else:
        dmin = PV["Data"].dropna().min().date()
        dmax = PV["Data"].dropna().max().date()
        periodo = st.date_input("Período", value=(max(dmin, dmax - timedelta(days=365)), dmax),
                                min_value=dmin, max_value=dmax)

        def uniq(s):
            if s in PV.columns:
                return sorted(PV[s].dropna().astype(str).unique().tolist())
            return []
        classe_sel = st.multiselect("Classe", options=uniq("Classe"), default=uniq("Classe"))
        ticker_sel = st.multiselect("Ticker", options=uniq("Ticker"))

# aplica filtros
if not PV.empty and "Data" in PV.columns:
    if isinstance(periodo, tuple) and len(periodo) == 2:
        d0, d1 = periodo
    else:
        d0, d1 = PV["Data"].min().date(), PV["Data"].max().date()
    PV = PV[PV["Data"].between(pd.to_datetime(d0), pd.to_datetime(d1))]
if "Classe" in PV.columns and 'classe_sel' in locals() and classe_sel:
    PV = PV[PV["Classe"].isin(classe_sel)]
if "Ticker" in PV.columns and 'ticker_sel' in locals() and ticker_sel:
    PV = PV[PV["Ticker"].isin(ticker_sel)]

# ==============================
# KPIs
# ==============================
if PV.empty:
    st.info("Nenhum provento no período/seleção atual.")
    st.stop()

total_periodo = PV["Valor"].sum(skipna=True) if "Valor" in PV.columns else 0.0
hoje = date.today()
inicio_ytd = date(hoje.year, 1, 1)
ytd = PV[PV["Data"].dt.date >= inicio_ytd]["Valor"].sum() if "Valor" in PV.columns else 0.0
ult_12m_inicio = date(hoje.year - 1, hoje.month, 1)
ult12m = PV[PV["Data"].dt.date >= ult_12m_inicio]["Valor"].sum() if "Valor" in PV.columns else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total no Período", moeda_br(total_periodo))
c2.metric("Total YTD", moeda_br(ytd))
c3.metric("Total últimos 12 meses", moeda_br(ult12m))

# ==============================
# Gráficos
# ==============================
PVm = PV.dropna(subset=["Data"]).copy()
PVm["Competencia"] = pd.to_datetime(PVm["Data"].dt.strftime("%Y-%m-01"))
grp_mes = PVm.groupby("Competencia", dropna=False)["Valor"].sum().reset_index()

fig1 = px.bar(grp_mes, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos por Mês")
fig1.update_layout(xaxis_title="Competência", yaxis_title="R$")
st.plotly_chart(fig1, use_container_width=True)

colA, colB = st.columns(2)

with colA:
    if "Ticker" in PV.columns:
        top_ticker = PV.groupby("Ticker", dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
        fig2 = px.bar(top_ticker, x="Ticker", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos por Ticker")
        fig2.update_layout(yaxis_title="R$")
        st.plotly_chart(fig2, use_container_width=True)

with colB:
    if "Classe" in PV.columns:
        por_classe = PV.groupby("Classe", dropna=False)["Valor"].sum().reset_index()
        fig3 = px.pie(por_classe, names="Classe", values="Valor", hole=0.4, template=PLOTLY_TEMPLATE, title="Proventos por Classe")
        st.plotly_chart(fig3, use_container_width=True)

# ==============================
# Tabela e Download
# ==============================
st.subheader("Tabela de Proventos (filtrada)")
st.dataframe(PV.sort_values("Data"), use_container_width=True, hide_index=True)

csv = PV.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV (filtrado)", data=csv, file_name="proventos_filtrado.csv", mime="text/csv")

# ==============================
# Rodapé de diagnóstico
# ==============================
modo = st.session_state.get("proventos_modo", "desconhecido")
st.caption(f"Modo de leitura: **{modo}** (aba '{ABA_PROVENTOS}')")
