# pages/2_💰_Proventos.py — página Proventos (turbinada + patch de boxplot)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from io import BytesIO

st.set_page_config(page_title="💰 Proventos", page_icon="💰", layout="wide")
st.title("💰 Proventos")
PLOTLY_TEMPLATE = "plotly_dark"

# -------------------- Config / Secrets --------------------
SHEET_ID = st.secrets.get("SHEET_ID", "").strip()
ABA_PROVENTOS = st.secrets.get("ABA_PROVENTOS", "3. Proventos")
ABA_PROVENTOS_GID = str(st.secrets.get("ABA_PROVENTOS_GID", "")).strip()

# -------------------- Helpers --------------------
def br_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","-","--"}: return None
    s = (s.replace("R$", "").replace("US$", "").replace("$","")
           .replace("%","").replace(" ","")).replace(".","").replace(",",".")
    try: return float(s)
    except: return None

def to_datetime_br(series): return pd.to_datetime(series, dayfirst=True, errors="coerce")
def moeda_br(v):
    try: v = float(v)
    except: v = 0.0
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def _has_sa(): return bool(st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account"))
def _get_sa_info(): return st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account") or {}

def _find_header_row(values, expect_cols):
    exp = [e.strip().lower() for e in expect_cols]
    best, hits_best = 0, 0
    for i,row in enumerate(values):
        row_low = [str(c).strip().lower() for c in row]
        hits = sum(1 for e in exp if e in row_low)
        if hits > hits_best: best, hits_best = i, hits
        if hits >= 2: return i
    return best

def _read_ws_values(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials
    info = _get_sa_info()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
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
    if not values: return pd.DataFrame()
    header_idx = _find_header_row(values, ["Ticker","Tipo Provento","Data"])
    headers_raw = [h.strip() for h in values[header_idx]]
    seen, headers = {}, []
    for h in headers_raw:
        base = h if h else "col"
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base]==1 else f"{base}_{seen[base]}")
    df = pd.DataFrame(values[header_idx+1:], columns=headers)
    df = df.replace({"": None}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df

def _read_csv_by_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    import urllib.error
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        if e.code in (401,403): return pd.DataFrame()
        raise
    except Exception: return pd.DataFrame()

def _read_csv_by_name(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    import urllib.error
    from urllib.parse import quote
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(aba_nome, safe='')}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        if e.code in (401,403): return pd.DataFrame()
        raise
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def ler_proventos(sheet_id: str, aba_nome: str, gid: str="") -> pd.DataFrame:
    if sheet_id and _has_sa():
        try:
            df = _read_ws_values(sheet_id, aba_nome)
            if not df.empty:
                st.session_state["proventos_modo"] = "service_account"
                return df
        except Exception as e:
            st.info(f"[SA] leitura de '{aba_nome}' falhou: {e}")
    if sheet_id and gid:
        df = _read_csv_by_gid(sheet_id, gid)
        if not df.empty:
            st.session_state["proventos_modo"] = "csv_gid"
            return df
    if sheet_id and aba_nome:
        df = _read_csv_by_name(sheet_id, aba_nome)
        if not df.empty:
            st.session_state["proventos_modo"] = "csv_nome"
            return df
    st.session_state["proventos_modo"] = "falhou"
    return pd.DataFrame()

def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo","Tipo Provento"],
        "ValorUnitario": ["Unitário R$","Unitario R$","Unitário","Unitario"],
        "Valor": ["Total Líquido R$","Total Liquido R$","Valor","Total"],
        "Classe": ["Classe do Ativo","Classe"],
        "Quantidade": ["Quantidade","Qtd"],
        "PTAX": ["PTAX"], "ValorBruto": ["Total Bruto R$","Total Bruto"],
        "IRRF": ["IRRF"], "Mes": ["Mês","Mes"], "Ano": ["Ano"],
    }
    out = pd.DataFrame()
    for novo,cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None
    out["Data"] = to_datetime_br(out["Data"])
    for c in ["Quantidade","ValorUnitario","Valor","ValorBruto","IRRF"]:
        if c in out.columns: out[c] = out[c].map(br_to_float)
    if "Valor" not in out.columns or out["Valor"].isna().all():
        if {"Quantidade","ValorUnitario"}.issubset(out.columns):
            out["Valor"] = out["Quantidade"].fillna(0).astype(float) * out["ValorUnitario"].fillna(0).astype(float)
    if "Tipo" in out.columns: out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
    return out

# -------------------- Carregar --------------------
if not SHEET_ID:
    st.error("❌ `SHEET_ID` não definido nos secrets.")
    st.stop()

with st.spinner("Carregando Proventos..."):
    df_raw = ler_proventos(SHEET_ID, ABA_PROVENTOS, ABA_PROVENTOS_GID)
PV = padronizar_proventos(df_raw)

# -------------------- Filtros --------------------
with st.sidebar:
    st.header("Filtros")
    if PV.empty or "Data" not in PV.columns:
        st.info("Sem dados de Proventos.")
    else:
        dmin = PV["Data"].dropna().min().date()
        dmax = PV["Data"].dropna().max().date()
        default_ini = max(dmin, dmax - timedelta(days=540))  # ~18 meses
        periodo = st.date_input("Período", value=(default_ini, dmax), min_value=dmin, max_value=dmax)
        def uniq(col): 
            return sorted(PV[col].dropna().astype(str).unique().tolist()) if col in PV.columns else []
        classe_sel = st.multiselect("Classe", options=uniq("Classe"), default=uniq("Classe"))
        ticker_sel = st.multiselect("Ticker", options=uniq("Ticker"))
        meta_mensal = st.number_input("🎯 Meta mensal de proventos (R$)", min_value=0.0, value=500.0, step=50.0)

# aplica filtros
if not PV.empty and "Data" in PV.columns:
    d0, d1 = periodo if isinstance(periodo, tuple) else (PV["Data"].min().date(), PV["Data"].max().date())
    PV = PV[PV["Data"].between(pd.to_datetime(d0), pd.to_datetime(d1))]
if "Classe" in PV.columns and classe_sel: PV = PV[PV["Classe"].isin(classe_sel)]
if "Ticker" in PV.columns and ticker_sel: PV = PV[PV["Ticker"].isin(ticker_sel)]

if PV.empty:
    st.info("Nenhum provento no período/seleção atual.")
    st.stop()

# -------------------- KPIs --------------------
total_periodo = PV["Valor"].sum(skipna=True)
hoje = date.today()
inicio_ytd = date(hoje.year, 1, 1)
ytd = PV[PV["Data"].dt.date >= inicio_ytd]["Valor"].sum()
ult12_ini = date(hoje.year - 1, hoje.month, 1)
ult12 = PV[PV["Data"].dt.date >= ult12_ini]["Valor"].sum()

# preencher meses faltantes
def month_range(d0, d1):
    cur = pd.to_datetime(f"{d0.year}-{d0.month}-01")
    end = pd.to_datetime(f"{d1.year}-{d1.month}-01")
    out=[]
    while cur <= end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(2)) - pd.offsets.MonthBegin(1)
    return pd.to_datetime(out)

meses_periodo = month_range(d0, d1)
PVm = PV.dropna(subset=["Data"]).copy()
PVm["Competencia"] = pd.to_datetime(PVm["Data"].dt.strftime("%Y-%m-01"))
grp_mes = PVm.groupby("Competencia", dropna=False)["Valor"].sum().reindex(meses_periodo, fill_value=0).reset_index()
grp_mes.columns = ["Competencia","Valor"]

media_mensal = grp_mes["Valor"].mean()
melhor_mes = grp_mes.loc[grp_mes["Valor"].idxmax()]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total no Período", moeda_br(total_periodo))
c2.metric("YTD", moeda_br(ytd))
c3.metric("Últimos 12 meses", moeda_br(ult12))
c4.metric("Média mensal (período)", moeda_br(media_mensal))

# progresso vs meta (média 3m)
media_3m = grp_mes.tail(3)["Valor"].mean()
pct_meta = 0 if meta_mensal<=0 else min(1.0, media_3m / meta_mensal)
st.progress(pct_meta, text=f"Média 3m: {moeda_br(media_3m)} / Meta: {moeda_br(meta_mensal)}")

# -------------------- Gráfico: barras + acumulado com eixo mensal contínuo --------------------
fig = go.Figure()
fig.add_bar(x=grp_mes["Competencia"], y=grp_mes["Valor"], name="Mensal")
fig.add_trace(go.Scatter(x=grp_mes["Competencia"], y=grp_mes["Valor"].cumsum(),
                         name="Acumulado", mode="lines+markers"))
fig.update_layout(template=PLOTLY_TEMPLATE, title="Proventos por Mês (com acumulado)",
                  xaxis_title="Competência", yaxis_title="R$")
fig.update_xaxes(dtick="M1", tickformat="%b/%Y")  # força todos os meses
st.plotly_chart(fig, use_container_width=True)

# -------------------- Rankings --------------------
colA, colB = st.columns(2)
with colA:
    if "Ticker" in PV.columns:
        top_ticker = PV.groupby("Ticker", dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
        st.subheader("🏆 Tickers que mais pagaram")
        st.dataframe(top_ticker, use_container_width=True, hide_index=True)
with colB:
    if "Classe" in PV.columns:
        por_classe = PV.groupby("Classe", dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
        st.subheader("📦 Proventos por Classe")
        st.dataframe(por_classe, use_container_width=True, hide_index=True)

# -------------------- Sazonalidade --------------------
st.subheader("📆 Sazonalidade")
PVm["Ano"] = PVm["Data"].dt.year
PVm["Mes"] = PVm["Data"].dt.month
tabela_cal = PVm.groupby(["Ano","Mes"], dropna=False)["Valor"].sum().unstack(fill_value=0).sort_index()
tabela_cal.columns = [pd.to_datetime(f"2000-{m}-01").strftime("%b") for m in tabela_cal.columns]
st.dataframe(tabela_cal, use_container_width=True)

# Boxplot por mês (patch: usa assign para nomear colunas antes do groupby)
pv_box = (
    PVm.assign(Mes=PVm["Data"].dt.month, Ano=PVm["Data"].dt.year)
       .groupby(["Mes","Ano"], dropna=False)["Valor"]
       .sum()
       .reset_index()
)
pv_box["MesNome"] = pv_box["Mes"].apply(lambda m: pd.to_datetime(f"2000-{int(m)}-01").strftime("%b"))
fig_box = px.box(pv_box, x="MesNome", y="Valor", template=PLOTLY_TEMPLATE, title="Distribuição por Mês (histórico)")
fig_box.update_layout(yaxis_title="R$")
st.plotly_chart(fig_box, use_container_width=True)

# -------------------- Tabela filtrada + downloads --------------------
st.subheader("Tabela de Proventos (filtrada)")
st.dataframe(PV.sort_values("Data"), use_container_width=True, hide_index=True)

# CSV
csv = PV.sort_values("Data").to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV (filtrado)", data=csv, file_name="proventos_filtrado.csv", mime="text/csv")

# Excel com múltiplas abas
def to_excel_bytes(dfs: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        for name,df in dfs.items():
            df.to_excel(wr, index=False, sheet_name=name[:31])
    buf.seek(0)
    return buf.read()

excel_bytes = to_excel_bytes({
    "Proventos_Filtrados": PV.sort_values("Data"),
    "Mensal": grp_mes,
    "Por_Ticker": PV.groupby("Ticker", dropna=False)["Valor"].sum().reset_index(),
    "Por_Classe": PV.groupby("Classe", dropna=False)["Valor"].sum().reset_index(),
    "Sazonalidade_AnoMes": PVm.groupby(["Ano","Mes"], dropna=False)["Valor"].sum().reset_index(),
})
st.download_button("⬇️ Baixar Excel (abas)", data=excel_bytes,
                   file_name="proventos_analise.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- Rodapé --------------------
modo = st.session_state.get("proventos_modo", "desconhecido")
st.caption(f"Modo de leitura: **{modo}** | Melhor mês do período: "
           f"**{melhor_mes['Competencia'].strftime('%b/%Y')}** = {moeda_br(melhor_mes['Valor'])}")
