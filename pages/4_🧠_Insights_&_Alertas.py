# pages/0_🧠_Insights_&_Alertas.py — feed limpo de Insights & Alertas
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="🧠 Insights & Alertas", page_icon="🧠", layout="wide")
st.title("🧠 Insights & Alertas")

PLOTLY_TEMPLATE = "plotly_dark"

# ------------------ Config / Abas ------------------
SHEET_ID = st.secrets.get("SHEET_ID", "").strip()
ABA_ATIVOS    = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_PROVENTOS = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

# ------------------ Helpers ------------------
def br_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","-","--"}: return None
    s = (s.replace("R$", "").replace("US$", "").replace("$","")
           .replace("%","").replace(" ","")).replace(".","").replace(",",".")
    try: return float(s)
    except: return None

def moeda_br(v):
    try: v = float(v)
    except: v = 0.0
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def _has_sa(): return bool(st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account"))
def _sa_info(): return st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account") or {}

def _find_header_row(values, expect_cols):
    exp = [e.strip().lower() for e in expect_cols]
    best, hits_best = 0, 0
    for i,row in enumerate(values):
        row_low = [str(c).strip().lower() for c in row]
        hits = sum(1 for e in exp if e in row_low)
        if hits > hits_best: best, hits_best = i, hits
        if hits >= 2: return i
    return best

def read_ws(sheet_id, aba):
    import gspread
    from google.oauth2.service_account import Credentials
    info = _sa_info()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(aba)
    except Exception:
        titles = [w.title for w in sh.worksheets()]
        match = next((t for t in titles if t.casefold()==aba.casefold()), None)
        ws = sh.worksheet(match if match else aba)
    vals = ws.get_all_values()
    if not vals: return pd.DataFrame()
    if "provento" in ws.title.lower():
        hdr = _find_header_row(vals, ["Ticker","Tipo Provento","Data"])
    else:
        hdr = _find_header_row(vals, [c for c in vals[0] if str(c).strip()])
    cols = vals[hdr]
    df = pd.DataFrame(vals[hdr+1:], columns=[c.strip() for c in cols])
    df = df.replace({"": None}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df

def to_datetime_br(series): return pd.to_datetime(series, dayfirst=True, errors="coerce")

def padronizar_ativos(df):
    if df.empty: return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Ticker": ["Ticker"],
        "Classe": ["Classe","Classe do Ativo","Tipo"],
        "ValorInvestido": ["Valor Investido","Valor investido"],
        "ValorAtual": ["Valor Atual","Valor atual"],
        "Quantidade": ["Quantidade","Qtd","Quantidade (líquida)","Quantidade (liquida)"],
    }
    out = pd.DataFrame()
    for k,cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[k] = df[col] if col else None
    for c in ["ValorInvestido","ValorAtual","Quantidade"]:
        out[c] = out[c].map(br_to_float)
    return out

def padronizar_proventos(df):
    if df.empty: return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo","Tipo Provento"],
        "Valor": ["Total Líquido R$","Total Liquido R$","Valor","Total"],
        "Classe": ["Classe do Ativo","Classe"],
        "Quantidade": ["Quantidade","Qtd"],
        "Unitario": ["Unitário R$","Unitario R$","Unitário","Unitario"],
    }
    out = pd.DataFrame()
    for k,cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[k] = df[col] if col else None
    out["Data"] = to_datetime_br(out["Data"])
    for c in ["Valor","Quantidade","Unitario"]:
        if c in out.columns:
            out[c] = out[c].map(br_to_float)
    if ("Valor" not in out.columns or out["Valor"].isna().all()) and {"Quantidade","Unitario"}.issubset(out.columns):
        out["Valor"] = out["Quantidade"].fillna(0)*out["Unitario"].fillna(0)
    return out

# ------------------ Carregar ------------------
if not SHEET_ID or not _has_sa():
    st.error("Configure SHEET_ID e a Service Account nos secrets.")
    st.stop()

with st.spinner("Carregando dados..."):
    dfA = padronizar_ativos(read_ws(SHEET_ID, ABA_ATIVOS))
    PV  = padronizar_proventos(read_ws(SHEET_ID, ABA_PROVENTOS))

if PV.empty:
    st.info("Sem proventos para gerar insights.")
    st.stop()

# ------------------ Sidebar: parâmetros ------------------
with st.sidebar:
    st.header("Parâmetros de Alerta")
    meses_lookback = st.slider("Histórico para médias (meses)", 3, 18, 6)
    corte_down = st.slider("⚠️ Alerta se queda mensal > (%)", 5, 80, 20)
    alta_up    = st.slider("🎉 Sinal se alta mensal > (%)", 5, 80, 20)
    meses_sem_pagar = st.slider("⏰ Alerta se ficar sem pagar (meses)", 1, 6, 2)
    topN_conc = st.slider("Top-N p/ concentração", 3, 10, 5)
    alvo_conc = st.slider("Limite de concentração Top-N (%)", 20, 90, 40)
    alvo_yoc  = st.number_input("Meta Yield on Cost (a.a. %)", value=8.0, step=0.5)

# ------------------ Séries mensais ------------------
PVm = PV.dropna(subset=["Data"]).copy()
PVm["Competencia"] = pd.to_datetime(PVm["Data"].dt.strftime("%Y-%m-01"))
m_ticker = PVm.groupby(["Competencia","Ticker"], dropna=False)["Valor"].sum().reset_index()

def expand_months(df, key_col):
    full = []
    for k, sub in df.groupby(key_col):
        idx = pd.period_range(sub["Competencia"].min(), sub["Competencia"].max(), freq="M").to_timestamp()
        s = sub.set_index("Competencia")["Valor"].reindex(idx, fill_value=0.0)\
              .rename("Valor").reset_index().rename(columns={"index":"Competencia"})
        s[key_col] = k
        full.append(s)
    return pd.concat(full, ignore_index=True) if full else df

m_ticker_full = expand_months(m_ticker, "Ticker")
ultima_comp = PVm["Competencia"].max()
janela12 = ultima_comp - pd.DateOffset(months=12)

# ------------------ Geração de sinais ------------------
rows = []

def add_signal(cat, tipo, ticker, score, msg):
    rows.append({
        "categoria": cat,      # "Alerta", "Sinal", "Obs"
        "tipo": tipo,          # queda_mensal, alta_mensal, etc.
        "ticker": ticker,
        "score": score,        # severidade (0..1)
        "mensagem": msg
    })

# 1) MoM e vs média N meses
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub = sub.sort_values("Competencia")
    if len(sub) < 2: 
        continue
    sub["MoM"] = sub["Valor"].pct_change()
    sub["MA"]  = sub["Valor"].rolling(meses_lookback, min_periods=1).mean()
    v_atual = sub["Valor"].iloc[-1]
    mom = sub["MoM"].iloc[-1] or 0
    desv = (v_atual - (sub["MA"].iloc[-1] or 0)) / (sub["MA"].iloc[-1] or 1e-9)

    if mom <= -(corte_down/100):
        add_signal("Alerta", "queda_mensal", tkr, min(1.0, abs(mom)),
                   f"🔻 **{tkr}** caiu **{mom*100:.1f}%** vs mês anterior.")
    if mom >= (alta_up/100):
        add_signal("Sinal", "alta_mensal", tkr, min(1.0, mom),
                   f"📈 **{tkr}** subiu **{mom*100:.1f}%** vs mês anterior.")
    if abs(desv) >= 0.2:
        arrow = "↑" if desv>0 else "↓"
        cat = "Sinal" if desv>0 else "Alerta"
        add_signal(cat, "desvio_media", tkr, min(1.0, abs(desv)),
                   f"{arrow} **{tkr}** ficou **{desv*100:.1f}%** {'acima' if desv>0 else 'abaixo'} da média {meses_lookback}m.")

# 2) Meses sem pagar
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub = sub.sort_values("Competencia")
    ult_pag = sub.loc[sub["Valor"]>0, "Competencia"].max()
    if pd.isna(ult_pag): 
        continue
    gap = (ultima_comp.to_period("M") - ult_pag.to_period("M")).n
    if gap >= meses_sem_pagar:
        add_signal("Alerta", "sem_pagar", tkr, min(1.0, gap/6),
                   f"⏰ **{tkr}** está há **{gap}** mês(es) sem pagar.")

# 3) Máx/Mín 12m
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub12 = sub[sub["Competencia"]>=janela12]
    if sub12.empty: continue
    v = sub12["Valor"].iloc[-1]
    if v>0 and v == sub12["Valor"].max():
        add_signal("Sinal","novo_max",tkr,0.6,f"🏆 **{tkr}** marcou **máximo de 12m** no provento mensal.")
    if v>0 and v == sub12["Valor"].min():
        add_signal("Obs","novo_min",tkr,0.4,f"⚠️ **{tkr}** marcou **mínimo de 12m** no provento mensal.")

# 4) Concentração da carteira
if not dfA.empty and {"Ticker","ValorAtual"}.issubset(dfA.columns):
    aloc = dfA.groupby("Ticker", dropna=False)["ValorAtual"].sum().sort_values(ascending=False)
    top = aloc.head(topN_conc)
    total = aloc.sum() or 1e-9
    perc = 100*top.sum()/total
    if perc > alvo_conc:
        add_signal("Alerta","concentracao",",".join(top.index[:3]),min(1.0, perc/100),
                   f"🎯 **Concentração**: Top {topN_conc} = **{perc:.1f}%** (limite {alvo_conc}%).")

# 5) Yield 12m / Yield on Cost
if not dfA.empty:
    pv12 = PVm[PVm["Competencia"]>=janela12].groupby("Ticker")["Valor"].sum()
    if "ValorInvestido" in dfA.columns:
        yoc = (pv12 / (dfA.set_index("Ticker")["ValorInvestido"]+1e-9)*100).dropna()
        for tkr, yy in yoc.items():
            if yy >= st.session_state.get("yoc_meta", 8.0):
                add_signal("Sinal","yoc_ok",tkr,min(1.0, yy/20),f"💵 **{tkr}** Yield-on-Cost 12m = **{yy:.1f}% a.a.**.")
    if "ValorAtual" in dfA.columns:
        ya = (pv12 / (dfA.set_index("Ticker")["ValorAtual"]+1e-9)*100).dropna().sort_values(ascending=False).head(5)
        for tkr, yy in ya.items():
            add_signal("Obs","yield_atual",tkr,0.5,f"📊 **{tkr}** Yield 12m / ValorAtual = **{yy:.1f}% a.a.**")

feed = pd.DataFrame(rows)
if feed.empty:
    st.success("Nenhum insight no momento. 👍")
    st.stop()

# ------------------ Controles de visual ------------------
with st.container():
    col1, col2, col3 = st.columns(3)
    col1.metric("Alertas", int((feed["categoria"]=="Alerta").sum()))
    col2.metric("Sinais",  int((feed["categoria"]=="Sinal").sum()))
    col3.metric("Observações", int((feed["categoria"]=="Obs").sum()))

st.divider()
st.subheader("Feed")

colf1, colf2, colf3 = st.columns([1,1,1.2])
sev_min = colf1.slider("Severidade mínima", 0.0, 1.0, 0.2, 0.05)
cats = colf2.multiselect("Categorias", ["Alerta","Sinal","Obs"], default=["Alerta","Sinal","Obs"])
busca = colf3.text_input("Buscar ticker (ex.: HGLG11)", "").strip().upper()

# aplica filtros
view = feed.copy()
view = view[view["score"] >= sev_min]
view = view[view["categoria"].isin(cats)]
if busca:
    view = view[view["ticker"].str.contains(busca)]

# ordena por severidade (desc) e tipo
ordem_tipo = {"queda_mensal":0,"desvio_media":1,"sem_pagar":2,"concentracao":3,"alta_mensal":4,"novo_min":5,"novo_max":6,"yoc_ok":7,"yield_atual":8}
view = view.sort_values(["categoria", view["tipo"].map(ordem_tipo).fillna(99), "score"], ascending=[True, True, False])

# abas por categoria
tab_alerta, tab_sinal, tab_obs = st.tabs(["⚠️ Alertas", "🎉 Sinais", "ℹ️ Observações"])

def render_feed(df, empty_msg):
    if df.empty:
        st.caption(empty_msg)
        return
    # agrupa por ticker e mostra cards
    for tkr, sub in df.groupby("ticker"):
        sub = sub.sort_values("score", ascending=False)
        with st.expander(f"{tkr} — {len(sub)} item(ns)", expanded=False):
            for _, r in sub.iterrows():
                st.markdown(f"- {r['mensagem']}")

with tab_alerta:
    render_feed(view[view["categoria"]=="Alerta"], "Sem alertas nos filtros atuais.")
with tab_sinal:
    render_feed(view[view["categoria"]=="Sinal"], "Sem sinais positivos nos filtros atuais.")
with tab_obs:
    render_feed(view[view["categoria"]=="Obs"], "Sem observações nos filtros atuais.")

st.download_button(
    "⬇️ Exportar feed filtrado (CSV)",
    view.to_csv(index=False).encode("utf-8-sig"),
    "insights_filtrados.csv",
    "text/csv"
)

st.divider()
st.subheader("Resumos rápidos")

# Top quedas / altas (mês atual vs anterior)
m_full = m_ticker_full.sort_values("Competencia")
ult = m_full.groupby("Ticker").tail(2).groupby("Ticker")["Valor"].apply(lambda s: s.iloc[-1] - s.iloc[0]).rename("Delta")
mom_df = m_full.groupby("Ticker")["Valor"].apply(lambda s: s.pct_change().iloc[-1] if len(s)>1 else np.nan).rename("MoM")
mom_tab = pd.concat([ult, mom_df], axis=1).dropna().reset_index()
mom_tab["MoM%"] = (mom_tab["MoM"]*100).round(1)

colA, colB, colC = st.columns(3)
with colA:
    st.caption("🔻 Maiores quedas (MoM)")
    st.dataframe(mom_tab.sort_values("MoM").head(10)[["Ticker","MoM%"]], hide_index=True, use_container_width=True)
with colB:
    st.caption("📈 Maiores altas (MoM)")
    st.dataframe(mom_tab.sort_values("MoM", ascending=False).head(10)[["Ticker","MoM%"]], hide_index=True, use_container_width=True)
with colC:
    st.caption("⏰ Sem pagar (gap ≥ parâmetro)")
    gaps = []
    for tkr, sub in m_ticker_full.groupby("Ticker"):
        last_pay = sub.loc[sub["Valor"]>0, "Competencia"].max()
        if pd.isna(last_pay): continue
        gap = (ultima_comp.to_period("M") - last_pay.to_period("M")).n
        if gap >= meses_sem_pagar:
            gaps.append({"Ticker": tkr, "Meses": gap})
    st.dataframe(pd.DataFrame(gaps).sort_values("Meses", ascending=False), hide_index=True, use_container_width=True)
