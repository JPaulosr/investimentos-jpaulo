# pages/0_🧠_Insights_&_Alertas.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="🧠 Insights & Alertas", page_icon="🧠", layout="wide")
st.title("🧠 Insights & Alertas")

PLOTLY_TEMPLATE = "plotly_dark"

# ------------------ Secrets / Abas ------------------
SHEET_ID = st.secrets.get("SHEET_ID", "").strip()
ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

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
    # heurística: proventos vs ativos
    if "provento" in ws.title.lower():
        hdr = _find_header_row(vals, ["Ticker","Tipo Provento","Data"])
    elif "meus ativos" in ws.title.lower() or "ativo" in ws.title.lower():
        hdr = _find_header_row(vals, ["Ticker","Valor Investido","Valor Atual"])
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
        "PrecoMedioCompra": ["Preço (compra R$)","Preco Medio (compra R$)","Preço Médio (compra R$)","Preco Medio Compra R$","Preço Médio"],
        "Quantidade": ["Quantidade","Qtd","Quantidade (líquida)","Quantidade (liquida)"],
    }
    out = pd.DataFrame()
    for k,cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[k] = df[col] if col else None
    for c in ["ValorInvestido","ValorAtual","PrecoMedioCompra","Quantidade"]:
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
    # se não vier Valor, calcula Quantidade*Unitario
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

# ------------------ Controles ------------------
with st.sidebar:
    st.header("Parâmetros de Alerta")
    meses_lookback = st.slider("Histórico para médias (meses)", 3, 18, 6)
    corte_down = st.slider("⚠️ Alerta se queda mensal > (%)", 5, 80, 20)
    alta_up    = st.slider("🎉 Sinal se alta mensal > (%)", 5, 80, 20)
    meses_sem_pagar = st.slider("⏰ Alerta se ficar sem pagar (meses)", 1, 6, 2)
    topN_conc = st.slider("Top-N p/ concentração", 3, 10, 5)
    alvo_conc = st.slider("Limite de concentração Top-N (%)", 20, 90, 40)
    alvo_yoc  = st.number_input("Meta Yield on Cost (a.a. %)", value=8.0, step=0.5)

# ------------------ Pré-processo mensal ------------------
if PV.empty:
    st.info("Sem proventos para gerar insights.")
    st.stop()

PVm = PV.dropna(subset=["Data"]).copy()
PVm["Competencia"] = pd.to_datetime(PVm["Data"].dt.strftime("%Y-%m-01"))
# total por mês/ticker
m_ticker = PVm.groupby(["Competencia","Ticker"], dropna=False)["Valor"].sum().reset_index()

# gerar série completa por mês para cada ticker (preencher faltas com 0)
def expand_months(df, key_col):
    full = []
    for k, sub in df.groupby(key_col):
        idx = pd.period_range(sub["Competencia"].min(), sub["Competencia"].max(), freq="M").to_timestamp()
        s = sub.set_index("Competencia")["Valor"].reindex(idx, fill_value=0.0).rename("Valor").reset_index().rename(columns={"index":"Competencia"})
        s[key_col] = k
        full.append(s)
    return pd.concat(full, ignore_index=True) if full else df

m_ticker_full = expand_months(m_ticker, "Ticker")

# ------------------ Regras de insights ------------------
insights = []

# 1) MoM e vs média últimos N meses
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub = sub.sort_values("Competencia")
    if len(sub) < 2: 
        continue
    sub["MoM"] = sub["Valor"].pct_change()
    # média dos últimos N meses para cada ponto (rolling)
    sub["MA"] = sub["Valor"].rolling(meses_lookback, min_periods=1).mean()
    if sub["Valor"].iloc[-1] > 0:
        mom = sub["MoM"].iloc[-1]
        desv = (sub["Valor"].iloc[-1] - sub["MA"].iloc[-1]) / (sub["MA"].iloc[-1] or 1e-9)
        if mom <= -(corte_down/100):
            insights.append({
                "tipo":"queda_mensal","ticker":tkr,
                "msg": f"🔻 **{tkr}** caiu **{mom*100:.1f}%** vs mês anterior."
            })
        if mom >= (alta_up/100):
            insights.append({
                "tipo":"alta_mensal","ticker":tkr,
                "msg": f"📈 **{tkr}** subiu **{mom*100:.1f}%** vs mês anterior."
            })
        if abs(desv) >= 0.2:  # desvio relevante vs média N meses
            arrow = "↑" if desv>0 else "↓"
            insights.append({
                "tipo":"desvio_media","ticker":tkr,
                "msg": f"{arrow} **{tkr}** ficou **{desv*100:.1f}%** {'acima' if desv>0 else 'abaixo'} da média {meses_lookback}m."
            })

# 2) Meses sem pagar (frequência)
ultima_comp = PVm["Competencia"].max()
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub = sub.sort_values("Competencia")
    # último mês com pagamento
    ult_pag = sub.loc[sub["Valor"]>0, "Competencia"].max()
    if pd.isna(ult_pag):
        continue
    meses_gap = (ultima_comp.to_period("M") - ult_pag.to_period("M")).n
    if meses_gap >= meses_sem_pagar:
        insights.append({
            "tipo":"sem_pagar","ticker":tkr,
            "msg": f"⏰ **{tkr}** está há **{meses_gap}** mês(es) sem pagar."
        })

# 3) Novos Máx/Mín (12m)
janela = PVm["Competencia"].max() - pd.DateOffset(months=12)
for tkr, sub in m_ticker_full.groupby("Ticker"):
    sub12 = sub[sub["Competencia"]>=janela]
    if sub12.empty: 
        continue
    v = sub12["Valor"].iloc[-1]
    if v == sub12["Valor"].max() and v>0:
        insights.append({"tipo":"novo_max","ticker":tkr, "msg": f"🏆 **{tkr}** marcou **máximo de 12m** no provento mensal."})
    if v == sub12["Valor"].min() and v>0:
        insights.append({"tipo":"novo_min","ticker":tkr, "msg": f"⚠️ **{tkr}** marcou **mínimo de 12m** no provento mensal."})

# 4) Concentração da carteira (top N)
if not dfA.empty and {"Ticker","ValorAtual"}.issubset(dfA.columns):
    aloc = dfA.groupby("Ticker", dropna=False)["ValorAtual"].sum().sort_values(ascending=False)
    top = aloc.head(topN_conc)
    total = aloc.sum() or 1e-9
    perc = 100*top.sum()/total
    if perc > alvo_conc:
        insights.append({
            "tipo":"concentracao","ticker":",".join(top.index[:3]),
            "msg": f"🎯 **Concentração**: Top {topN_conc} = **{perc:.1f}%** (limite {alvo_conc}%)."
        })

# 5) Yield 12m e Yield on Cost
if not dfA.empty:
    pv12 = PVm[PVm["Competencia"]>=janela].groupby("Ticker")["Valor"].sum()
    # Yield on Cost = proventos 12m / Valor investido
    if "ValorInvestido" in dfA.columns:
        yoc = (pv12 / (dfA.set_index("Ticker")["ValorInvestido"]+1e-9)*100).dropna()
        yoc = yoc[yoc>0].sort_values(ascending=False).head(10)
        for tkr, yy in yoc.items():
            if yy >= alvo_yoc:
                insights.append({"tipo":"yoc_ok","ticker":tkr, "msg": f"💵 **{tkr}** Yield-on-Cost 12m = **{yy:.1f}% a.a.** (≥ {alvo_yoc}%)."})
    # Yield atual = proventos 12m / Valor atual
    if "ValorAtual" in dfA.columns:
        ya = (pv12 / (dfA.set_index("Ticker")["ValorAtual"]+1e-9)*100).dropna()
        topya = ya.sort_values(ascending=False).head(5)
        for tkr, yy in topya.items():
            insights.append({"tipo":"yield_atual","ticker":tkr, "msg": f"📊 **{tkr}** Yield 12m / ValorAtual = **{yy:.1f}% a.a.**"})

# ------------------ Saída / UI ------------------
st.subheader("Feed de Insights")
if not insights:
    st.success("Nenhum alerta no momento. 👍")
else:
    # ranking por severidade simples: quedas > desvios > sem pagar > concentração > demais
    ordem = {"queda_mensal":0,"desvio_media":1,"sem_pagar":2,"concentracao":3,"alta_mensal":4,"novo_min":5,"novo_max":6,"yoc_ok":7,"yield_atual":8}
    insights = sorted(insights, key=lambda x: ordem.get(x["tipo"], 99))
    for it in insights:
        st.markdown(f"- {it['msg']}")

    # download
    df_ins = pd.DataFrame(insights)
    st.download_button("⬇️ Baixar insights (CSV)", df_ins.to_csv(index=False).encode("utf-8-sig"),
                       "insights.csv", "text/csv")

# ------------------ Visual rápido (opcional) ------------------
st.subheader("Proventos mensais (todos os tickers)")
m_total = PVm.groupby("Competencia")["Valor"].sum().reset_index()
fig = px.bar(m_total, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos Totais por Mês")
fig.update_xaxes(dtick="M1", tickformat="%b/%Y")
fig.update_layout(yaxis_title="R$")
st.plotly_chart(fig, use_container_width=True)
