# pages/1_Resumo_Investimentos.py — ligado ao seu SHEET_ID e GID de Movimentações
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="💼 Resumo de Investimentos", page_icon="💼", layout="wide")
TEMPLATE = "plotly_dark"
st.title("💼 Resumo de Investimentos")

# =========================
# CONFIG DA SUA PLANILHA
# =========================
SHEET_ID = "1p9IzDr-5ZV0phUHfNA_9d5xNvZW1IRo84LA__JyiiQc"

# Já conectado: MOVIMENTAÇÕES/LANÇAMENTOS
GID_MOV  = "2109089485"

# Falta você informar (cole os gid das abas correspondentes):
GID_POS  = ""   # EX.: "987654321"  (Posição/Consolidado)
GID_PROV = ""   # EX.: "456789123"  (Proventos)

def csv_url(gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

def moeda(v) -> str:
    try: v = float(v)
    except Exception: v = 0.0
    return f"R$ {v:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

def to_float_br(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace("R$","").replace(" ","").replace(".","").replace(",",".")
    try: return float(s)
    except: return np.nan

# =========================
# CARREGAMENTO
# =========================
@st.cache_data(ttl=300)
def carregar_mov():
    if not (SHEET_ID and GID_MOV):
        return pd.DataFrame()
    df = pd.read_csv(csv_url(GID_MOV))
    df.columns = [c.strip().lower() for c in df.columns]
    # DATA
    for c in ["data","data operação","data_op","date"]:
        if c in df.columns:
            df["data"] = pd.to_datetime(df[c], errors="coerce"); break
    if "data" not in df: df["data"] = pd.NaT
    # TOTAL
    if "total" in df.columns:
        df["total"] = df["total"].apply(to_float_br)
    else:
        preco = None
        for c in ["preco","preço","preco unit.","preco_unit","preço unit."]:
            if c in df.columns: preco = df[c].apply(to_float_br); break
        qtd = None
        for c in ["quantidade","qtd","qte","qde"]:
            if c in df.columns: qtd = pd.to_numeric(df[c], errors="coerce"); break
        df["total"] = preco * qtd if (preco is not None and qtd is not None) else 0.0
    # TIPO
    if "tipo" not in df.columns:
        for c in ["movimento","operation","operacao","operação"]:
            if c in df.columns: df["tipo"] = df[c]; break
    if "tipo" not in df.columns: df["tipo"] = ""
    df["tipo_norm"] = df["tipo"].astype(str).str.lower().str.strip()
    df["mes"] = pd.to_datetime(df["data"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_data(ttl=300)
def carregar_pos():
    if not (SHEET_ID and GID_POS): return pd.DataFrame()
    df = pd.read_csv(csv_url(GID_POS))
    df.columns = [c.strip().lower() for c in df.columns]
    if "valor_atual" not in df.columns:
        qcol = next((c for c in ["quantidade","qtd"] if c in df.columns), None)
        pcol = next((c for c in ["preco_atual","preço_atual","ultimo_preco","preco","preço"] if c in df.columns), None)
        if qcol and pcol:
            df["valor_atual"] = pd.to_numeric(df[qcol], errors="coerce") * df[pcol].apply(to_float_br)
        else:
            df["valor_atual"] = 0.0
    return df

@st.cache_data(ttl=300)
def carregar_prov():
    if not (SHEET_ID and GID_PROV): return pd.DataFrame()
    df = pd.read_csv(csv_url(GID_PROV))
    df.columns = [c.strip().lower() for c in df.columns]
    if "data_com" not in df.columns:
        for c in ["data com","com_data"]:
            if c in df.columns: df["data_com"] = pd.to_datetime(df[c], errors="coerce"); break
    else:
        df["data_com"] = pd.to_datetime(df["data_com"], errors="coerce")
    if "pagamento" in df.columns:
        df["pagamento"] = pd.to_datetime(df["pagamento"], errors="coerce")
    if "valor_cota" in df.columns:
        df["valor_cota"] = df["valor_cota"].apply(to_float_br)
    if "qtde" in df.columns:
        df["qtde"] = pd.to_numeric(df["qtde"], errors="coerce")
    if "total" not in df.columns and {"valor_cota","qtde"} <= set(df.columns):
        df["total"] = df["valor_cota"] * df["qtde"]
    return df

df_mov  = carregar_mov()
df_pos  = carregar_pos()
df_prov = carregar_prov()

# Badges de conexão
conexoes = []
conexoes.append("Movimentações ✅" if not df_mov.empty else "Movimentações ❌")
conexoes.append("Posição ✅"        if not df_pos.empty else "Posição ❌ (adicione GID_POS)")
conexoes.append("Proventos ✅"      if not df_prov.empty else "Proventos ❌ (adicione GID_PROV)")
st.info(" | ".join(conexoes))

# =========================
# LÓGICA FINANCEIRA
# =========================
DEPOSITOS = {"aporte","depósito","deposito","transferência recebida","transferencia recebida"}
RETIRADAS = {"retirada","saque","transferência enviada","transferencia enviada"}
COMPRAS   = {"compra","compra - fracionário","compra - mercado"}
VENDAS    = {"venda","venda - fracionário","venda - mercado"}

st.sidebar.header("Filtros")
metodo_aporte = st.sidebar.radio(
    "Como calcular o Aportado (bolso)?",
    options=["Depósito - Retirada", "Compra - Venda"],
    index=0
)

mov = df_mov.copy()
if mov.empty:
    st.warning("Não encontrei dados na aba de Movimentações (GID_MOV). Verifique se o GID está correto.")
    st.stop()

if metodo_aporte == "Depósito - Retirada":
    mov["aporte_sinal"] = np.select(
        [mov["tipo_norm"].isin(DEPOSITOS), mov["tipo_norm"].isin(RETIRADAS)],
        [mov["total"].fillna(0.0), -mov["total"].fillna(0.0)],
        default=0.0
    )
else:
    mov["aporte_sinal"] = np.select(
        [mov["tipo_norm"].isin(COMPRAS), mov["tipo_norm"].isin(VENDAS)],
        [mov["total"].abs().fillna(0.0), -mov["total"].abs().fillna(0.0)],
        default=0.0
    )

mov["mes"] = pd.to_datetime(mov["data"], errors="coerce").dt.to_period("M").dt.to_timestamp()
aportes_m = mov.groupby("mes", as_index=False)["aporte_sinal"].sum()
aportes_m["aportado_acum"] = aportes_m["aporte_sinal"].clip(lower=0).cumsum()

aportado_total = float(aportes_m["aporte_sinal"].sum())
patrimonio_atual = float(pd.to_numeric(df_pos.get("valor_atual", 0), errors="coerce").sum()) if not df_pos.empty else None
variacao = (patrimonio_atual - aportado_total) if patrimonio_atual is not None else None
perc_var = (variacao / aportado_total * 100) if (variacao is not None and aportado_total) else None

if not df_prov.empty and {"pagamento","total"} <= set(df_prov.columns):
    prov12 = df_prov.set_index("pagamento").sort_index()
    prov12 = prov12.last("365D") if len(prov12) else prov12
    media12 = float(prov12["total"].sum()) / 12 if len(prov12) else 0.0
    total_prov = float(df_prov["total"].sum())
else:
    media12, total_prov = None, None

# FILTRO DE ANO
anos = sorted(aportes_m["mes"].dt.year.unique())
ano_sel = st.sidebar.multiselect("Ano", options=anos, default=anos[-1:] if anos else None)
aportes_plot = aportes_m[aportes_m["mes"].dt.year.isin(ano_sel)] if ano_sel else aportes_m.copy()
prov_plot = df_prov[df_prov.get("pagamento", pd.NaT).dt.year.isin(ano_sel)] if (not df_prov.empty and "pagamento" in df_prov) else pd.DataFrame()

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.caption("Variação Patrimonial")
    if variacao is None:
        st.metric(" ", "—", "adicione GID_POS")
    else:
        delta = f"{(perc_var or 0):.2f}%".replace(".", ",")
        st.metric(" ", moeda(variacao), delta)
with c2:
    st.caption("Aportado (bolso)")
    st.metric(" ", moeda(aportado_total))
with c3:
    st.caption("Patrimônio atual")
    st.metric(" ", "—" if patrimonio_atual is None else moeda(patrimonio_atual))
with c4:
    st.caption("Proventos (média 12m)")
    if media12 is None:
        st.metric(" ", "—", "adicione GID_PROV")
    else:
        st.metric(" ", moeda(media12), f"Total {moeda(total_prov)}")

st.markdown("---")

# =========================
# Evolução x Calendário
# =========================
a, b = st.columns([2, 1])

with a:
    st.subheader("Evolução Patrimonial")
    fig = px.bar(aportes_plot, x="mes", y="aportado_acum",
                 labels={"mes": "", "aportado_acum": "Aportado (Acumulado)"},
                 template=TEMPLATE)
    fig.add_scatter(x=aportes_plot["mes"], y=aportes_plot["aportado_acum"],
                    mode="lines+markers", name="Evolução")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with b:
    st.subheader("Calendário de eventos")
    if not prov_plot.empty and {"pagamento","ticker"}.issubset(prov_plot.columns):
        hoje = pd.to_datetime(date.today())
        prox = prov_plot[pd.to_datetime(prov_plot["pagamento"], errors="coerce") >= hoje].copy()
        prox = prox.sort_values("pagamento").head(8)
        if len(prox)==0: st.info("Sem pagamentos futuros nos filtros.")
        for _, r in prox.iterrows():
            total = r["total"] if "total" in r else (float(r.get("valor_cota",0))*float(r.get("qtde",0)))
            st.write(f"**{r.get('ticker','-')}** — {moeda(total)}")
            dcom = r.get("data_com", pd.NaT)
            dcom_txt = pd.to_datetime(dcom).date() if pd.notna(dcom) else "-"
            dpg  = pd.to_datetime(r.get("pagamento", pd.NaT))
            dpg_txt = dpg.date() if pd.notna(dpg) else "-"
            st.caption(f"data com: {dcom_txt} | pagamento: {dpg_txt}")
            st.divider()
    else:
        st.info("Proventos indisponíveis (adicione GID_PROV).")
