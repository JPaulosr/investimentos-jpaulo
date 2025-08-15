# pages/1_Resumo_Investimentos.py
# -------------------------------------------------------
# Dashboard de Investimentos — estilo "md" (dark)
# - KPIs no topo
# - Evolução patrimonial (barras + linha)
# - Calendário de proventos (próximos pagamentos)
# - Composição da carteira (rosca)
# - Objetivos (anel)
# - Rendimento mensal (linha)
#
# Funciona com dados de exemplo (mock). Para ler do Google Sheets:
# 1) Preencha SHEET_ID e os GIDs das abas abaixo (ou use Service Account)
# 2) Ajuste os mapeamentos de colunas (se necessário)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, datetime, timedelta

# =========================
# CONFIG GERAL
# =========================
st.set_page_config(page_title="💼 Resumo de Investimentos", page_icon="💼", layout="wide")
PLOTLY_TEMPLATE = "plotly_dark"
st.title("💼 Resumo de Investimentos")

# =========================
# CONFIG PLANILHA (OPCIONAL)
# =========================
SHEET_ID = ""  # ex: "1p9IzDr-5ZV0phUHfNA_9d5xNvZW1IRo84LA__JyiiQc"
GID_MOV      = ""  # Lançamentos/Movimentações (compras, vendas, aportes)
GID_POS      = ""  # Posição/Consolidado (quantidade, preço médio, valor atual)
GID_PROV     = ""  # Proventos (anúncio, ex, pagamento, valor/cota, qtde)

# Se usar Service Account, defina st.secrets["gcp_service_account"] no Streamlit.
USE_SERVICE_ACCOUNT = False  # True se quiser gspread (conta de serviço)

# =========================
# HELPERS
# =========================
def csv_url(gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

def moeda(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return f"R$ {v:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

def pct(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return f"{v:.2f}%".replace(".", ",")

@st.cache_data(ttl=300)
def carregar_mock():
    """Gera dados de exemplo para rodar já."""
    hoje = date.today()
    datas = pd.date_range(hoje - pd.DateOffset(months=17), periods=18, freq="MS")
    # Aportes e evolução
    aportes = np.random.randint(800, 2500, size=len(datas)).astype(float)
    valor_investido = np.cumsum(aportes)
    evol = (valor_investido * (1 + np.random.normal(0.004, 0.012, len(datas)))).cummax()
    df_evol = pd.DataFrame({"mes": datas, "aporte": aportes, "valor_investido": valor_investido, "evolucao": evol})

    # Composição
    comp = pd.DataFrame({
        "classe": ["FII", "Ação", "FIAGRO"],
        "pct": [0.53, 0.38, 0.09]
    })

    # Proventos (calendário)
    prov_datas = [hoje + timedelta(days=d) for d in (0, 3, 7, 11, 20)]
    df_prov = pd.DataFrame({
        "ticker": ["BBAS3", "CXSE3", "HGLG11", "MXRF11", "ITSA4"],
        "valor_cota": [0.35, 0.31, 1.10, 0.12, 0.40],
        "qtde": [100, 50, 20, 300, 200],
        "pagamento": prov_datas,
        "data_com": [d - timedelta(days=10) for d in prov_datas],
    })
    df_prov["total"] = df_prov["valor_cota"] * df_prov["qtde"]

    # Rendimento mensal (%)
    df_rend = df_evol.copy()
    df_rend["ret_mes"] = (df_rend["evolucao"].pct_change().fillna(0) * 100).clip(-2, 2)

    return df_evol, comp, df_prov, df_rend

# =========================
# CARREGAMENTO (Sheets ou Mock)
# =========================
def carregar_dados():
    if SHEET_ID and GID_MOV and GID_POS and GID_PROV:
        # Caminho rápido via CSV export (evita gspread)
        try:
            mov = pd.read_csv(csv_url(GID_MOV))
            pos = pd.read_csv(csv_url(GID_POS))
            prov = pd.read_csv(csv_url(GID_PROV))
            # ---- NORMALIZAÇÕES BÁSICAS (adapte se precisar) ----
            # Esperado para 'mov': Data, Tipo (Compra/Venda/Aporte), Valor
            mov.columns = [c.strip().lower() for c in mov.columns]
            if "data" in mov.columns:
                mov["data"] = pd.to_datetime(mov["data"], errors="coerce")
            if "valor" in mov.columns:
                mov["valor"] = (
                    mov["valor"].astype(str)
                    .str.replace("R$", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float, errors="ignore")
                )

            # Posição: ticker, classe, valor_atual
            pos.columns = [c.strip().lower() for c in pos.columns]

            # Proventos: ticker, valor_cota, qtde, pagamento
            prov.columns = [c.strip().lower() for c in prov.columns]
            for col in ["data com", "data_com", "com_data"]:
                if col in prov.columns:
                    prov["data_com"] = pd.to_datetime(prov[col], errors="coerce")
                    break
            for col in ["pagamento", "data_pagamento", "payment"]:
                if col in prov.columns:
                    prov["pagamento"] = pd.to_datetime(prov[col], errors="coerce")
                    break
            if "valor_cota" in prov.columns:
                prov["valor_cota"] = (
                    prov["valor_cota"].astype(str)
                    .str.replace("R$", "", regex=False)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float, errors="ignore")
                )
            if "qtde" in prov.columns:
                prov["qtde"] = pd.to_numeric(prov["qtde"], errors="coerce")
            if "total" not in prov.columns and {"valor_cota", "qtde"} <= set(prov.columns):
                prov["total"] = prov["valor_cota"] * prov["qtde"]

            # ---- Construção dos frames do dashboard ----
            # Evolução (ex.: acumulado de aportes + variação simplificada)
            if "data" in mov.columns and "valor" in mov.columns:
                mov_m = mov.groupby(pd.Grouper(key="data", freq="MS"), as_index=False)["valor"].sum()
                mov_m["valor_investido"] = mov_m["valor"].clip(lower=0).cumsum()
                # Linha de evolução (placeholder: investido com ruído)
                mov_m["evolucao"] = (mov_m["valor_investido"] * 1.0).rolling(3, min_periods=1).mean()
                df_evol = mov_m.rename(columns={"data": "mes", "valor": "aporte"})
            else:
                df_evol, comp, df_prov, df_rend = carregar_mock()
                return df_evol, comp, df_prov, df_rend

            # Composição
            if {"classe"}.issubset(pos.columns):
                comp = pos.groupby("classe", as_index=False).agg(valor=("valor_atual", "sum") if "valor_atual" in pos.columns else ("preco_atual", "sum"))
                comp["pct"] = comp["valor"] / comp["valor"].sum()
                comp = comp[["classe", "pct"]]
            else:
                comp = pd.DataFrame({"classe": ["FII", "Ação", "FIAGRO"], "pct": [0.5, 0.4, 0.1]})

            # Proventos
            df_prov = prov[["ticker", "valor_cota", "qtde", "total", "data_com", "pagamento"]].copy()

            # Rendimento mensal (%)
            df_rend = df_evol[["mes", "evolucao"]].copy()
            df_rend["ret_mes"] = df_rend["evolucao"].pct_change().fillna(0) * 100

            return df_evol, comp, df_prov, df_rend

        except Exception as e:
            st.warning(f"Falha ao ler Sheets ({e}). Usando dados de exemplo.")
            return carregar_mock()
    else:
        return carregar_mock()

df_evol, df_comp, df_prov, df_rend = carregar_dados()

# =========================
# FILTROS (Sidebar)
# =========================
st.sidebar.header("Filtros")
anos = sorted(df_evol["mes"].dt.year.unique())
ano_sel = st.sidebar.multiselect("Ano", options=anos, default=anos[-1:] if anos else None)
if ano_sel:
    df_evol = df_evol[df_evol["mes"].dt.year.isin(ano_sel)]
    df_rend = df_rend[df_rend["mes"].dt.year.isin(ano_sel)]
    df_prov = df_prov[df_prov["pagamento"].dt.year.isin(ano_sel)] if "pagamento" in df_prov.columns else df_prov

# =========================
# KPIs SUPERIORES
# =========================
col1, col2, col3, col4 = st.columns(4)
valor_investido_total = float(df_evol["valor_investido"].iloc[-1]) if "valor_investido" in df_evol.columns else float(df_evol["aporte"].clip(lower=0).cumsum().iloc[-1])
evol_total = float(df_evol["evolucao"].iloc[-1])
variacao = evol_total - valor_investido_total
perc_variacao = (variacao / max(valor_investido_total, 1)) * 100

media_12m_prov = (
    df_prov.set_index("pagamento").sort_index()["total"].last("365D").sum() / 12
    if {"pagamento", "total"} <= set(df_prov.columns) and len(df_prov) else 0.0
)
prov_total = df_prov["total"].sum() if "total" in df_prov.columns else 0.0

with col1:
    st.metric("Variação Patrimonial", moeda(variacao), pct(perc_variacao))
with col2:
    st.metric("Capital (investido)", moeda(valor_investido_total))
with col3:
    st.metric("Variação Total vs investido", moeda(variacao), None)
with col4:
    st.metric("Proventos (média 12m)", moeda(media_12m_prov), f"Total {moeda(prov_total)}")

st.markdown("---")

# =========================
# LINHA 1: Evolução x Calendário
# =========================
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Evolução Patrimonial")
    fig = px.bar(
        df_evol,
        x="mes", y="valor_investido",
        labels={"mes": "", "valor_investido": "Valor Investido (Acumulado)"},
        title=None, template=PLOTLY_TEMPLATE
    )
    fig.add_scatter(x=df_evol["mes"], y=df_evol["evolucao"], mode="lines+markers", name="Evolução")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Calendário de eventos")
    hoje = pd.to_datetime(date.today())
    if {"pagamento", "ticker", "total"} <= set(df_prov.columns):
        prox = df_prov[df_prov["pagamento"] >= hoje].sort_values("pagamento").head(6)
        for _, r in prox.iterrows():
            st.write(f"**{r['ticker']}** — {moeda(r['total'])}")
            st.caption(f"data com: {r.get('data_com', pd.NaT).date() if pd.notna(r.get('data_com', pd.NaT)) else '-'} | pagamento: {r['pagamento'].date()}")
            st.divider()
        if prox.empty:
            st.info("Sem pagamentos futuros nos filtros.")
    else:
        st.info("Adapte mapeamento de colunas de proventos para ver o calendário.")

# =========================
# LINHA 2: Composição x Objetivos x Rendimento
# =========================
c3, c4, c5 = st.columns([1, 1, 1.3])

with c3:
    st.subheader("Composição da Carteira")
    if not df_comp.empty:
        figc = px.pie(df_comp, names="classe", values="pct", hole=0.55, template=PLOTLY_TEMPLATE)
        figc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Sem dados de composição.")

with c4:
    st.subheader("Meus Objetivos")
    # Objetivo simples: atingir 30k acima do investido
    objetivo = valor_investido_total * 1.3
    progresso = min(evol_total / max(objetivo, 1), 1.0)
    figo = px.pie(values=[progresso, 1 - progresso], names=["Alcançado", "Falta"], hole=0.8, template=PLOTLY_TEMPLATE)
    figo.update_traces(textinfo="none")
    figo.update_layout(
        annotations=[dict(text=f"{moeda(evol_total)}\n{progresso*100:.2f}%", x=0.5, y=0.5, font=dict(size=18), showarrow=False)],
        height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    st.plotly_chart(figo, use_container_width=True)

with c5:
    st.subheader("Rendimento da Carteira (Mensal)")
    if {"mes", "ret_mes"} <= set(df_rend.columns):
        figr = px.line(df_rend, x="mes", y="ret_mes", markers=True, template=PLOTLY_TEMPLATE,
                       labels={"mes": "", "ret_mes": "% no mês"})
        figr.add_hline(y=0, line_dash="dot", opacity=0.4)
        figr.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figr, use_container_width=True)
    else:
        st.info("Sem dados de rendimento.")

# =========================
# DICAS DE MAPEAMENTO (caso sua planilha tenha outros nomes)
# =========================
with st.expander("⚙️ Como ligar na sua planilha (passo a passo)"):
    st.markdown(
        """
**Rápido (sem Service Account):**
1. Abra sua planilha no Google Sheets → clique na aba **Movimentações/Lançamentos** → copie o `gid` da URL.  
2. Repita para **Posição/Consolidado** e **Proventos**.  
3. Preencha `SHEET_ID`, `GID_MOV`, `GID_POS`, `GID_PROV` no topo deste arquivo.

**Colunas esperadas (padrão, você pode adaptar nos `try/except` do código):**
- **MOV**: `data` (Data), `valor` (valor líquido do aporte/compra — positivo para aporte), `tipo` (opcional).  
- **POS**: `classe` (FII, Ação, FIAGRO…), `valor_atual` (ou `preco_atual`).  
- **PROV**: `ticker`, `valor_cota`, `qtde`, `total` (se não existir, calculo = valor_cota * qtde), `data_com`, `pagamento`.

Se preferir usar **Service Account**, troque `USE_SERVICE_ACCOUNT=True` e coloque o JSON em `st.secrets["gcp_service_account"]`.
"""
    )
