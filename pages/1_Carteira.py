# pages/1_Carteira.py
# Painel de Investimentos — Linkado ao Google Sheets

import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse

st.set_page_config(page_title="Painel de Investimentos – Linkado ao Google Sheets",
                   page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")

# =============================================================================
# CONFIG
# =============================================================================
SHEET_ID = "1p9IzDr-5ZV0phUHfNA_9d5xNvZW1IRo84LA__JyiiQc"

# Use as **chaves exatamente iguais aos nomes das abas**
GIDS = {
    "1. Meus Ativos": "441194831",
    "3. Proventos": "2109089485",
}

ABAS_CARTEIRA  = ["1. Meus Ativos"]
ABAS_PROVENTOS = ["3. Proventos"]

# =============================================================================
# HELPERS: leitura Google Sheets
# =============================================================================
@st.cache_data(ttl=300)
def _csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data(ttl=300)
def _csv_url_by_name(sheet_id: str, sheet_name: str) -> str:
    encoded = urllib.parse.quote(sheet_name, safe="")
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"

@st.cache_data(ttl=300)
def ler_aba(sheet_id: str, candidatos: list[str], gids: dict[str, str], dtype=str) -> pd.DataFrame:
    # 1) tenta por GID usando a **mesma chave do nome da aba**
    for nome in candidatos:
        if nome in gids:
            try:
                df = pd.read_csv(_csv_url_by_gid(sheet_id, gids[nome]), dtype=dtype)
                if not df.empty:
                    df.columns = [c.strip() for c in df.columns]
                    return df
            except Exception:
                pass
    # 2) fallback por NOME (gviz)
    for nome in candidatos:
        try:
            df = pd.read_csv(_csv_url_by_name(sheet_id, nome), dtype=dtype)
            if not df.empty:
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            pass
    return pd.DataFrame()

# =============================================================================
# HELPERS: parsing/normalização
# =============================================================================
def parse_brl_number(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA
    s = re.sub(r"[R$\s%]", "", s)
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA

def parse_int(x):
    try:
        return int(float(str(x).replace(",", ".").strip()))
    except Exception:
        return pd.NA

def parse_date(x):
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

# =============================================================================
# Padronização de Proventos
# =============================================================================
def padronizar_proventos(df_pv_raw: pd.DataFrame) -> pd.DataFrame:
    if df_pv_raw is None or df_pv_raw.empty:
        return pd.DataFrame(columns=[
            "data", "ticker", "tipo", "quantidade", "valor",
            "corretagem", "impostos", "instituicao"
        ])

    mapa = {
        "data": ["Data", "data", "Data do Crédito", "Data Crédito", "Data Credito"],
        "ticker": ["Ticker", "Ativo", "Código", "Código de Negociação", "Codigo"],
        "tipo": ["Tipo", "Tipo de Provento", "Evento", "Tipo Provento"],
        "quantidade": ["Qtd", "Quantidade", "QTD"],
        "valor": ["Total Líquido R$", "Valor Líquido", "Valor", "Total Líquido", "Provento Líquido", "Total"],
        "instituicao": ["Instituição", "Corretora", "Conta"],
        "corretagem": ["Corretagem", "Taxa de Corretagem"],
        "impostos": ["Impostos", "IR", "Taxas/Impostos", "IRRF"],
    }

    out = pd.DataFrame(index=df_pv_raw.index)

    for destino, candidatos in mapa.items():
        col_origem = next((c for c in candidatos if c in df_pv_raw.columns), None)
        out[destino] = df_pv_raw[col_origem] if col_origem else pd.NA

    out["data"] = out["data"].apply(parse_date)
    for c in ["valor", "corretagem", "impostos"]:
        out[c] = out[c].apply(parse_brl_number).astype("Float64")
    out["quantidade"] = out["quantidade"].apply(parse_int).astype("Int64")

    if "ticker" in out.columns:
        out["ticker"] = (out["ticker"].astype("string")
                         .str.upper().str.strip().str.replace(" ", "", regex=False))
    for c in ["tipo", "instituicao"]:
        out[c] = out[c].astype("string")

    col_ordem = ["data", "ticker", "tipo", "quantidade", "valor",
                 "corretagem", "impostos", "instituicao"]
    out = out[col_ordem]
    out = out[~(out["data"].isna() & out["valor"].isna())].reset_index(drop=True)
    return out

# =============================================================================
# Padronização de Carteira
# =============================================================================
def padronizar_carteira(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=[
            "ticker", "qtde", "pm", "preco_atual", "valor_posicao", "setor"
        ])

    mapa = {
        "ticker": ["Ticker", "Ativo", "Código", "Codigo"],
        "qtde": ["Quantidade (Liquida)", "Quantidade", "Qtd", "QTD"],
        "pm": ["Preço Médio Ajustado R$", "PM", "Preço Médio", "Preco Medio"],
        "preco_atual": ["Valor Atual", "Preço Atual", "Cotação", "Preco Atual", "Cotacao"],
        "valor_posicao": ["Valor Investido", "Valor da Posição", "Valor Posicao", "Valor"],
        "setor": ["Setor", "Classe", "Segmento"]
    }

    out = pd.DataFrame(index=df_raw.index)
    for destino, candidatos in mapa.items():
        col = next((c for c in candidatos if c in df_raw.columns), None)
        out[destino] = df_raw[col] if col else pd.NA

    out["ticker"] = (out["ticker"].astype("string")
                     .str.upper().str.strip().str.replace(" ", "", regex=False))
    for c in ["pm", "preco_atual", "valor_posicao"]:
        out[c] = out[c].apply(parse_brl_number).astype("Float64")
    out["qtde"] = out["qtde"].apply(parse_int).astype("Int64")

    out = out[~out["ticker"].isna()].reset_index(drop=True)
    return out

# =============================================================================
# CARREGAMENTO
# =============================================================================
with st.spinner("Carregando dados da planilha..."):
    df_cart_raw = ler_aba(SHEET_ID, ABAS_CARTEIRA, GIDS, dtype=str)
    df_pv_raw   = ler_aba(SHEET_ID, ABAS_PROVENTOS, GIDS, dtype=str)

# Diagnóstico
with st.expander("🔎 Diagnóstico das abas (colunas lidas)", expanded=False):
    st.write("**Carteira (raw):**", df_cart_raw.shape)
    if not df_cart_raw.empty:
        st.write(list(df_cart_raw.columns))
        st.dataframe(df_cart_raw.head(10), use_container_width=True)
    else:
        st.warning("Aba de **Carteira** não encontrada por GID nem por nome.")

    st.write("---")
    st.write("**Proventos (raw):**", df_pv_raw.shape)
    if not df_pv_raw.empty:
        st.write(list(df_pv_raw.columns))
        st.dataframe(df_pv_raw.head(10), use_container_width=True)
    else:
        st.warning("Aba de **Proventos** não encontrada por GID nem por nome.")

# Padronização
CARTEIRA  = padronizar_carteira(df_cart_raw)
PROVENTOS = padronizar_proventos(df_pv_raw)

# =============================================================================
# UI / Resultados
# =============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Carteira (padronizada)")
    if CARTEIRA.empty:
        st.info("Nenhum dado de Carteira disponível.")
    else:
        st.dataframe(CARTEIRA, use_container_width=True)
        total_posicao = (CARTEIRA["valor_posicao"].fillna(0)).sum()
        total_qtde = (CARTEIRA["qtde"].fillna(0)).sum()
        m1, m2 = st.columns(2)
        m1.metric("Valor total da posição", f"R$ {float(total_posicao or 0):,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
        m2.metric("Total de papéis (qtde)", f"{int(total_qtde or 0)}")

with col2:
    st.subheader("💰 Proventos (padronizados)")
    if PROVENTOS.empty:
        st.info("Nenhum dado de Proventos disponível.")
    else:
        st.dataframe(PROVENTOS, use_container_width=True)
        proventos_validos = PROVENTOS.dropna(subset=["data", "valor"]).copy()
        proventos_validos["ano"] = proventos_validos["data"].dt.year
        agg = proventos_validos.groupby("ano", dropna=True)["valor"].sum().reset_index()
        if not agg.empty:
            st.bar_chart(agg.set_index("ano"))
        total_prov = float(proventos_validos["valor"].sum() or 0.0)
        st.metric("Total de proventos", f"R$ {total_prov:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

st.caption("Se alguma aba não carregar, confira `SHEET_ID`, `GIDS` e os nomes em `ABAS_CARTEIRA` / `ABAS_PROVENTOS`.")
