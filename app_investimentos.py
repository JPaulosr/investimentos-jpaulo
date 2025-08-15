# app_investimentos_linkado.py
# -----------------------------------------------------------------
# Versão adaptada para a sua planilha real do Google Sheets
# FIXES nesta versão:
# 1) Corrige erro "URL can't contain control characters" (nomes de abas com espaços e parênteses)
#    -> agora usa GID quando disponível OU faz URL-encode do nome da aba.
# 2) Converte valores no formato BRL ("R$ 1.234,56") e percentuais ("5,23%") para float.
# 3) Corrige filtros por ticker/classe e pequenos ajustes de robustez.
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from urllib.parse import quote
import re

# =========================
# CONFIGURAÇÃO GERAL
# =========================
st.set_page_config(page_title="📈 Investimentos – Linkado ao Google Sheets", page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")
PLOTLY_TEMPLATE = "plotly_dark"

# =========================
# PARAMS DA SUA PLANILHA
# =========================
# 👉 Substitua pelo ID da SUA planilha (o ID é a parte entre /d/ e /edit)
SHEET_ID = st.secrets.get("SHEET_ID", "")

# Abas por NOME (usadas se GID não for informado)
ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_LANCAMENTOS = st.secrets.get("ABA_LANCAMENTOS", "2. Lançamentos (B3)")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

# Abas por GID (recomendado para nomes com espaços/acentos)
# Dica: copie o número após "gid=" da URL quando a aba estiver aberta.
GID_ATIVOS      = st.secrets.get("GID_ATIVOS", "")
GID_LANCAMENTOS = st.secrets.get("GID_LANCAMENTOS", "")
GID_PROVENTOS   = st.secrets.get("GID_PROVENTOS", "")

# =========================
# HELPERS DE FORMATAÇÃO
# =========================

def parse_brl(x):
    """Converte strings como 'R$ 1.234,56', '1.234,56', '1,234.56' em float.
    Mantém números já numéricos. Retorna NaN quando não der para converter."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return pd.NA
    # remove símbolo R$ e espaços
    s = re.sub(r"[^0-9,.-]", "", s)
    # casos do BR: 1.234,56
    if s.count(",") == 1 and (s.rfind(",") > s.rfind(".")):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA


def parse_pct(x):
    """Converte '5,23%'/'5.23%' para decimal (0.0523)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    s = str(x).strip().replace("%", "")
    v = parse_br
