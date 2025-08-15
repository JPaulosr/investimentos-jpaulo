# app_investimentos.py
# -----------------------------------------------------------------
# Painel de Investimentos — Linkado ao Google Sheets
# - Detecta abas por NOME (sem depender de GID)
# - Alvos: "2. Lançamentos (B3)" (carteira) e "3. Proventos"
# - Faz normalização de acentos/maiúsculas/nomes de colunas
# - Padroniza colunas principais para análise
# - Mostra diagnóstico e prévias
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata
from urllib.parse import quote

# =========================
# CONFIG PRINCIPAL
# =========================
st.set_page_config(page_title="Painel de Investimentos", page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")

# >>>>>>>> EDITE APENAS ISTO <<<<<<<<
SHEET_ID = "1p9IzDr-5ZV0phUHfNA_9d5xNvZW1IRo84LA__JyiiQc"  # <-- coloque aqui o ID da SUA planilha
# (opcional) para testes locais com Excel, passe excel_path no carregar_dados_investimentos(...)
EXCEL_LOCAL = None  # ex: r"/caminho/_PLANILHA - v4.5 (8).xlsx"

ALVOS_CARTEIRA = ["2. Lançamentos (B3)", "2.1. Lançamentos (Manual)"]
ALVOS_PROVENTOS = ["3. Proventos"]

# =========================
# FUNÇÕES UTILITÁRIAS
# =========================
def _norm(s: str) -> str:
    """Normaliza acento/caixa/espaços para comparar strings."""
    if s is None: 
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def escolher_aba_existente(lista_abas, candidatos):
    """Escolhe o primeiro candidato que existir na lista de abas (comparação normalizada)."""
    norm_map = { _norm(a): a for a in lista_abas }
    for cand in candidatos:
        n = _norm(cand)
        if n in norm_map:
            return norm_map[n]
    return None

def ler_gsheet_por_nome(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Lê Google Sheets por NOME da aba usando endpoint CSV (gviz).
    Evita depender de GID e funciona com nomes com espaços/acentos (via URL-encode).
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"
    df = pd.read_csv(url, dtype=str)
    return df

def limpar_numero_ptbr(x):
    """
    Converte strings 'R$ 1.234,56' / '1.234,56' / '3,1%' / '-1.234' em float.
    Se não der, retorna NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("R$", "").replace("%", "").replace(" ", "")
    # separador milhar '.' e decimal ','
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def to_datetime_br(s):
    if pd.isna(s): 
        return pd.NaT
    s = str(s).strip()
    # pandas infere DD/MM/YYYY automaticamente quando dayfirst=True
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except:
        return pd.NaT

def renomear_primeiro_match(df: pd.DataFrame, possiveis, novo_nome):
    """
    Procura, dentre 'possiveis' (lista de nomes alternativos), 
    qual coluna existe no DF e renomeia para 'novo_nome'.
    """
    colmap = {}
    cols_norm = { _norm(c): c for c in df.columns }
    for alt in possiveis:
        n = _norm(alt)
        if n in cols_norm:
            colmap[cols_norm[n]] = novo_nome
            break
    if colmap:
        df = df.rename(columns=colmap)
    return df

# =========================
# CARREGAMENTO DE DADOS
# =========================
@st.cache_data(ttl=300)
def descobrir_abas(sheet_id=None, excel_path=None):
    if excel_path:
        xls = pd.ExcelFile(excel_path)
        abas = xls.sheet_names
    else:
        # Para Sheets, se não listarmos, tentaremos diretamente pelos candidatos.
        # Ainda assim, retornamos os candidatos para a função leitora tentar um por um.
        abas = ALVOS_CARTEIRA + ALVOS_PROVENTOS

    nome_carteira = escolher_aba_existente(abas, ALVOS_CARTEIRA) or ALVOS_CARTEIRA[0]
    nome_proventos = escolher_aba_existente(abas, ALVOS_PROVENTOS) or ALVOS_PROVENTOS[0]
    return nome_carteira, nome_proventos, abas

@st.cache_data(ttl=300)
def carregar_dados_investimentos(sheet_id=SHEET_ID, excel_path=None):
    nome_carteira, nome_proventos, lista_abas = descobrir_abas(
        sheet_id=sheet_id if sheet_id else None, 
        excel_path=excel_path
    )

    err = None
    df_cart_raw = pd.DataFrame()
    df_prov_raw = pd.DataFrame()

    # Tenta ler CARTEIRA
    lidos = []
    for alvo in [nome_carteira] + [a for a in ALVOS_CARTEIRA if a != nome_carteira]:
        try:
            if excel_path:
                df = pd.read_excel(excel_path, sheet_name=alvo, dtype=str)
            else:
                df = ler_gsheet_por_nome(sheet_id, alvo)
            if df.shape[0] > 0 and df.shape[1] > 0:
                df_cart_raw = df.copy()
                lidos.append(("carteira", alvo))
                break
        except Exception as e:
            err = f"Falha lendo carteira ({alvo}): {e}"

    # Tenta ler PROVENTOS
    for alvo in [nome_proventos] + [a for a in ALVOS_PROVENTOS if a != nome_proventos]:
        try:
            if excel_path:
                df = pd.read_excel(excel_path, sheet_name=alvo, dtype=str)
            else:
                df = ler_gsheet_por_nome(sheet_id, alvo)
            if df.shape[0] > 0 and df.shape[1] > 0:
                df_prov_raw = df.copy()
                lidos.append(("proventos", alvo))
                break
        except Exception as e:
            err = f"Falha lendo proventos ({alvo}): {e}"

    return df_cart_raw, df_prov_raw, lidos, lista_abas, err

# =========================
# PADRONIZAÇÕES
# =========================
def padronizar_carteira(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza a aba '2. Lançamentos (B3)' para colunas principais:
    ['data','ticker','tipo','quantidade','preco','total','taxa','irrf','classe','nome_acao']
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    # Primeiro tenta renomear as colunas que conhecemos na planilha modelo:
    df = renomear_primeiro_match(df, ["Data (DD/MM/YYYY)", "Data\n(DD/MM/YYYY)", "Data"], "data")
    df = renomear_primeiro_match(df, ["Ticker"], "ticker")
    df = renomear_primeiro_match(df, ["Tipo de Operação", "Tipo Operação", "Operação"], "tipo")
    df = renomear_primeiro_match(df, ["Quantidade", "Qtd", "Qtde"], "quantidade")
    df = renomear_primeiro_match(df, ["Preço (por unidade)", "Preço\n(por unidade)", "Preço"], "preco")
    df = renomear_primeiro_match(df, ["Total da Operação", "Total Operação"], "total")
    df = renomear_primeiro_match(df, ["Taxa"], "taxa")
    df = renomear_primeiro_match(df, ["IRRF"], "irrf")
    df = renomear_primeiro_match(df, ["Classe"], "classe")
    df = renomear_primeiro_match(df, ["Nome da ação", "Nome"], "nome_acao")

    # Converte tipos
    if "data" in df.columns:
        df["data"] = df["data"].apply(to_datetime_br)

    for c in ["quantidade", "preco", "total", "taxa", "irrf"]:
        if c in df.columns:
            df[c] = df[c].apply(limpar_numero_ptbr)

    # Remove linhas totalmente vazias
    df = df.dropna(how="all")
    return df

def padronizar_proventos(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    A aba '3. Proventos' geralmente tem um cabeçalho em 2ª linha.
    Detectamos a linha que contém 'Ticker' e reconstituímos o header.
    Colunas padrão adotadas: ['ticker','tipo','data','quantidade','unitario','total']
    (total é calculado se não vier)
    """
    if df_raw.empty:
        return df_raw

    df0 = df_raw.copy()

    # Descobrir a linha de cabeçalho: onde a segunda coluna seja 'Ticker' (observado na planilha)
    header_row = None
    for i in range(min(10, len(df0))):
        row = df0.iloc[i].tolist()
        # Se em alguma das colunas está 'Ticker', usamos aquela linha como header
        if any(str(x).strip().lower() == "ticker" for x in row):
            header_row = i
            break

    if header_row is not None:
        new_header = df0.iloc[header_row].tolist()
        df = df0.iloc[header_row+1:].copy()
        df.columns = new_header
    else:
        # Se não encontrar, segue como veio
        df = df0.copy()

    # Renomeia os campos importantes
    df = renomear_primeiro_match(df, ["Ticker"], "ticker")
    df = renomear_primeiro_match(df, ["Tipo Provento", "Tipo", "Evento"], "tipo")
    df = renomear_primeiro_match(df, ["Data"], "data")
    df = renomear_primeiro_match(df, ["Quantidade", "Qtd"], "quantidade")
    df = renomear_primeiro_match(df, ["Unitário R$", "Unitario R$", "Unitário", "Unitario"], "unitario")
    # Alguns modelos trazem 'Total Líquido'/'Total Liquido R$'
    df = renomear_primeiro_match(df, ["Total Líquido", "Total Liquido R$", "Total"], "total")

    # Converte tipos
    if "data" in df.columns:
        df["data"] = df["data"].apply(to_datetime_br)
    for c in ["quantidade", "unitario", "total"]:
        if c in df.columns:
            df[c] = df[c].apply(limpar_numero_ptbr)

    # Calcula total se faltar
    if "total" not in df.columns and {"quantidade", "unitario"}.issubset(df.columns):
        df["total"] = df["quantidade"] * df["unitario"]

    # Mantém apenas colunas principais + extras úteis se existirem
    keep = [c for c in ["ticker", "tipo", "data", "quantidade", "unitario", "total"] if c in df.columns]
    extras = [c for c in df.columns if c not in keep]
    df = df[keep + extras]

    # Remove linhas vazias/ruído
    df = df.dropna(how="all")
    # Remove linhas onde ticker é nulo (após limpar cabeçalho)
    if "ticker" in df.columns:
        df = df[~df["ticker"].isna()]

    return df

# =========================
# CARREGA E EXIBE
# =========================
with st.expander("🔎 Diagnóstico das abas (colunas lidas)", expanded=True):
    df_cart_raw, df_prov_raw, lidos, lista_abas, err = carregar_dados_investimentos(
        sheet_id=SHEET_ID, excel_path=EXCEL_LOCAL
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Carteira (raw)")
        if not df_cart_raw.empty:
            st.write(df_cart_raw.shape)
            st.dataframe(df_cart_raw.head(10), use_container_width=True)
        else:
            st.warning("Aba de **Carteira** não encontrada por NOME (tentativas: " 
                       + ", ".join(ALVOS_CARTEIRA) + ").")

    with col2:
        st.subheader("Proventos (raw)")
        if not df_prov_raw.empty:
            st.write(df_prov_raw.shape)
            st.dataframe(df_prov_raw.head(10), use_container_width=True)
        else:
            st.warning("Aba de **Proventos** não encontrada por NOME (tentativas: " 
                       + ", ".join(ALVOS_PROVENTOS) + ").")

    if lidos:
        st.caption("✔️ Abas reconhecidas: " + ", ".join([f"{t}:{n}" for t, n in lidos]))
    if err:
        st.error(err)

# Padroniza
df_carteira = padronizar_carteira(df_cart_raw)
df_proventos = padronizar_proventos(df_prov_raw)

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("🧱 Carteira (padronizada)")
    if df_carteira.empty:
        st.info("Nenhum dado de Carteira disponível.")
    else:
        # Métricas simples
        qt_ops = len(df_carteira)
        tickers = df_carteira["ticker"].dropna().nunique() if "ticker" in df_carteira.columns else 0
        total_ops = df_carteira["total"].sum() if "total" in df_carteira.columns else np.nan
        m1, m2, m3 = st.columns(3)
        m1.metric("Operações", f"{qt_ops:,}".replace(",", "."))
        m2.metric("Tickers distintos", f"{tickers:,}".replace(",", "."))
        m3.metric("Somatório Total da Operação", 
                  "R$ {:,.2f}".format(total_ops if pd.notna(total_ops) else 0).replace(",", "v").replace(".", ",").replace("v", "."))

        st.dataframe(df_carteira.head(50), use_container_width=True, height=400)

with c2:
    st.subheader("💰 Proventos (padronizados)")
    if df_proventos.empty:
        st.info("Nenhum dado de Proventos disponível.")
    else:
        # Métricas simples
        qt_linhas = len(df_proventos)
        tickers_p = df_proventos["ticker"].dropna().nunique() if "ticker" in df_proventos.columns else 0
        total_prov = df_proventos["total"].sum() if "total" in df_proventos.columns else np.nan
        n1, n2, n3 = st.columns(3)
        n1.metric("Registros", f"{qt_linhas:,}".replace(",", "."))
        n2.metric("Tickers (proventos)", f"{tickers_p:,}".replace(",", "."))
        n3.metric("Total (estimado)", 
                  "R$ {:,.2f}".format(total_prov if pd.notna(total_prov) else 0).replace(",", "v").replace(".", ",").replace("v", "."))

        st.dataframe(df_proventos.head(50), use_container_width=True, height=400)

# =========================
# SIDEBAR: navegação simples
# =========================
st.sidebar.markdown("## app investimentos")
st.sidebar.button("Carteira", use_container_width=True)
st.sidebar.button("Proventos", use_container_width=True)

st.caption("Se alguma aba não carregar, confira o **SHEET_ID** e os nomes em `ALVOS_CARTEIRA` / `ALVOS_PROVENTOS`.")
