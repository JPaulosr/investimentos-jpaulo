# app_investimentos_linkado.py
# -----------------------------------------------------------------
# Versão adaptada para a sua planilha real do Google Sheets
# - Lê diretamente as abas:
#   * "1. Meus Ativos"          -> Carteira atual / alocação / P&L
#   * "2. Lançamentos (B3)"     -> Histórico de compras/vendas/aportes
#   * "3. Proventos"            -> Proventos por data (dividendos/JCP/rendimentos)
# - Usa Service Account (se disponível em st.secrets) OU CSV export público como fallback
# - Mantém seus cálculos do Sheets (Preço Médio, Valor Atual etc.)
# - Filtros por período, classe e ticker
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

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

ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_LANCAMENTOS = st.secrets.get("ABA_LANCAMENTOS", "2. Lançamentos (B3)")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

# =========================
# LEITORES (gspread ou CSV export)
# =========================

def _ler_gspread(sheet_id: str, aba: str) -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        info = st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account")
        if not info:
            return pd.DataFrame()
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(aba)
        df = pd.DataFrame(ws.get_all_records())
        return df
    except Exception as e:
        st.warning(f"[gspread] Falha lendo '{aba}': {e}")
        return pd.DataFrame()


def _ler_csv_export(sheet_id: str, aba: str) -> pd.DataFrame:
    # Requer que a planilha permita leitura por link
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={aba}"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"[csv-export] Falha lendo '{aba}': {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def ler_aba(sheet_id: str, aba: str) -> pd.DataFrame:
    # Tenta gspread (secrets); cai para CSV export se não houver credenciais
    df = _ler_gspread(sheet_id, aba) if SHEET_ID else pd.DataFrame()
    if df.empty:
        df = _ler_csv_export(sheet_id, aba)
    return df

# =========================
# NORMALIZAÇÕES
# =========================

def _to_datetime_br(x):
    try:
        return pd.to_datetime(x, dayfirst=True, errors='coerce')
    except Exception:
        return pd.NaT


def padronizar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza as colunas mais comuns vistas na sua aba "1. Meus Ativos"."""
    if df.empty:
        return df
    cols = {c.lower().strip(): c for c in df.columns}

    # Tenta detectar colunas principais (aceita variações de acento/maiúsculas)
    def pick(*names):
        for n in names:
            for k in list(cols.keys()):
                if k.replace("ç", "c").replace("ã", "a").replace("é", "e").replace("ê", "e") == n.lower():
                    return cols[k]
        return None

    mapa = {
        "Ticker": pick("ticker"),
        "%NaCarteira": pick("% na carteira", "% na carteira", "percentual na carteira"),
        "Quantidade": pick("quantidade (liquida)", "quantidade", "qtd"),
        "PrecoMedioCompra": pick("preco medio (compra r$)", "preco medio compra r$", "preco medio"),
        "PrecoMedioAjustado": pick("preco medio ajustado (r$)", "preco medio ajustado"),
        "CotacaoHojeBRL": pick("cotacao de hoje (r$)", "cotacao hoje r$", "cotacao"),
        "CotacaoHojeUSD": pick("cotacao de hoje (us$)", "cotacao hoje us$"),
        "ValorInvestido": pick("valor investido"),
        "ValorAtual": pick("valor atual"),
        "ProventosMes": pick("proventos (do mes)", "proventos do mes"),
        "ProventosAnterior": pick("proventos (anterior)", "proventos anterior"),
        "ProventosProjetado": pick("proventos (projetado)", "proventos projetado"),
        "Classe": pick("classe", "classe do ativo", "tipo")
    }

    # Cria um DF enxuto com nomes padronizados
    out = pd.DataFrame()
    for novo, original in mapa.items():
        if original and original in df.columns:
            out[novo] = df[original]
        else:
            out[novo] = None

    # Tipagens
    for col in ["Quantidade", "PrecoMedioCompra", "PrecoMedioAjustado", "CotacaoHojeBRL", "CotacaoHojeUSD", "ValorInvestido", "ValorAtual", "ProventosMes", "ProventosAnterior", "ProventosProjetado"]:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    # Classe pode não existir nessa aba; manter como None
    return out


def padronizar_lancamentos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Colunas comuns dessa aba: Data, Tipo, Ticker, Quantidade, Preco, Taxas, Conta/Corretora, Classe
    ren = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=ren)
    possiveis = {
        "Data": ["Data"],
        "Tipo": ["Tipo"],
        "Ticker": ["Ticker"],
        "Quantidade": ["Quantidade", "Qtd"],
        "Preco": ["Preco", "Preço", "Preço (R$)"],
        "Taxas": ["Taxas", "Taxa"],
        "Conta": ["Conta", "Corretora"],
        "Classe": ["Classe", "Classe Do Ativo", "Tipo De Ativo"],
    }
    out = pd.DataFrame()
    for novo, cands in possiveis.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None
    out["Data"] = out["Data"].apply(_to_datetime_br)
    for n in ["Quantidade", "Preco", "Taxas"]:
        out[n] = pd.to_numeric(out[n], errors='coerce')
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
    return out


def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ren = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=ren)
    possiveis = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo", "Tipo De Provento"],
        "Valor": ["Valor", "Valor (R$)", "Provento R$"],
        "Classe": ["Classe", "Classe Do Ativo"],
    }
    out = pd.DataFrame()
    for novo, cands in possiveis.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None
    out["Data"] = out["Data"].apply(_to_datetime_br)
    out["Valor"] = pd.to_numeric(out["Valor"], errors='coerce')
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
    return out

# =========================
# CARREGAMENTO
# =========================
with st.spinner("Carregando dados da planilha..."):
    df_ativos_raw = ler_aba(SHEET_ID, ABA_ATIVOS)
    df_tx_raw     = ler_aba(SHEET_ID, ABA_LANCAMENTOS)
    df_pv_raw     = ler_aba(SHEET_ID, ABA_PROVENTOS)

DF_ATIVOS = padronizar_ativos(df_ativos_raw)
TX        = padronizar_lancamentos(df_tx_raw)
PV        = padronizar_proventos(df_pv_raw)

# =========================
# FILTROS – SIDEBAR
# =========================
with st.sidebar:
    st.header("Filtros")
    # Período
    min_data = min([d.min() for d in [TX.get("Data"), PV.get("Data")] if isinstance(d, pd.Series) and not d.empty] or [pd.to_datetime("2020-01-01")])
    max_data = max([d.max() for d in [TX.get("Data"), PV.get("Data")] if isinstance(d, pd.Series) and not d.empty] or [pd.Timestamp.today()])
    periodo = st.date_input("Período", (min_data.date(), max_data.date()))

    # Classe (se existir) e ticker
    classes = sorted(list(pd.unique(pd.concat([TX.get("Classe"), PV.get("Classe"), DF_ATIVOS.get("Classe")]).dropna()))) if not (TX.empty and PV.empty and DF_ATIVOS.empty) else []
    classe_sel = st.multiselect("Classe", options=classes, default=classes)

    tickers = sorted(list(pd.unique(pd.concat([TX.get("Ticker"), PV.get("Ticker"), DF_ATIVOS.get("Ticker")]).dropna()))) if not (TX.empty and PV.empty and DF_ATIVOS.empty) else []
    ticker_sel = st.multiselect("Ticker", options=tickers)

# Aplica filtros de período em TX/PV e de classe/ticker em todos
if not TX.empty:
    TX = TX[(TX["Data"].dt.date >= periodo[0]) & (TX["Data"].dt.date <= periodo[1])]
if not PV.empty:
    PV = PV[(PV["Data"].dt.date >= periodo[0]) & (PV["Data"].dt.date <= periodo[1])]

if classe_sel:
    if "Classe" in TX.columns: TX = TX[TX["Classe"].isin(classe_sel)]
    if "Classe" in PV.columns: PV = PV[PV["Classe"].isin(classe_sel)]
    if "Classe" in DF_ATIVOS.columns: DF_ATIVOS = DF_ATIVOS[DF_ATIVOS["Classe"].isin(classe_sel)]
if ticker_sel:
    for df in [TX, PV, DF_ATIVOS]:
        if not df.empty and "Ticker" in df.columns:
            df.dropna(subset=["Ticker"], inplace=True)
            df = df[df["Ticker"].isin(ticker_sel)]

# =========================
# SEÇÃO – CARTEIRA ATUAL (direto da sua aba "Meus Ativos")
# =========================
st.subheader("📦 Carteira Atual (da sua aba 'Meus Ativos')")
if DF_ATIVOS.empty:
    st.info("Não encontrei dados na aba de ativos. Confira o nome da aba nas configs.")
else:
    # KPIs
    v_investido = float(DF_ATIVOS["ValorInvestido"].sum(skipna=True)) if "ValorInvestido" in DF_ATIVOS.columns else 0.0
    v_atual = float(DF_ATIVOS["ValorAtual"].sum(skipna=True)) if "ValorAtual" in DF_ATIVOS.columns else 0.0
    pl = v_atual - v_investido
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor Investido", f"R$ {v_investido:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    col2.metric("Valor Atual",    f"R$ {v_atual:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    col3.metric("P/L Latente",    f"R$ {pl:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

    st.dataframe(DF_ATIVOS, use_container_width=True)

    # Alocação por ticker / classe
    if "ValorAtual" in DF_ATIVOS.columns and DF_ATIVOS["ValorAtual"].notna().any():
        aloc_ticker = DF_ATIVOS.groupby("Ticker", dropna=False)["ValorAtual"].sum().reset_index()
        fig = px.pie(aloc_ticker, names="Ticker", values="ValorAtual", hole=0.4, template=PLOTLY_TEMPLATE, title="Alocação por Ticker (Valor Atual)")
        st.plotly_chart(fig, use_container_width=True)

    if "Classe" in DF_ATIVOS.columns and "ValorAtual" in DF_ATIVOS.columns:
        aloc_classe = DF_ATIVOS.groupby("Classe", dropna=True)["ValorAtual"].sum().reset_index()
        if not aloc_classe.empty:
            fig = px.bar(aloc_classe, x="Classe", y="ValorAtual", template=PLOTLY_TEMPLATE, title="Alocação por Classe (Valor Atual)")
            st.plotly_chart(fig, use_container_width=True)

# =========================
# SEÇÃO – Aportes/Retiradas por mês (dos Lançamentos)
# =========================
st.subheader("💸 Aportes x Retiradas (mensal)")
if TX.empty:
    st.caption("Sem dados em '2. Lançamentos (B3)'.")
else:
    mov = TX[TX["Tipo"].isin(["APORTE", "RETIRADA"])].copy()
    if mov.empty:
        st.caption("Nenhum Aporte/Retirada no período.")
    else:
        mov["Ano"] = mov["Data"].dt.year
        mov["Mes"] = mov["Data"].dt.month
        mov["Valor"] = mov.apply(lambda r: (r.get("Quantidade", 0) or 0) * (r.get("Preco", 0) or 0), axis=1)
        mov.loc[mov["Tipo"] == "RETIRADA", "Valor"] *= -1
        grp = mov.groupby(["Ano", "Mes"], dropna=False)["Valor"].sum().reset_index()
        grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str) + "-" + grp["Mes"].astype(str) + "-01")
        fig = px.bar(grp, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Fluxo de Caixa Mensal")
        fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# SEÇÃO – Proventos por mês
# =========================
st.subheader("💰 Proventos (mensal)")
if PV.empty:
    st.caption("Sem dados na aba '3. Proventos'.")
else:
    pv = PV.copy()
    pv["Ano"] = pv["Data"].dt.year
    pv["Mes"] = pv["Data"].dt.month
    grp = pv.groupby(["Ano", "Mes"], dropna=False)["Valor"].sum().reset_index()
    grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str) + "-" + grp["Mes"].astype(str) + "-01")
    fig = px.bar(grp, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos por Mês")
    fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Proventos por Ticker no período filtrado")
    tab = pv.groupby(["Ticker", "Tipo"], dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
    st.dataframe(tab, hide_index=True, use_container_width=True)

# =========================
# DICAS / AJUSTES
# =========================
with st.expander("⚙️ Ajustes e dicas"):
    st.markdown(
        """
        - Se alguma coluna não aparecer, confira o **nome exato** das abas e colunas na planilha.
        - Para acesso com credenciais, adicione nos *Secrets* do Streamlit:
            - `SHEET_ID` = *ID da planilha*
            - `GCP_SERVICE_ACCOUNT` = *JSON da service account*
            - (opcional) renomeie as abas via `ABA_ATIVOS`, `ABA_LANCAMENTOS`, `ABA_PROVENTOS`.
        - Caso não use *Secrets*, deixe a planilha com leitura por link e o app usará o modo **CSV export**.
        """
    )
