# app_investimentos_linkado.py
# -----------------------------------------------------------------
# Versão adaptada para a sua planilha real do Google Sheets
# Corrigido:
#  - Leitura por CSV export com nomes de abas contendo espaços/()/. (URL-encode)
#  - Alternativa por GID (export?format=csv&gid=...)
#  - Limpeza de números no formato pt-BR ("R$ 1.234,56", "3,21%")
#  - Mapeamento das colunas de "2. Lançamentos (B3)" (Tipo de Operação, Total da Operação etc.)
#  - Filtro por ticker/classe aplicado corretamente em todos os DFs
#  - Aportes = Compras (+) e Vendas (-) quando não houver APORTE/RETIRADA
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from urllib.parse import quote

# =========================
# CONFIGURAÇÃO GERAL
# =========================
st.set_page_config(page_title="📈 Investimentos – Linkado ao Google Sheets", page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")
PLOTLY_TEMPLATE = "plotly_dark"

# =========================
# PARAMS DA SUA PLANILHA
# =========================
# 👉 Defina nos Secrets (ou edite aqui):
SHEET_ID = st.secrets.get("SHEET_ID", "")
# Abas por NOME (podem ter espaços e parênteses)
ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_LANCAMENTOS = st.secrets.get("ABA_LANCAMENTOS", "2. Lançamentos (B3)")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")
# OU use GIDs (numéricos). Se preencher *_GID, eles têm prioridade sobre o nome da aba.
ABA_ATIVOS_GID      = st.secrets.get("ABA_ATIVOS_GID", "")
ABA_LANCAMENTOS_GID = st.secrets.get("ABA_LANCAMENTOS_GID", "")
ABA_PROVENTOS_GID   = st.secrets.get("ABA_PROVENTOS_GID", "")

# =========================
# HELPERS DE LIMPEZA
# =========================

def br_to_float(x):
    """Converte strings no formato pt-BR para float. Ex.: 'R$ 1.234,56' -> 1234.56."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    # remove símbolo e espaços
    s = s.replace("R$", "").replace("%", "").replace("US$", "").replace("$", "").replace(" ", "")
    # remove separador de milhar "." e troca vírgula por ponto
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def to_datetime_br(series):
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

# =========================
# LEITORES (gspread ou CSV export)
# =========================

def _ler_gspread(sheet_id: str, aba_nome: str) -> pd.DataFrame:
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
        ws = sh.worksheet(aba_nome)
        df = pd.DataFrame(ws.get_all_records())
        return df
    except Exception as e:
        st.warning(f"[gspread] Falha lendo '{aba_nome}': {e}")
        return pd.DataFrame()


def _ler_csv_export_by_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"[csv-export] Falha lendo gid={gid}: {e}")
        return pd.DataFrame()


def _ler_csv_export_by_name(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    try:
        aba_enc = quote(aba_nome, safe="")  # URL-encode do nome da aba
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={aba_enc}"
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"[csv-export] Falha lendo '{aba_nome}': {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def ler_aba(sheet_id: str, aba_nome: str, gid: str = "") -> pd.DataFrame:
    # 1) Se tiver GID, usa export por gid (mais estável)
    if sheet_id and gid:
        df = _ler_csv_export_by_gid(sheet_id, gid)
        if not df.empty:
            return df
    # 2) Se tiver credenciais, tenta gspread por nome
    info = st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account")
    if sheet_id and info and aba_nome:
        df = _ler_gspread(sheet_id, aba_nome)
        if not df.empty:
            return df
    # 3) Fallback: CSV export por nome (precisa planilha pública p/ leitura)
    if sheet_id and aba_nome:
        return _ler_csv_export_by_name(sheet_id, aba_nome)
    return pd.DataFrame()

# =========================
# NORMALIZAÇÕES / PADRONIZAÇÃO
# =========================

def padronizar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Normaliza nomes (case/acentos variáveis)
    lower_map = {c.lower().strip(): c for c in df.columns}
    def get_col(*cands):
        for c in cands:
            for k, v in lower_map.items():
                kk = k.replace("ç","c").replace("ã","a").replace("õ","o").replace("é","e").replace("ê","e")
                cc = c.lower().replace("ç","c").replace("ã","a").replace("õ","o").replace("é","e").replace("ê","e")
                if kk == cc:
                    return v
        return None

    mapa = {
        "Ticker": get_col("ticker"),
        "%NaCarteira": get_col("% na carteira", "percentual na carteira"),
        "Quantidade": get_col("quantidade (liquida)", "quantidade", "qtd"),
        "PrecoMedioCompra": get_col("preco medio (compra r$)", "preco medio compra r$", "preco medio"),
        "PrecoMedioAjustado": get_col("preco medio ajustado (r$)", "preco medio ajustado"),
        "CotacaoHojeBRL": get_col("cotacao de hoje (r$)", "cotacao hoje r$", "cotacao r$"),
        "CotacaoHojeUSD": get_col("cotacao de hoje (us$)", "cotacao hoje us$"),
        "ValorInvestido": get_col("valor investido"),
        "ValorAtual": get_col("valor atual"),
        "ProventosMes": get_col("proventos (do mes)", "proventos do mes"),
        "ProventosAnterior": get_col("proventos (anterior)", "proventos anterior"),
        "ProventosProjetado": get_col("proventos (projetado)", "proventos projetado"),
        "Classe": get_col("classe", "classe do ativo", "tipo")
    }

    out = pd.DataFrame({k: (df[v] if v in df.columns else None) for k, v in mapa.items()})
    # Limpeza numérica
    for col in ["%NaCarteira", "Quantidade", "PrecoMedioCompra", "PrecoMedioAjustado", "CotacaoHojeBRL", "CotacaoHojeUSD", "ValorInvestido", "ValorAtual", "ProventosMes", "ProventosAnterior", "ProventosProjetado"]:
        out[col] = out[col].map(br_to_float)
    return out


def padronizar_lancamentos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Renomeia títulos comum
    df = df.rename(columns={c: c.strip() for c in df.columns})
    poss = {
        "Classe": ["Classe", "Classe do Ativo", "Tipo de Ativo"],
        "Ticker": ["Ticker"],
        "Data": ["Data", "Data (DD/MM/YYYY)", "Data da Operação"],
        "Tipo": ["Tipo", "Tipo de Operação", "Operação"],
        "Quantidade": ["Quantidade", "Qtd"],
        "Preco": ["Preço (por unidade)", "Preço", "Preco"],
        "Taxas": ["Taxa", "Taxas"],
        "IRRF": ["IRRF"],
        "TotalOperacao": ["Total da Operação", "Total da Operacao", "Valor Bruto"],
        "Mes": ["Mês", "Mes"],
        "Ano": ["Ano"],
    }
    out = pd.DataFrame()
    for novo, cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None

    # Tipos
    out["Data"] = to_datetime_br(out["Data"])
    for col in ["Quantidade", "Preco", "Taxas", "IRRF", "TotalOperacao"]:
        out[col] = out[col].map(br_to_float)
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
        # normaliza possíveis valores
        out["Tipo"] = out["Tipo"].replace({"COMPRA": "COMPRA", "VENDA": "VENDA", "APORTE": "APORTE", "RETIRADA": "RETIRADA"})
    return out


def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: c.strip() for c in df.columns})
    poss = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo", "Tipo de Provento"],
        "Valor": ["Valor", "Valor (R$)", "Provento R$"],
        "Classe": ["Classe", "Classe do Ativo"],
    }
    out = pd.DataFrame()
    for novo, cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None
    out["Data"] = to_datetime_br(out["Data"])
    out["Valor"] = out["Valor"].map(br_to_float)
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
    return out

# =========================
# CARREGAMENTO
# =========================
with st.spinner("Carregando dados da planilha..."):
    df_ativos_raw = ler_aba(SHEET_ID, ABA_ATIVOS, ABA_ATIVOS_GID)
    df_tx_raw     = ler_aba(SHEET_ID, ABA_LANCAMENTOS, ABA_LANCAMENTOS_GID)
    df_pv_raw     = ler_aba(SHEET_ID, ABA_PROVENTOS, ABA_PROVENTOS_GID)

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

    # Classe e ticker
    def uniq(series_list):
        vals = pd.Series(dtype="object")
        for s in series_list:
            if isinstance(s, pd.Series) and not s.empty:
                vals = pd.concat([vals, s.dropna().astype(str)])
        return sorted(vals.unique().tolist())

    classes = uniq([TX.get("Classe"), PV.get("Classe"), DF_ATIVOS.get("Classe")])
    classe_sel = st.multiselect("Classe", options=classes, default=classes)

    tickers = uniq([TX.get("Ticker"), PV.get("Ticker"), DF_ATIVOS.get("Ticker")])
    ticker_sel = st.multiselect("Ticker", options=tickers)

# Aplica filtros
if not TX.empty:
    TX = TX[(TX["Data"].dt.date >= periodo[0]) & (TX["Data"].dt.date <= periodo[1])]
if not PV.empty:
    PV = PV[(PV["Data"].dt.date >= periodo[0]) & (PV["Data"].dt.date <= periodo[1])]

if classe_sel:
    if "Classe" in DF_ATIVOS.columns:
        DF_ATIVOS = DF_ATIVOS[DF_ATIVOS["Classe"].isin(classe_sel)]
    if not TX.empty and "Classe" in TX.columns:
        TX = TX[TX["Classe"].isin(classe_sel)]
    if not PV.empty and "Classe" in PV.columns:
        PV = PV[PV["Classe"].isin(classe_sel)]

if ticker_sel:
    if "Ticker" in DF_ATIVOS.columns:
        DF_ATIVOS = DF_ATIVOS[DF_ATIVOS["Ticker"].isin(ticker_sel)]
    if not TX.empty and "Ticker" in TX.columns:
        TX = TX[TX["Ticker"].isin(ticker_sel)]
    if not PV.empty and "Ticker" in PV.columns:
        PV = PV[PV["Ticker"].isin(ticker_sel)]

# =========================
# SEÇÃO – CARTEIRA ATUAL (aba "Meus Ativos")
# =========================
st.subheader("📦 Carteira Atual (da sua aba 'Meus Ativos')")
if DF_ATIVOS.empty:
    st.info("Não encontrei dados na aba de ativos. Confira o SHEET_ID, os nomes das abas ou preencha os GIDs nos Secrets.")
else:
    v_investido = float(pd.Series(DF_ATIVOS.get("ValorInvestido", pd.Series())).sum(skipna=True) or 0)
    v_atual     = float(pd.Series(DF_ATIVOS.get("ValorAtual", pd.Series())).sum(skipna=True) or 0)
    pl          = v_atual - v_investido
    c1, c2, c3 = st.columns(3)
    c1.metric("Valor Investido", f"R$ {v_investido:,.2f}".replace(",","v").replace(".",",").replace("v","."))
    c2.metric("Valor Atual",    f"R$ {v_atual:,.2f}".replace(",","v").replace(".",",").replace("v","."))
    c3.metric("P/L Latente",    f"R$ {pl:,.2f}".replace(",","v").replace(".",",").replace("v","."))

    st.dataframe(DF_ATIVOS, use_container_width=True)

    # Alocação por ticker
    if "ValorAtual" in DF_ATIVOS.columns and DF_ATIVOS["ValorAtual"].notna().any():
        aloc_ticker = DF_ATIVOS.groupby("Ticker", dropna=False)["ValorAtual"].sum().reset_index()
        fig = px.pie(aloc_ticker, names="Ticker", values="ValorAtual", hole=0.4, template=PLOTLY_TEMPLATE, title="Alocação por Ticker (Valor Atual)")
        st.plotly_chart(fig, use_container_width=True)

    # Alocação por classe
    if "Classe" in DF_ATIVOS.columns and "ValorAtual" in DF_ATIVOS.columns:
        aloc_classe = DF_ATIVOS.dropna(subset=["Classe"]).groupby("Classe")["ValorAtual"].sum().reset_index()
        if not aloc_classe.empty:
            fig = px.bar(aloc_classe, x="Classe", y="ValorAtual", template=PLOTLY_TEMPLATE, title="Alocação por Classe (Valor Atual)")
            st.plotly_chart(fig, use_container_width=True)

# =========================
# SEÇÃO – Aportes/Retiradas por mês (Lançamentos)
# =========================
st.subheader("💸 Aportes x Retiradas (mensal)")
if TX.empty:
    st.caption("Sem dados em '2. Lançamentos (B3)'.")
else:
    mov = TX.copy()
    # Se não houver APORTE/RETIRADA explícitos, usa regra: COMPRA = +TotalOperacao; VENDA = -TotalOperacao
    if "TotalOperacao" in mov.columns and mov["TotalOperacao"].notna().any():
        mov["Valor"] = mov["TotalOperacao"].fillna(0)
    else:
        mov["Valor"] = (mov.get("Quantidade", 0) * mov.get("Preco", 0)).fillna(0)
    mov.loc[mov["Tipo"]=="VENDA", "Valor"] *= -1
    mov.loc[mov["Tipo"]=="RETIRADA", "Valor"] *= -1
    mov = mov[(mov["Tipo"].isin(["COMPRA","VENDA","APORTE","RETIRADA"])) & mov["Data"].notna()]

    if mov.empty:
        st.caption("Nenhum movimento válido no período.")
    else:
        grp = mov.assign(Ano=mov["Data"].dt.year, Mes=mov["Data"].dt.month)
        grp = grp.groupby(["Ano","Mes"], dropna=False)["Valor"].sum().reset_index()
        grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str) + "-" + grp["Mes"].astype(str) + "-01")
        fig = px.bar(grp, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Fluxo de Caixa Mensal (Aportes líquidos)")
        fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# SEÇÃO – Proventos por mês
# =========================
st.subheader("💰 Proventos (mensal)")
if PV.empty:
    st.caption("Sem dados na aba '3. Proventos'.")
else:
    pv = PV.dropna(subset=["Data"]).copy()
    if pv.empty:
        st.caption("Registros de proventos sem data.")
    else:
        grp = pv.assign(Ano=pv["Data"].dt.year, Mes=pv["Data"].dt.month)
        grp = grp.groupby(["Ano", "Mes"], dropna=False)["Valor"].sum().reset_index()
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
        **Se aparecer aviso de URL com espaços**: agora já tratamos com *URL-encode*. Ainda assim, nomes de abas com acentos/() funcionam.

        **Preferência por GID**: para máxima estabilidade, informe nos *Secrets* os GIDs das abas:
        - `ABA_ATIVOS_GID`, `ABA_LANCAMENTOS_GID`, `ABA_PROVENTOS_GID`
        (abra a aba no navegador e copie o número após `gid=`).

        **Aportes**: quando sua aba não traz APORTE/RETIRADA explícitos, o gráfico usa `COMPRA` como entrada (+) e `VENDA` como saída (-) com base no **Total da Operação**.
        """
    )
