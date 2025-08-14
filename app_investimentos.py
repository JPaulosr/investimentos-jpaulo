# app_investimentos.py
# -------------------------------------------------------------
# MVP de um app de investimentos focado no seu fluxo (JPaulo)
# - Lê dados da sua planilha do Google Sheets (ou CSV como fallback)
# - Mostra: Carteira atual, Aportes, Proventos, Alocação e Painéis por classe
# - Filtros por período, classe e ativo
# - Pronto para virar multipágina (basta quebrar as seções em /pages)
# -------------------------------------------------------------

import os
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# CONFIG GERAL
# =========================
st.set_page_config(page_title="📈 Meus Investimentos (MVP)", page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – MVP")
PLOTLY_TEMPLATE = "plotly_dark"

# =========================
# CONFIG DA FONTE DE DADOS
# =========================
# Opção A) Google Sheets (recomendado)
# - Crie uma planilha com as abas: RELATORIO_TRANSACOES, PROVENTOS, COTACOES (opcional)
# - Adicione suas credenciais JSON nos Secrets do Streamlit como GCP_SERVICE_ACCOUNT
# - Informe o SHEET_ID abaixo

SHEET_ID = st.secrets.get("SHEET_ID", "")  # opcional: defina nos Secrets
USAR_GOOGLE_SHEETS = bool(SHEET_ID)

# Opção B) CSV (fallback local)
CSV_TRANSACOES = "relatorio_transacoes.csv"
CSV_PROVENTOS  = "proventos.csv"
CSV_COTACOES   = "cotacoes.csv"  # opcional: preço atual por ticker

# =========================
# COLUNAS ESPERADAS
# =========================
# RELATORIO_TRANSACOES (uma linha por evento de compra/venda/aporte/retirada)
# Data, Tipo, Classe, Ticker, Ativo, Quantidade, Preco, Taxas, Corretora, Conta
#  - Tipo: COMPRA | VENDA | APORTE | RETIRADA | TRANSFERENCIA
#  - Classe: ACOES | FII | RF | ETF | BDR | CRIPTO | OUTROS
# PROVENTOS
# Data, Classe, Ticker, Ativo, Tipo, Valor, Quantidade (opcional), Conta
#  - Tipo: DIVIDENDO | JCP | RENDIMENTO | OUTRO
# COTACOES (opcional)
# Ticker, PrecoAtual, DataCotacao

# =========================
# HELPERS
# =========================
@st.cache_data(ttl=300)
def _ler_google_sheet(sheet_id: str, aba: str) -> pd.DataFrame:
    """Lê uma aba específica do Google Sheets usando pandas + CSV export.
    Precisa que a planilha seja compartilhada como 'Qualquer pessoa com o link: Leitor'
    OU use acesso com Service Account, trocando para gspread. Aqui vou pelo CSV export p/ simplicidade.
    """
    try:
        # Export CSV da aba via gid quando possível; como fallback, usa 'gviz/tq' que ignora gid
        # Se você souber o GID, pode montar a URL com gid. Aqui, usamos o formato 'gviz' mais geral.
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={aba}"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"Falha ao ler a aba '{aba}' no Google Sheets: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def carregar_bases() -> dict:
    if USAR_GOOGLE_SHEETS:
        tx = _ler_google_sheet(SHEET_ID, "RELATORIO_TRANSACOES")
        pv = _ler_google_sheet(SHEET_ID, "PROVENTOS")
        ct = _ler_google_sheet(SHEET_ID, "COTACOES")
    else:
        tx = pd.read_csv(CSV_TRANSACOES) if os.path.exists(CSV_TRANSACOES) else pd.DataFrame()
        pv = pd.read_csv(CSV_PROVENTOS)  if os.path.exists(CSV_PROVENTOS)  else pd.DataFrame()
        ct = pd.read_csv(CSV_COTACOES)   if os.path.exists(CSV_COTACOES)   else pd.DataFrame()
    return {"transacoes": tx, "proventos": pv, "cotacoes": ct}


def _padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().title() for c in df.columns]
    # Normalizações usuais
    rename = {
        "Preço": "Preco", "Preço Medio": "Preco", "Preço Médio": "Preco",
        "Quantidade (Un)": "Quantidade", "Qtd": "Quantidade",
        "Taxa": "Taxas", "Taxas Totais": "Taxas",
        "Valor Bruto": "Valor", "Classe Ativo": "Classe"
    }
    df = df.rename(columns=rename)
    # Datas
    for col in ["Data", "Data Cotacao", "DataCotacao", "Data Provento"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            except Exception:
                pass
    # Consolidar colunas equivalentes
    if "Data Cotacao" in df.columns and "DataCotacao" not in df.columns:
        df["DataCotacao"] = df["Data Cotacao"]
    return df


def preparar_dados(bases: dict):
    tx = _padronizar_colunas(bases.get("transacoes", pd.DataFrame()))
    pv = _padronizar_colunas(bases.get("proventos", pd.DataFrame()))
    ct = _padronizar_colunas(bases.get("cotacoes", pd.DataFrame()))

    # Garantir colunas mínimas
    for df, obrigatorias in [
        (tx, ["Data", "Tipo", "Classe", "Ticker", "Ativo", "Quantidade", "Preco"]),
        (pv, ["Data", "Classe", "Ticker", "Tipo", "Valor"])
    ]:
        for c in obrigatorias:
            if c not in df.columns:
                df[c] = None

    # Tipagem básica
    for c in ["Quantidade", "Preco", "Taxas", "Valor"]:
        if c in tx.columns:
            tx[c] = pd.to_numeric(tx[c], errors='coerce')
        if c in pv.columns:
            pv[c] = pd.to_numeric(pv[c], errors='coerce')

    # Classes e tipos padronizados
    for col in ["Tipo", "Classe"]:
        if col in tx.columns:
            tx[col] = tx[col].astype(str).str.upper().str.strip()
        if col in pv.columns:
            pv[col] = pv[col].astype(str).str.upper().str.strip()

    # Preços atuais
    if not ct.empty:
        ct_cols = ["Ticker", "PrecoAtual", "DataCotacao"]
        # Tenta detectar nomes alternativos
        if "Preco" in ct.columns and "PrecoAtual" not in ct.columns:
            ct = ct.rename(columns={"Preco": "PrecoAtual"})
        for c in ct_cols:
            if c not in ct.columns:
                ct[c] = None
    return tx, pv, ct

# =========================
# LÓGICAS DE CÁLCULO
# =========================

def carteira_atual(transacoes: pd.DataFrame, cotacoes: pd.DataFrame) -> pd.DataFrame:
    """Calcula posição atual por Ticker (qtd atual, PM, custo, valor de mercado, P/L latente)."""
    if transacoes.empty:
        return pd.DataFrame()
    tx = transacoes[transacoes["Tipo"].isin(["COMPRA", "VENDA"])].copy()
    tx["Sinal"] = tx["Tipo"].map({"COMPRA": 1, "VENDA": -1})
    tx["Quantidade"] = pd.to_numeric(tx["Quantidade"], errors='coerce').fillna(0)
    tx["Preco"] = pd.to_numeric(tx["Preco"], errors='coerce').fillna(0)
    tx["Taxas"] = pd.to_numeric(tx.get("Taxas", 0), errors='coerce').fillna(0)

    # Acúmulo por ordem cronológica para PM
    tx = tx.sort_values("Data")
    carteiras = []
    for ticker, grp in tx.groupby("Ticker", dropna=True):
        qtd = 0.0
        pm  = 0.0
        for _, r in grp.iterrows():
            q, p, s = r["Quantidade"], r["Preco"], r["Sinal"]
            if s == 1:  # compra
                custo_anterior = pm * qtd
                custo_novo = custo_anterior + (q * p) + float(r.get("Taxas", 0) or 0)
                qtd = qtd + q
                pm = custo_novo / qtd if qtd > 0 else 0.0
            else:       # venda (reduz quantidade; PM não muda)
                qtd = max(qtd - q, 0.0)
                if qtd == 0:
                    pm = 0.0
        if qtd > 0:
            carteiras.append({
                "Ticker": ticker,
                "Ativo": grp["Ativo"].dropna().iloc[-1] if "Ativo" in grp.columns else ticker,
                "Classe": grp["Classe"].dropna().iloc[-1] if "Classe" in grp.columns else None,
                "QuantidadeAtual": qtd,
                "PrecoMedio": pm,
                "Custo": qtd * pm
            })
    df = pd.DataFrame(carteiras)
    if df.empty:
        return df

    # Merge com preços atuais (se houver)
    if not cotacoes.empty:
        df = df.merge(
            cotacoes[["Ticker", "PrecoAtual"]],
            how="left", on="Ticker"
        )
        df["PrecoAtual"] = pd.to_numeric(df["PrecoAtual"], errors='coerce')
        df["ValorMercado"] = df["QuantidadeAtual"] * df["PrecoAtual"]
        df["PL_Latente"]  = df["ValorMercado"] - df["Custo"]
        df["Retorno_%"]   = (df["ValorMercado"] / df["Custo"] - 1.0) * 100
    else:
        df["PrecoAtual"] = None
        df["ValorMercado"] = None
        df["PL_Latente"] = None
        df["Retorno_%"] = None

    return df.sort_values("ValorMercado", ascending=False, na_position='last')


def proventos_mensais(pv: pd.DataFrame) -> pd.DataFrame:
    if pv.empty:
        return pd.DataFrame()
    df = pv.copy()
    df["Ano"] = df["Data"].dt.year
    df["Mes"] = df["Data"].dt.month
    grp = df.groupby(["Ano", "Mes"], dropna=False)["Valor"].sum().reset_index()
    grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str) + "-" + grp["Mes"].astype(str) + "-01")
    return grp.sort_values(["Ano", "Mes"]) 


def aportes_mensais(tx: pd.DataFrame) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame()
    df = tx[tx["Tipo"].isin(["APORTE", "RETIRADA"])].copy()
    if df.empty:
        return pd.DataFrame()
    df["Ano"] = df["Data"].dt.year
    df["Mes"] = df["Data"].dt.month
    df["ValorMov"] = df.apply(lambda r: r.get("Valor") if pd.notna(r.get("Valor")) else (r.get("Quantidade", 0) * r.get("Preco", 0)), axis=1)
    df.loc[df["Tipo"] == "RETIRADA", "ValorMov"] *= -1
    grp = df.groupby(["Ano", "Mes"], dropna=False)["ValorMov"].sum().reset_index()
    grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str) + "-" + grp["Mes"].astype(str) + "-01")
    return grp.sort_values(["Ano", "Mes"]) 


# =========================
# CARREGAMENTO
# =========================
bases = carregar_bases()
TX, PV, CT = preparar_dados(bases)

# =========================
# FILTROS GERAIS (SIDEBAR)
# =========================
with st.sidebar:
    st.header("Filtros")
    # Período
    min_data = min([d.min() for d in [TX.get("Data"), PV.get("Data")] if isinstance(d, pd.Series) and not d.empty] or [pd.to_datetime("2020-01-01")])
    max_data = max([d.max() for d in [TX.get("Data"), PV.get("Data")] if isinstance(d, pd.Series) and not d.empty] or [pd.Timestamp.today()])
    periodo = st.date_input("Período", (min_data.date(), max_data.date()))

    # Classe e ticker
    classes = sorted(list(pd.unique(pd.concat([TX.get("Classe"), PV.get("Classe")]).dropna()))) if not TX.empty or not PV.empty else []
    classe_sel = st.multiselect("Classe", options=classes, default=classes)

    tickers = sorted(list(pd.unique(pd.concat([TX.get("Ticker"), PV.get("Ticker")]).dropna()))) if not TX.empty or not PV.empty else []
    ticker_sel = st.multiselect("Ticker", options=tickers)

# Aplicar filtros
if not TX.empty:
    TX = TX[(TX["Data"].dt.date >= periodo[0]) & (TX["Data"].dt.date <= periodo[1])]
    if classe_sel:
        TX = TX[TX["Classe"].isin(classe_sel)]
    if ticker_sel:
        TX = TX[TX["Ticker"].isin(ticker_sel)]

if not PV.empty:
    PV = PV[(PV["Data"].dt.date >= periodo[0]) & (PV["Data"].dt.date <= periodo[1])]
    if classe_sel:
        PV = PV[PV["Classe"].isin(classe_sel)]
    if ticker_sel:
        PV = PV[PV["Ticker"].isin(ticker_sel)]

# =========================
# CARTEIRA ATUAL
# =========================
st.subheader("📦 Carteira Atual")
carteira_df = carteira_atual(TX, CT)
if carteira_df.empty:
    st.info("Sem dados de compras/vendas para calcular a carteira.")
else:
    c1, c2 = st.columns([2,1])
    with c1:
        st.dataframe(
            carteira_df[["Ticker", "Ativo", "Classe", "QuantidadeAtual", "PrecoMedio", "PrecoAtual", "Custo", "ValorMercado", "PL_Latente", "Retorno_%"]],
            hide_index=True,
            use_container_width=True
        )
    with c2:
        # Alocação por classe
        aloc = carteira_df.groupby("Classe", dropna=False)["ValorMercado"].sum().reset_index()
        aloc = aloc.dropna(subset=["Classe"]) if not aloc.empty else aloc
        if not aloc.empty and aloc["ValorMercado"].notna().any():
            fig = px.pie(aloc, names="Classe", values="ValorMercado", hole=0.4, template=PLOTLY_TEMPLATE, title="Alocação por Classe")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Forneça COTACOES para ver a alocação por valor de mercado.")

# =========================
# APORTES MENSAIS
# =========================
st.subheader("💸 Aportes x Retiradas (mensal)")
ap = aportes_mensais(TX)
if ap.empty:
    st.info("Sem movimentos de Aporte/Retirada no período.")
else:
    fig = px.bar(ap, x="Competencia", y="ValorMov", template=PLOTLY_TEMPLATE, title="Fluxo de Caixa Mensal")
    fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PROVENTOS MENSAIS
# =========================
st.subheader("💰 Proventos (mensal)")
prov = proventos_mensais(PV)
if prov.empty:
    st.info("Sem proventos no período.")
else:
    fig = px.bar(prov, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos por Mês")
    fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
    st.plotly_chart(fig, use_container_width=True)

    # Tabela por ticker no período
    st.caption("Proventos por Ticker no período filtrado")
    tab = PV.groupby(["Ticker", "Tipo"], dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
    st.dataframe(tab, hide_index=True, use_container_width=True)

# =========================
# PAINEL POR CLASSE
# =========================
st.subheader("🎯 Painel por Classe")
if carteira_df.empty:
    st.caption("Adicione transações de compra/venda para ver o painel por classe.")
else:
    por_classe = carteira_df.groupby("Classe", dropna=False).agg({
        "ValorMercado": "sum", "Custo": "sum"
    }).reset_index()
    if not por_classe.empty:
        por_classe["Retorno_%"] = (por_classe["ValorMercado"] / por_classe["Custo"] - 1) * 100
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(por_classe, x="Classe", y="ValorMercado", template=PLOTLY_TEMPLATE, title="Valor de Mercado por Classe")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(por_classe, x="Classe", y="Retorno_%", template=PLOTLY_TEMPLATE, title="Retorno % por Classe")
            st.plotly_chart(fig, use_container_width=True)
    st.dataframe(por_classe, use_container_width=True, hide_index=True)

# =========================
# DICAS DE USO / PRÓXIMOS PASSOS
# =========================
with st.expander("⚙️ Como preparar a planilha e evoluir para o v1.0"):
    st.markdown(
        """
        **Estrutura das abas (recomendado):**
        - **RELATORIO_TRANSACOES**: Data, Tipo (COMPRA/VENDA/APORTE/RETIRADA), Classe, Ticker, Ativo, Quantidade, Preco, Taxas, Conta, Corretora
        - **PROVENTOS**: Data, Classe, Ticker, Ativo, Tipo (DIVIDENDO/JCP/RENDIMENTO), Valor, Quantidade (opcional), Conta
        - **COTACOES** *(opcional)*: Ticker, PrecoAtual, DataCotacao

        **Notas importantes:**
        - O cálculo da **Carteira Atual** usa PM (preço médio) por ticker com base nas compras e reduz posição nas vendas.
        - Para **alocação por valor**, preencha a aba **COTACOES** (pode ser manual no começo ou importado via script).
        - Você pode quebrar este arquivo em multipáginas criando uma pasta `pages/` e separando cada seção.

        **Ideias para as próximas iterações:**
        - Rentabilidade vs. benchmark (CDI/IPCA/Ibov) — importar séries históricas.
        - Metas por categoria (ex.: 60% RF, 30% Ações, 10% FII) + alertas.
        - Registro de ordens fracionadas com taxas por nota e ajuste de PM incluindo custos.
        - Tela de **Diário de Trade** (observações por operação) e **Riscos** (volatilidade, drawdown simples).
        - Integração com B3/Yahoo p/ cotação diária (script agendado). 
        """
    )
