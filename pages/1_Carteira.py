# app_investimentos_linkado.py — versão robusta + patches
# - Lê com gspread (Service Account) de forma resiliente a cabeçalhos bagunçados
# - Detecta linha de cabeçalho, renomeia duplicados e remove colunas/linhas vazias
# - Mapeia nomes ignorando acentos/pontuação e variações ("Quantidade (Líquida)", etc.)
# - Remove "linhas mortas" (ticker vazio e tudo numérico vazio)
# - Converte datas pt-BR e números "R$ 1.234,56" -> float
# - Filtros seguros e gráficos
# - Expander com diagnóstico das colunas lidas em cada aba

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from urllib.parse import quote
import re, unicodedata

st.set_page_config(page_title="📈 Investimentos – Linkado ao Google Sheets",
                   page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")
PLOTLY_TEMPLATE = "plotly_dark"

# -------------------------------
# Secrets / Config
# -------------------------------
SHEET_ID = st.secrets.get("SHEET_ID", "")
ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_LANCAMENTOS = st.secrets.get("ABA_LANCAMENTOS", "2. Lançamentos (B3)")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

# (opcionais) GIDs – se preenchidos, o CSV via gid é tentado primeiro
ABA_ATIVOS_GID      = st.secrets.get("ABA_ATIVOS_GID", "")
ABA_LANCAMENTOS_GID = st.secrets.get("ABA_LANCAMENTOS_GID", "")
ABA_PROVENTOS_GID   = st.secrets.get("ABA_PROVENTOS_GID", "")

# -------------------------------
# Helpers de normalização
# -------------------------------
def br_to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-", "--"}:
        return None
    s = (s.replace("R$", "").replace("US$", "").replace("$", "")
           .replace("%", "").replace(" ", ""))
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def to_datetime_br(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def moeda_br(v):
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)   # tudo que nao for letra/numero -> espaço
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick(cols: pd.Index, *cands) -> str | None:
    """Escolhe a coluna em 'cols' que melhor casa com qualquer candidato em cands (normalizado)."""
    norm_map = {_norm(c): c for c in cols}
    for c in cands:
        nc = _norm(c)
        if nc in norm_map:
            return norm_map[nc]
    # fallback: começa com
    for c in cands:
        nc = _norm(c)
        for k, orig in norm_map.items():
            if k.startswith(nc):
                return orig
    return None

def _has_sa():
    return bool(st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account"))

# -------------------------------
# Leitura robusta (gspread -> values)
# -------------------------------
def _read_ws_values(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    """Lê a worksheet pelo nome e cria DataFrame robusto, corrigindo cabeçalho duplicado/vazio."""
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
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()

        # detecta a primeira linha com >=2 células não vazias como cabeçalho
        header_idx = None
        for i, row in enumerate(values):
            non_empty = [c for c in row if str(c).strip() != ""]
            if len(non_empty) >= 2:
                header_idx = i
                break
        if header_idx is None:
            return pd.DataFrame()

        headers_raw = [h.strip() for h in values[header_idx]]
        # normaliza cabeçalhos vazios e duplica com sufixo
        seen = {}
        headers = []
        for h in headers_raw:
            base = h if h else "col"
            cnt = seen.get(base, 0)
            headers.append(base if cnt == 0 else f"{base}_{cnt+1}")
            seen[base] = cnt + 1

        data_rows = values[header_idx + 1 :]
        df = pd.DataFrame(data_rows, columns=headers)
        df = df.replace({"": None})
        # remove colunas totalmente vazias e linhas totalmente vazias
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        return df
    except Exception as e:
        st.warning(f"[gspread] Falha lendo '{aba_nome}': {e}")
        return pd.DataFrame()

def _read_csv_by_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"[csv-export] Falha lendo gid={gid}: {e}")
        return pd.DataFrame()

def _read_csv_by_name(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    try:
        aba_enc = quote(aba_nome, safe="")
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={aba_enc}"
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"[csv-export] Falha lendo '{aba_nome}': {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def ler_aba(sheet_id: str, aba_nome: str, gid: str = "") -> pd.DataFrame:
    # 1) GID (se você preencher e a aba for pública)
    if sheet_id and str(gid).strip():
        df = _read_csv_by_gid(sheet_id, str(gid).strip())
        if not df.empty:
            return df
    # 2) Service Account robusto (recomendado)
    if sheet_id and _has_sa() and aba_nome:
        df = _read_ws_values(sheet_id, aba_nome)
        if not df.empty:
            return df
    # 3) Fallback CSV por nome (pede aba pública)
    if sheet_id and aba_nome:
        return _read_csv_by_name(sheet_id, aba_nome)
    return pd.DataFrame()

# -------------------------------
# Padronizações (com _pick e remoção de linhas mortas)
# -------------------------------
def padronizar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    mapa = {
        "Ticker": _pick(df.columns, "Ticker"),
        "%NaCarteira": _pick(df.columns, "% na carteira", "percentual na carteira"),
        "Quantidade": _pick(df.columns, "Quantidade (Líquida)", "Quantidade", "Qtd"),
        "PrecoMedioCompra": _pick(df.columns, "Preço Médio (Compra R$)", "Preço Médio Compra R$", "Preço Médio"),
        "PrecoMedioAjustado": _pick(df.columns, "Preço Médio Ajustado (R$)", "Preço Médio Ajustado"),
        "CotacaoHojeBRL": _pick(df.columns, "Cotação de Hoje (R$)", "Cotação Hoje R$", "Cotação R$"),
        "CotacaoHojeUSD": _pick(df.columns, "Cotação de Hoje (US$)", "Cotação Hoje US$"),
        "ValorInvestido": _pick(df.columns, "Valor Investido"),
        "ValorAtual": _pick(df.columns, "Valor Atual"),
        "ProventosMes": _pick(df.columns, "Proventos (do mês)", "Proventos do mês"),
        "ProventosAnterior": _pick(df.columns, "Proventos (anterior)", "Proventos anterior"),
        "ProventosProjetado": _pick(df.columns, "Proventos (projetado)", "Proventos projetado"),
        "Classe": _pick(df.columns, "Classe", "Classe do Ativo", "Tipo"),
    }

    out = pd.DataFrame({k: (df[v] if v in df.columns else None) for k, v in mapa.items()})

    # conversões numéricas
    for col in ["%NaCarteira","Quantidade","PrecoMedioCompra","PrecoMedioAjustado",
                "CotacaoHojeBRL","CotacaoHojeUSD","ValorInvestido","ValorAtual",
                "ProventosMes","ProventosAnterior","ProventosProjetado"]:
        out[col] = out[col].map(br_to_float)

    # remove linhas mortas (ticker vazio/“-” E tudo numérico vazio)
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    numeric_cols = ["%NaCarteira","Quantidade","PrecoMedioCompra","PrecoMedioAjustado",
                    "CotacaoHojeBRL","CotacaoHojeUSD","ValorInvestido","ValorAtual",
                    "ProventosMes","ProventosAnterior","ProventosProjetado"]
    mask_all_nan = out[numeric_cols].isna().all(axis=1)
    mask_bad_ticker = out["Ticker"].isin(["", "None", "nan", "-"])
    out = out[~(mask_bad_ticker & mask_all_nan)].reset_index(drop=True)

    return out

def padronizar_lancamentos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    poss = {
        "Classe": _pick(df.columns, "Classe", "Classe do Ativo", "Tipo de Ativo"),
        "Ticker": _pick(df.columns, "Ticker"),
        "Data": _pick(df.columns, "Data", "Data da Operação", "Data (DD/MM/YYYY)"),
        "Tipo": _pick(df.columns, "Tipo", "Tipo de Operação", "Operação"),
        "Quantidade": _pick(df.columns, "Quantidade", "Qtd"),
        "Preco": _pick(df.columns, "Preço (por unidade)", "Preço", "Preco"),
        "Taxas": _pick(df.columns, "Taxa", "Taxas"),
        "IRRF": _pick(df.columns, "IRRF"),
        "TotalOperacao": _pick(df.columns, "Total da Operação (R$)", "Total da Operação",
                               "Total da Operacao", "Valor Bruto"),
        "Mes": _pick(df.columns, "Mês", "Mes"),
        "Ano": _pick(df.columns, "Ano"),
    }

    out = pd.DataFrame({k: (df[v] if v else None) for k, v in poss.items()})
    out["Data"] = to_datetime_br(out["Data"])
    for col in ["Quantidade","Preco","Taxas","IRRF","TotalOperacao"]:
        out[col] = out[col].map(br_to_float)
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()

    # remove linhas sem Data e sem Ticker e sem valores
    num_cols = ["Quantidade","Preco","Taxas","IRRF","TotalOperacao"]
    dead = out["Data"].isna() & (out["Ticker"].astype(str).str.strip().isin(["","None","nan","-"])) & out[num_cols].isna().all(axis=1)
    out = out[~dead].reset_index(drop=True)

    return out

def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Data": _pick(df.columns, "Data"),
        "Ticker": _pick(df.columns, "Ticker"),
        "Tipo": _pick(df.columns, "Tipo", "Tipo de Provento"),
        "Valor": _pick(df.columns, "Valor (R$)", "Valor", "Provento R$"),
        "Classe": _pick(df.columns, "Classe", "Classe do Ativo"),
    }
    out = pd.DataFrame({k: (df[v] if v else None) for k, v in poss.items()})
    out["Data"] = to_datetime_br(out["Data"])
    out["Valor"] = out["Valor"].map(br_to_float)
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()

    # remove linhas sem data e sem valor
    dead = out["Data"].isna() & (out["Valor"].isna() | (out["Valor"] == 0))
    out = out[~dead].reset_index(drop=True)

    return out

# -------------------------------
# Carregamento
# -------------------------------
if not SHEET_ID:
    st.error("❌ `SHEET_ID` não definido nos secrets.")
    st.stop()

with st.spinner("Carregando dados da planilha..."):
    df_ativos_raw = ler_aba(SHEET_ID, ABA_ATIVOS, ABA_ATIVOS_GID)
    df_tx_raw     = ler_aba(SHEET_ID, ABA_LANCAMENTOS, ABA_LANCAMENTOS_GID)
    df_pv_raw     = ler_aba(SHEET_ID, ABA_PROVENTOS, ABA_PROVENTOS_GID)

# Diagnóstico (ajuda a conferir nomes das colunas recebidas)
with st.expander("🔎 Diagnóstico das abas (colunas lidas)"):
    st.write("Ativos – colunas:", list(df_ativos_raw.columns))
    st.write("Lançamentos – colunas:", list(df_tx_raw.columns))
    st.write("Proventos – colunas:", list(df_pv_raw.columns))

DF_ATIVOS = padronizar_ativos(df_ativos_raw)
TX        = padronizar_lancamentos(df_tx_raw)
PV        = padronizar_proventos(df_pv_raw)

# -------------------------------
# Filtros – Sidebar
# -------------------------------
with st.sidebar:
    st.header("Filtros")

    # intervalo padrão pelas datas existentes
    series_datas = []
    for s in [TX.get("Data") if isinstance(TX, pd.DataFrame) else None,
              PV.get("Data") if isinstance(PV, pd.DataFrame) else None]:
        if isinstance(s, pd.Series) and not s.empty:
            s2 = s.dropna()
            if not s2.empty:
                series_datas.append((s2.min(), s2.max()))
    min_data = (min(s[0] for s in series_datas).date()
                if series_datas else date(2020,1,1))
    max_data = (max(s[1] for s in series_datas).date()
                if series_datas else date.today())

    periodo = st.date_input("Período", value=(min_data, max_data),
                            min_value=min_data, max_value=max_data)

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
if isinstance(periodo, tuple) and len(periodo) == 2:
    d0, d1 = periodo
else:
    d0, d1 = min_data, max_data

if not TX.empty and "Data" in TX.columns:
    TX = TX[TX["Data"].notna()]
    TX = TX[(TX["Data"].dt.date >= d0) & (TX["Data"].dt.date <= d1)]
if not PV.empty and "Data" in PV.columns:
    PV = PV[PV["Data"].notna()]
    PV = PV[(PV["Data"].dt.date >= d0) & (PV["Data"].dt.date <= d1)]

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

# -------------------------------
# Carteira
# -------------------------------
st.subheader("📦 Carteira Atual (aba 'Meus Ativos')")
if DF_ATIVOS.empty:
    st.info("Sem dados na aba de ativos (confira permissão/GID/nomes).")
else:
    v_investido = float(pd.Series(DF_ATIVOS.get("ValorInvestido", pd.Series(dtype=float))).sum(skipna=True) or 0)
    v_atual     = float(pd.Series(DF_ATIVOS.get("ValorAtual",     pd.Series(dtype=float))).sum(skipna=True) or 0)
    pl          = v_atual - v_investido
    c1,c2,c3 = st.columns(3)
    c1.metric("Valor Investido", moeda_br(v_investido))
    c2.metric("Valor Atual",     moeda_br(v_atual))
    c3.metric("P/L Latente",     moeda_br(pl))

    st.dataframe(DF_ATIVOS, use_container_width=True)

    if "ValorAtual" in DF_ATIVOS.columns and DF_ATIVOS["ValorAtual"].notna().any():
        aloc_ticker = DF_ATIVOS.groupby("Ticker", dropna=False)["ValorAtual"].sum().reset_index()
        if not aloc_ticker.empty:
            st.plotly_chart(
                px.pie(aloc_ticker, names="Ticker", values="ValorAtual", hole=0.4,
                       template=PLOTLY_TEMPLATE, title="Alocação por Ticker (Valor Atual)"),
                use_container_width=True
            )

    if all(c in DF_ATIVOS.columns for c in ["Classe","ValorAtual"]):
        aloc_classe = DF_ATIVOS.dropna(subset=["Classe"]).groupby("Classe")["ValorAtual"].sum().reset_index()
        if not aloc_classe.empty:
            fig = px.bar(aloc_classe, x="Classe", y="ValorAtual", template=PLOTLY_TEMPLATE,
                         title="Alocação por Classe (Valor Atual)")
            fig.update_layout(yaxis_title="R$")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Aportes x Retiradas
# -------------------------------
st.subheader("💸 Aportes x Retiradas (mensal)")
if TX.empty:
    st.caption("Sem dados em '2. Lançamentos (B3)'.")
else:
    mov = TX.copy()
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
        grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str)+"-"+grp["Mes"].astype(str)+"-01")
        fig = px.bar(grp, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE,
                     title="Fluxo de Caixa Mensal (Aportes líquidos)")
        fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver lançamentos filtrados"):
        st.dataframe(TX.sort_values("Data"), use_container_width=True)

# -------------------------------
# Proventos
# -------------------------------
st.subheader("💰 Proventos (mensal)")
if PV.empty:
    st.caption("Sem dados em '3. Proventos'.")
else:
    pv = PV.dropna(subset=["Data"]).copy()
    if pv.empty:
        st.caption("Registros de proventos sem data.")
    else:
        grp = pv.assign(Ano=pv["Data"].dt.year, Mes=pv["Data"].dt.month)
        grp = grp.groupby(["Ano","Mes"], dropna=False)["Valor"].sum().reset_index()
        grp["Competencia"] = pd.to_datetime(grp["Ano"].astype(str)+"-"+grp["Mes"].astype(str)+"-01")
        fig = px.bar(grp, x="Competencia", y="Valor", template=PLOTLY_TEMPLATE, title="Proventos por Mês")
        fig.update_layout(xaxis_title="Competência", yaxis_title="R$")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Proventos por Ticker no período filtrado")
        tab = pv.groupby(["Ticker","Tipo"], dropna=False)["Valor"].sum().reset_index().sort_values("Valor", ascending=False)
        st.dataframe(tab, hide_index=True, use_container_width=True)

    with st.expander("Ver proventos filtrados"):
        st.dataframe(PV.sort_values("Data"), use_container_width=True)

# -------------------------------
# Dicas
# -------------------------------
with st.expander("⚙️ Ajustes e dicas"):
    st.markdown(
        """
- **Compartilhe a planilha** com `streamlit-reader@barbearia-dashboard.iam.gserviceaccount.com` (Leitor).
- A leitura ignora cabeçalhos vazios e **renomeia duplicados** (`col`, `col_2`...).
- Para estabilidade extra, informe os **GIDs** das abas nos *secrets*.
- Datas usam `dayfirst=True`; valores BR são normalizados.
        """
    )
