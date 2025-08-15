# app_investimentos_linkado.py — versão robusta e completa
# --------------------------------------------------------
# - Lê via Service Account (gspread) como prioridade
# - Fallback para CSV por GID ou por NOME (se a planilha estiver pública)
# - Diagnóstico mostra e-mail da SA, lista de abas e "como cada aba foi lida"
# - Padroniza colunas (datas BR e números BR)
# - Filtros seguros + gráficos (carteira, aportes e proventos)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import numpy as np

st.set_page_config(page_title="📈 Investimentos – Linkado ao Google Sheets",
                   page_icon="📈", layout="wide")
st.title("📈 Painel de Investimentos – Linkado ao Google Sheets")
PLOTLY_TEMPLATE = "plotly_dark"

# ======================================================
# Secrets / Config (pode definir aqui ou em st.secrets)
# ======================================================
SHEET_ID = st.secrets.get("SHEET_ID", "").strip()

# nomes das abas (padrão compatível com sua planilha)
ABA_ATIVOS      = st.secrets.get("ABA_ATIVOS", "1. Meus Ativos")
ABA_LANCAMENTOS = st.secrets.get("ABA_LANCAMENTOS", "2. Lançamentos (B3)")
ABA_PROVENTOS   = st.secrets.get("ABA_PROVENTOS", "3. Proventos")

# GIDs opcionais (apenas se quiser usar CSV público como fallback)
ABA_ATIVOS_GID      = str(st.secrets.get("ABA_ATIVOS_GID", "")).strip()
ABA_LANCAMENTOS_GID = str(st.secrets.get("ABA_LANCAMENTOS_GID", "")).strip()
ABA_PROVENTOS_GID   = str(st.secrets.get("ABA_PROVENTOS_GID", "")).strip()

# ======================================================
# Helpers de conversão
# ======================================================
def br_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
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

# ======================================================
# Service Account (SA) utils + leitores com patch
# ======================================================
def _has_sa():
    return bool(st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account"))

def _get_sa_info():
    return st.secrets.get("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account") or {}

def _read_ws_values(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    """Lê pelo gspread usando Service Account e monta DataFrame robusto."""
    import gspread
    from google.oauth2.service_account import Credentials

    info = _get_sa_info()
    if not info:
        raise RuntimeError("Service Account ausente nos secrets.")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    # tenta por nome exato; se falhar, aproxima case-insensitive
    try:
        ws = sh.worksheet(aba_nome)
    except Exception:
        titles = [w.title for w in sh.worksheets()]
        match = next((t for t in titles if t.casefold() == aba_nome.casefold()), None)
        if not match:
            raise
        ws = sh.worksheet(match)

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    # primeira linha com >=2 células não vazias vira cabeçalho
    header_idx = None
    for i, row in enumerate(values):
        if sum(1 for c in row if str(c).strip()) >= 2:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    headers_raw = [h.strip() for h in values[header_idx]]
    seen, headers = {}, []
    for h in headers_raw:
        base = h if h else "col"
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base] == 1 else f"{base}_{seen[base]}")

    df = pd.DataFrame(values[header_idx + 1 :], columns=headers)
    df = df.replace({"": None}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df

def _read_csv_by_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    import urllib.error
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        # 401/403 => planilha não pública para CSV
        if e.code in (401, 403):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

def _read_csv_by_name(sheet_id: str, aba_nome: str) -> pd.DataFrame:
    import urllib.error
    from urllib.parse import quote
    try:
        aba_enc = quote(aba_nome, safe="")
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={aba_enc}"
        return pd.read_csv(url)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def ler_aba(sheet_id: str, aba_nome: str, gid: str = "") -> pd.DataFrame:
    """
    Ordem:
      1) Service Account (robusto, não exige público)
      2) CSV por GID (se público)
      3) CSV por NOME (se público)
    """
    # 1) SA primeiro
    if sheet_id and _has_sa():
        try:
            df = _read_ws_values(sheet_id, aba_nome)
            if not df.empty:
                st.session_state.setdefault("_como_leu", {})[aba_nome] = "service_account"
                return df
        except Exception as e:
            st.info(f"[SA] tentativa em '{aba_nome}' falhou: {e}")

    # 2) CSV por GID (só se público)
    if sheet_id and gid:
        df = _read_csv_by_gid(sheet_id, gid)
        if not df.empty:
            st.session_state.setdefault("_como_leu", {})[aba_nome] = "csv_gid"
            return df

    # 3) CSV por NOME (só se público)
    if sheet_id and aba_nome:
        df = _read_csv_by_name(sheet_id, aba_nome)
        if not df.empty:
            st.session_state.setdefault("_como_leu", {})[aba_nome] = "csv_nome"
            return df

    st.session_state.setdefault("_como_leu", {})[aba_nome] = "falhou"
    return pd.DataFrame()

# ======================================================
# Diagnóstico de conexão (mostra SA e abas)
# ======================================================
with st.expander("🧪 Diagnóstico de Conexão", expanded=False):
    st.write("**SHEET_ID:**", SHEET_ID or "(vazio)")
    if _has_sa():
        info = _get_sa_info()
        st.write("**Service Account:**", info.get("client_email", "(sem client_email nos secrets)"))
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            creds = Credentials.from_service_account_info(info, scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
            ])
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(SHEET_ID)
            titles = [w.title for w in sh.worksheets()]
            st.success("Conexão OK. Abas encontradas: " + ", ".join(titles))
        except Exception as e:
            st.error(f"Não consegui listar abas via SA: {e}")
    else:
        st.warning("Service Account ausente nos secrets (GCP_SERVICE_ACCOUNT / gcp_service_account).")

# ======================================================
# Carregamento das abas
# ======================================================
if not SHEET_ID:
    st.error("❌ `SHEET_ID` não definido nos secrets.")
    st.stop()

with st.spinner("Carregando dados da planilha..."):
    df_ativos_raw = ler_aba(SHEET_ID, ABA_ATIVOS, ABA_ATIVOS_GID)
    df_tx_raw     = ler_aba(SHEET_ID, ABA_LANCAMENTOS, ABA_LANCAMENTOS_GID)
    df_pv_raw     = ler_aba(SHEET_ID, ABA_PROVENTOS, ABA_PROVENTOS_GID)

# ======================================================
# Padronizações
# ======================================================
def padronizar_ativos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            on = (o.lower().strip()
                    .replace("ç","c").replace("ã","a").replace("õ","o")
                    .replace("é","e").replace("ê","e"))
            for k,v in cols.items():
                kn = (k.replace("ç","c").replace("ã","a").replace("õ","o")
                        .replace("é","e").replace("ê","e"))
                if kn == on:
                    return v
        return None
    mapa = {
        "Ticker": pick("ticker"),
        "%NaCarteira": pick("% na carteira","percentual na carteira"),
        "Quantidade": pick("quantidade (liquida)","quantidade","qtd"),
        "PrecoMedioCompra": pick("preco medio (compra r$)","preco medio compra r$","preco medio"),
        "PrecoMedioAjustado": pick("preco medio ajustado (r$)","preco medio ajustado"),
        "CotacaoHojeBRL": pick("cotacao de hoje (r$)","cotacao hoje r$","cotacao r$"),
        "CotacaoHojeUSD": pick("cotacao de hoje (us$)","cotacao hoje us$"),
        "ValorInvestido": pick("valor investido"),
        "ValorAtual": pick("valor atual"),
        "ProventosMes": pick("proventos (do mes)","proventos do mes"),
        "ProventosAnterior": pick("proventos (anterior)","proventos anterior"),
        "ProventosProjetado": pick("proventos (projetado)","proventos projetado"),
        "Classe": pick("classe","classe do ativo","tipo"),
    }
    out = pd.DataFrame({k: (df[v] if v in df.columns else None) for k,v in mapa.items()})
    for col in ["%NaCarteira","Quantidade","PrecoMedioCompra","PrecoMedioAjustado",
                "CotacaoHojeBRL","CotacaoHojeUSD","ValorInvestido","ValorAtual",
                "ProventosMes","ProventosAnterior","ProventosProjetado"]:
        out[col] = out[col].map(br_to_float)
    return out

def padronizar_lancamentos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Classe": ["Classe","Classe do Ativo","Tipo de Ativo"],
        "Ticker": ["Ticker"],
        "Data": ["Data","Data (DD/MM/YYYY)","Data da Operação"],
        "Tipo": ["Tipo","Tipo de Operação","Operação"],
        "Quantidade": ["Quantidade","Qtd"],
        "Preco": ["Preço (por unidade)","Preço","Preco"],
        "Taxas": ["Taxa","Taxas"],
        "IRRF": ["IRRF"],
        "TotalOperacao": ["Total da Operação","Total da Operacao","Valor Bruto"],
        "Mes": ["Mês","Mes"],
        "Ano": ["Ano"],
    }
    out = pd.DataFrame()
    for novo, cands in poss.items():
        col = next((c for c in cands if c in df.columns), None)
        out[novo] = df[col] if col else None
    out["Data"] = to_datetime_br(out["Data"])
    for col in ["Quantidade","Preco","Taxas","IRRF","TotalOperacao"]:
        out[col] = out[col].map(br_to_float)
    if "Tipo" in out.columns:
        out["Tipo"] = out["Tipo"].astype(str).str.upper().str.strip()
        out["Tipo"] = out["Tipo"].replace({"COMPRA":"COMPRA","VENDA":"VENDA","APORTE":"APORTE","RETIRADA":"RETIRADA"})
    return out

def padronizar_proventos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    poss = {
        "Data": ["Data"],
        "Ticker": ["Ticker"],
        "Tipo": ["Tipo","Tipo de Provento"],
        "Valor": ["Valor","Valor (R$)","Provento R$","Total Líquido","Total Liquido R$","Total"],
        "Classe": ["Classe","Classe do Ativo"],
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

DF_ATIVOS = padronizar_ativos(df_ativos_raw)
TX        = padronizar_lancamentos(df_tx_raw)
PV        = padronizar_proventos(df_pv_raw)

# ======================================================
# Filtros (seguros)
# ======================================================
with st.sidebar:
    st.header("Filtros")

    series_datas = []
    for s in [TX.get("Data") if isinstance(TX, pd.DataFrame) else None,
              PV.get("Data") if isinstance(PV, pd.DataFrame) else None]:
        if isinstance(s, pd.Series) and not s.empty:
            s = s.dropna()
            if not s.empty:
                series_datas.append((s.min(), s.max()))
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

# aplica filtros
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

# ======================================================
# Carteira (Meus Ativos)
# ======================================================
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

# ======================================================
# Aportes x Retiradas
# ======================================================
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

# ======================================================
# Proventos
# ======================================================
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

# ======================================================
# Dicas e modo de leitura
# ======================================================
with st.expander("⚙️ Ajustes e dicas"):
    st.markdown(
        """
- **Compartilhe a planilha** com a *Service Account* (Reader/Editor).
- Leitura prioriza **Service Account**; CSV (GID/NOME) só se a planilha for pública.
- Datas: `dayfirst=True`; números BR normalizados (R$, milhar, vírgula).
- Informe **GIDs** nos *secrets* apenas se quiser forçar CSV.
        """
    )
    if "_como_leu" in st.session_state:
        st.write("**Modo de leitura por aba:**", st.session_state["_como_leu"])
