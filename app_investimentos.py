# app_investimentos_linkado.py — versão robusta
# - Lê com gspread usando get_all_values() e monta DataFrame
# - Detecta automaticamente a linha do cabeçalho (primeira com >=2 células não vazias)
# - Torna nomes de colunas únicos e remove vazios
# - Converte 'Data' p/ datetime (pt-BR) e números 'R$ 1.234,56' p/ float
# - Filtros seguros mesmo com colunas ausentes ou vazias
# - CSV export só como fallback

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from urllib.parse import quote

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
ABA_ATIVOS_GID      = st.secrets.get("ABA_ATIVOS_GID", "")
ABA_LANCAMENTOS_GID = st.secrets.get("ABA_LANCAMENTOS_GID", "")
ABA_PROVENTOS_GID   = st.secrets.get("ABA_PROVENTOS_GID", "")

# -------------------------------
# Helpers
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
        values = ws.get_all_values()  # lista de listas
        if not values:
            return pd.DataFrame()

        # descobre a linha de cabeçalho (primeira com >=2 células não vazias)
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
            if cnt == 0:
                headers.append(base)
            else:
                headers.append(f"{base}_{cnt+1}")
            seen[base] = cnt + 1

        data_rows = values[header_idx + 1 :]
        df = pd.DataFrame(data_rows, columns=headers)
        # remove colunas totalmente vazias e linhas vazias
        df = df.replace({"": None})
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
    # 1) GID (se a aba estiver pública)
    if sheet_id and str(gid).strip():
        df = _read_csv_by_gid(sheet_id, str(gid).strip())
        if not df.empty:
            return df
    # 2) Service Account robusto
    if sheet_id and _has_sa() and aba_nome:
        df = _read_ws_values(sheet_id, aba_nome)
        if not df.empty:
            return df
    # 3) Fallback CSV por nome
    if sheet_id and aba_nome:
        return _read_csv_by_name(sheet_id, aba_nome)
    return pd.DataFrame()

# -------------------------------
# Padronização
# -------------------------------
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
        "Valor": ["Valor","Valor (R$)","Provento R$"],
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

DF_ATIVOS = padronizar_ativos(df_ativos_raw)
TX        = padronizar_lancamentos(df_tx_raw)
PV        = padronizar_proventos(df_pv_raw)

# -------------------------------
# Filtros (seguros)
# -------------------------------
with st.sidebar:
    st.header("Filtros")

    # intervalo padrão
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

# aplica filtros de período com checagem
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
# Carteira (Meus Ativos)
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
- **Compartilhe a planilha** com `streamlit-reader@barbearia-dashboard.iam.gserviceaccount.com`.
- A leitura ignora cabeçalhos vazios e **renomeia duplicados** (`col`, `col_2`...).
- Se quiser máxima estabilidade, informe os **GIDs** das abas nos *secrets*.
- Datas são convertidas com `dayfirst=True`; valores no formato BR são normalizados.
        """
    )
