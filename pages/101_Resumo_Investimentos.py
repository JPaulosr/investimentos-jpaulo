# 101_Resumo_Investimentos.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from urllib.parse import quote
from datetime import datetime, timedelta

st.set_page_config(page_title="Proventos & Calendário", page_icon="💸", layout="wide")
st.title("💸 Proventos & Calendário")

# === CONFIG ===
SHEET_ID = "1TQBzbueeBTgNmXwZPg04GFOwNL4vh_1ZbKlDAGQJ09o"
PROVENTOS_ALVOS   = ["APP_Proventos"]
MEUS_ATIVOS_ALVOS = ["APP_MeusAtivos"]
GID_PROVENTOS   = "2109089485"
GID_MEUS_ATIVOS = None

MESES_PT = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
            7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}
COL_ORDEM_MESES = list(MESES_PT.values())

# === HELPERS ===
def csv_url_by_name(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(sheet_name)}"
def csv_url_by_gid(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def to_float_br(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    s = (s.replace("R$","").replace("US$","").replace("$","").replace("€","").replace(" ",""))
    s = s.replace(".","").replace(",",".")
    try: return float(s)
    except: return np.nan
def to_int_safe(x):
    try:
        if pd.isna(x) or str(x).strip()=="": return np.nan
        return int(float(str(x).replace(",",".").strip()))
    except: return np.nan
def to_date_br(x):
    if pd.isna(x) or str(x).strip()=="": return pd.NaT
    s = str(x).strip()
    for fmt in ("%d/%m/%Y","%Y-%m-%d","%d-%m-%Y"):
        try: return datetime.strptime(s, fmt).date()
        except: pass
    try: return pd.to_datetime(s, dayfirst=True).date()
    except: return pd.NaT
def pick_col(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in cols: return cols[k]
    for cand in candidates:
        for k, orig in cols.items():
            if k.startswith(cand.lower().strip()): return orig
    return None
def fmt_moeda(v):
    try: return "R$ " + f"{float(v):,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")
    except: return v
def realinha_cabecalho(df, procurar=("ticker","data")):
    for i in range(min(25, len(df))):
        linha = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any(procurar[0] in x for x in linha) and any(procurar[1] in x for x in linha):
            cols = df.iloc[i].tolist()
            df2 = df.iloc[i+1:].copy()
            df2.columns = cols
            return df2.dropna(how="all").dropna(axis=1, how="all")
    return df

# === LOAD BASE ===
@st.cache_data(ttl=300)
def carregar_via_service_account(sheet_id, alvo_nomes):
    try:
        import gspread
        from gspread_dataframe import get_as_dataframe
        from google.oauth2.service_account import Credentials
        info = st.secrets.get("gcp_service_account") or st.secrets.get("GCP_SERVICE_ACCOUNT")
        if not info: return None
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly",
                    "https://www.googleapis.com/auth/drive.readonly"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        def norm(s): return re.sub(r"\s+"," ", re.sub(r"[^\w]+"," ", s.lower())).strip()
        mapa = {norm(ws.title): ws for ws in sh.worksheets()}
        for nome in alvo_nomes:
            if norm(nome) in mapa:
                ws = mapa[norm(nome)]
                df = get_as_dataframe(ws, evaluate_formulas=True, dtype=str)
                return df.dropna(how="all").dropna(axis=1, how="all")
        return None
    except Exception:
        return None

@st.cache_data(ttl=300)
def carregar_via_csv(sheet_id, alvo_nomes, gid):
    last_err = None
    for nome in alvo_nomes:
        try:  return pd.read_csv(csv_url_by_name(sheet_id, nome), dtype=str)
        except Exception as e: last_err = e
    if gid:
        try:  return pd.read_csv(csv_url_by_gid(sheet_id, gid), dtype=str)
        except Exception as e: last_err = e
    raise last_err if last_err else RuntimeError("Falha ao carregar CSV")

def carregar_tabela(sheet_id, alvo_nomes, gid):
    df = carregar_via_service_account(sheet_id, alvo_nomes)
    if df is not None: return df
    return carregar_via_csv(sheet_id, alvo_nomes, gid)

# === LOAD PROVENTOS ===
@st.cache_data(ttl=300)
def load_proventos(sheet_id):
    df = carregar_tabela(sheet_id, PROVENTOS_ALVOS, GID_PROVENTOS)
    df = realinha_cabecalho(df)

    # pega total original da planilha (se existir)
    orig_tot_col = pick_col(df, ["Total Líquido R$","Total Liquido R$","Total Líquido","Total Liquido"])
    orig_tot_series = df[orig_tot_col].copy() if orig_tot_col else None

    # map/rename
    col_ticker = pick_col(df, ["Ticker"])
    col_tipo   = pick_col(df, ["Tipo Provento","Tipo","Provento"])
    col_data   = pick_col(df, ["Data"])
    col_qtd    = pick_col(df, ["Quantidade","Qtd"])
    col_unit   = pick_col(df, ["Unitário R$","Unitario R$","Unitário","Unitario","Valor por cota","Valor unitário"])
    col_irrf   = pick_col(df, ["IRRF","Imposto","Impostos"])
    col_mes    = pick_col(df, ["Mês","Mes"])
    col_ano    = pick_col(df, ["Ano"])
    col_class  = pick_col(df, ["Classe do Ativo","Classe","Classe do ativo"])

    rename = {}
    if col_ticker: rename[col_ticker] = "Ticker"
    if col_tipo:   rename[col_tipo]   = "Tipo"
    if col_data:   rename[col_data]   = "Data"
    if col_qtd:    rename[col_qtd]    = "Quantidade"
    if col_unit:   rename[col_unit]   = "Unitario_R$"
    if col_irrf:   rename[col_irrf]   = "IRRF"
    if col_mes:    rename[col_mes]    = "Mes"
    if col_ano:    rename[col_ano]    = "Ano"
    if col_class:  rename[col_class]  = "Classe"
    df = df.rename(columns=rename)

    for c in ["Ticker","Tipo","Data","Quantidade","Unitario_R$","IRRF","Mes","Ano","Classe"]:
        if c not in df.columns: df[c] = np.nan

    # conversões
    df["Data"]        = df["Data"].map(to_date_br)
    df["Quantidade"]  = df["Quantidade"].map(to_int_safe)
    df["Unitario_R$"] = df["Unitario_R$"].map(to_float_br)
    df["IRRF"]        = df["IRRF"].map(to_float_br)

    if df["Data"].notna().any():
        ano_series = df["Data"].apply(lambda d: d.year if pd.notna(d) else np.nan)
        mes_series = df["Data"].apply(lambda d: MESES_PT.get(d.month) if pd.notna(d) else np.nan)
        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").fillna(ano_series)
        df["Mes"] = df["Mes"].fillna(mes_series)

    # cálculo (sempre disponível)
    df[["Quantidade","Unitario_R$","IRRF"]] = df[["Quantidade","Unitario_R$","IRRF"]].fillna(0)
    df["Total_Liquido_R$"] = (df["Quantidade"] * df["Unitario_R$"] - df["IRRF"]).clip(lower=0)

    # total original (se existir)
    if orig_tot_series is not None:
        df["Total_Liquido_Orig_R$"] = pd.to_numeric(orig_tot_series.map(to_float_br), errors="coerce")
    else:
        df["Total_Liquido_Orig_R$"] = np.nan

    # === coluna final usada pelo app ===
    df["Total_Final_R$"] = np.where(
        df["Total_Liquido_Orig_R$"].notna(),
        df["Total_Liquido_Orig_R$"],
        df["Total_Liquido_R$"]
    )

    df = df[df["Ticker"].astype(str).str.strip() != ""].copy()
    df["Classe"] = df["Classe"].fillna("")
    return df

# === LOAD MEUS ATIVOS ===
@st.cache_data(ttl=300)
def load_meus_ativos(sheet_id):
    df = carregar_tabela(sheet_id, MEUS_ATIVOS_ALVOS, GID_MEUS_ATIVOS)
    df = realinha_cabecalho(df)

    col_ticker = pick_col(df, ["Ticker"])
    col_pct    = pick_col(df, ["% na Carteira","% na carteira"])
    col_pos    = pick_col(df, ["Posição","Posicao"])
    col_class  = pick_col(df, ["Classe"])
    col_qtd    = pick_col(df, ["Quantidade (Líquida)","Quantidade","Qtd"])
    col_pm     = pick_col(df, ["Preço Médio Ajustado (R$)","Preço Médio Ajustado","PM Ajustado"])
    col_cot    = pick_col(df, ["Cotação de Hoje (R$)","Cotação de Hoje","Cotacao de Hoje"])

    rename = {}
    if col_ticker: rename[col_ticker] = "Ticker"
    if col_pct:    rename[col_pct]    = "Pct_Carteira"
    if col_pos:    rename[col_pos]    = "Posicao"
    if col_class:  rename[col_class]  = "Classe"
    if col_qtd:    rename[col_qtd]    = "Qtd_Liquida"
    if col_pm:     rename[col_pm]     = "PM_Ajustado_R$"
    if col_cot:    rename[col_cot]    = "Cotacao_R$"
    df = df.rename(columns=rename)

    if "Pct_Carteira" in df.columns:
        df["Pct_Carteira"] = (df["Pct_Carteira"].astype(str).str.replace("%","",regex=False).map(to_float_br)/100.0)
    if "Qtd_Liquida" in df.columns:   df["Qtd_Liquida"] = df["Qtd_Liquida"].map(to_int_safe)
    if "PM_Ajustado_R$" in df.columns: df["PM_Ajustado_R$"] = df["PM_Ajustado_R$"].map(to_float_br)
    if "Cotacao_R$" in df.columns:     df["Cotacao_R$"] = df["Cotacao_R$"].map(to_float_br)

    qtd = df.get("Qtd_Liquida", pd.Series(0)).fillna(0)
    cot = df.get("Cotacao_R$",   pd.Series(0)).fillna(0)
    pm  = df.get("PM_Ajustado_R$", pd.Series(0)).fillna(0)

    if "Posicao" not in df.columns or df["Posicao"].isna().all():
        df["Posicao"] = np.where(qtd>0, "Ativa", "Encerrada")

    valor_atual     = (cot * qtd)
    valor_investido = (pm  * qtd)
    base = valor_atual.where(valor_atual>0, valor_investido)
    total_base = base.sum()
    df["Pct_Carteira"] = np.where(total_base>0, base/total_base, 0.0)

    if "Classe" not in df.columns: df["Classe"] = ""
    return df

# === DADOS ===
prov = load_proventos(SHEET_ID)
ativos = load_meus_ativos(SHEET_ID)

if st.sidebar.checkbox("🔧 Mostrar colunas detectadas (debug)"):
    st.write("Proventos:", sorted(prov.columns.tolist()))
    st.write("Meus Ativos:", sorted(ativos.columns.tolist()))

prov["Ano_num"] = pd.to_numeric(prov.get("Ano"), errors="coerce")
if prov["Ano_num"].isna().all() and "Data" in prov.columns:
    prov["Ano_num"] = prov["Data"].apply(lambda d: d.year if pd.notna(d) else np.nan)

anos_disponiveis = sorted([int(a) for a in prov["Ano_num"].dropna().unique()])
if not anos_disponiveis:
    st.error("Não encontrei anos válidos em 'Proventos'. Verifique a coluna 'Data'.")
    st.stop()

# === FILTROS ===
ano_sel    = st.sidebar.selectbox("Ano", anos_disponiveis[::-1])
classes    = ["(todas)"] + sorted([c for c in prov["Classe"].dropna().unique() if str(c).strip()!=""])
classe_sel = st.sidebar.selectbox("Classe", classes, index=0)
posicoes   = ["(todas)","Ativa","Encerrada"]
pos_sel    = st.sidebar.selectbox("Posição", posicoes, index=0)
tickers_all= ["(todos)"] + sorted(prov["Ticker"].dropna().unique())
ticker_sel = st.sidebar.selectbox("Ticker", tickers_all, index=0)

df = prov[prov["Ano_num"]==int(ano_sel)].copy()
if classe_sel!="(todas)": df = df[df["Classe"]==classe_sel]
if pos_sel!="(todas)":
    df = df.merge(ativos[["Ticker","Posicao"]], on="Ticker", how="left")
    df = df[df["Posicao"].fillna("(desconhecida)")==pos_sel]
if ticker_sel!="(todos)": df = df[df["Ticker"]==ticker_sel]

if df.empty:
    st.warning("Sem registros para os filtros escolhidos.")
    st.stop()

# === AUDITORIA ===
with st.expander("🔎 Auditoria — linhas com maior valor (cálculo vs. planilha)", expanded=False):
    c1,c2,c3 = st.columns(3)
    c1.metric("Soma usada (FINAL)", fmt_moeda(df["Total_Final_R$"].sum()))
    c2.metric("Soma planilha (orig.)", fmt_moeda(df["Total_Liquido_Orig_R$"].sum()))
    c3.metric("Soma calculada", fmt_moeda(df["Total_Liquido_R$"].sum()))
    audit = df.assign(
        Diff_final_orig = df["Total_Final_R$"] - df["Total_Liquido_Orig_R$"],
        Diff_final_calc = df["Total_Final_R$"] - df["Total_Liquido_R$"]
    )
    st.dataframe(
        audit.sort_values("Total_Final_R$", ascending=False)[
            ["Ticker","Data","Tipo","Quantidade","Unitario_R$","IRRF",
             "Total_Final_R$","Total_Liquido_Orig_R$","Total_Liquido_R$",
             "Diff_final_orig","Diff_final_calc"]
        ].head(25),
        use_container_width=True
    )

# === CALENDÁRIO ===
df["Mes"] = pd.Categorical(df["Mes"], categories=COL_ORDEM_MESES, ordered=True)
g_mes_ticker = df.groupby(["Ticker","Mes"], dropna=False)["Total_Final_R$"].sum().reset_index()
pivot = g_mes_ticker.pivot(index="Ticker", columns="Mes", values="Total_Final_R$").fillna(0.0)
pivot = pivot.reindex(columns=COL_ORDEM_MESES, fill_value=0.0)
pivot["Total no ano"] = pivot.sum(axis=1)

total_ano = pivot["Total no ano"].sum()
media_mensal = total_ano/12.0
melhor_mes_idx = pivot.drop(columns=["Total no ano"]).sum().idxmax()
melhor_mes_val = pivot.drop(columns=["Total no ano"]).sum().max()

# Metadados
meta_cols = ["Ticker","Classe","Pct_Carteira","Posicao"]
for c in ["Classe","Posicao","Pct_Carteira"]:
    if c not in ativos.columns:
        ativos[c] = "" if c!="Pct_Carteira" else 0.0
meta = ativos[meta_cols].copy()
meta["Pct_Carteira"] = meta["Pct_Carteira"].fillna(0.0)
tabela = meta.merge(pivot.reset_index(), on="Ticker", how="right").fillna(0.0)

# === YIELDS ===
dez31 = datetime(int(ano_sel),12,31).date()
ini12 = (datetime(int(ano_sel),12,31) - timedelta(days=365)).date()
prov_12m = prov[(prov["Data"].notna()) & (prov["Data"]>ini12) & (prov["Data"]<=dez31)]
agg_12m = prov_12m.groupby("Ticker")["Total_Final_R$"].sum().reset_index().rename(columns={"Total_Final_R$":"Prov_12m_R$"})
ult_unit = (df.sort_values("Data").dropna(subset=["Unitario_R$"])
              .groupby("Ticker")["Unitario_R$"].last().reset_index()
              .rename(columns={"Unitario_R$":"Ultimo_Unitario_R$"}))
yields = ativos[["Ticker","Qtd_Liquida","PM_Ajustado_R$","Cotacao_R$"]].copy()
yields = yields.merge(agg_12m, on="Ticker", how="left").merge(ult_unit, on="Ticker", how="left")
yields["Valor_Atual_R$"] = yields["Cotacao_R$"] * yields["Qtd_Liquida"]
yields["DY_12m"] = np.where(yields["Valor_Atual_R$"]>0, yields["Prov_12m_R$"]/yields["Valor_Atual_R$"], np.nan)
yields["YOC"]    = np.where(yields["PM_Ajustado_R$"]>0, yields["Ultimo_Unitario_R$"]/yields["PM_Ajustado_R$"], np.nan)
yields["Yield_mensal"] = yields["YOC"]
tabela = tabela.merge(yields[["Ticker","DY_12m","YOC","Yield_mensal"]], on="Ticker", how="left")

# === UI ===
c1,c2,c3 = st.columns(3)
c1.metric("Total no ano", fmt_moeda(total_ano))
c2.metric("Média mensal (12 meses)", fmt_moeda(media_mensal))
c3.metric(f"Melhor mês ({melhor_mes_idx})", fmt_moeda(melhor_mes_val))

st.markdown("### Calendário por Ticker (Jan–Dez)")
tabela_view = tabela.copy()
for col in COL_ORDEM_MESES + ["Total no ano"]:
    if col in tabela_view.columns: tabela_view[col] = tabela_view[col].map(fmt_moeda)
for col in ["Pct_Carteira","DY_12m","YOC","Yield_mensal"]:
    if col in tabela_view.columns:
        tabela_view[col] = tabela_view[col].apply(lambda x: "-" if pd.isna(x) else f"{100*float(x):.2f}%")
st.dataframe(tabela_view, use_container_width=True, hide_index=True)

st.markdown("### Ranking de Proventos no Ano")
ranking = pivot["Total no ano"].sort_values(ascending=False).reset_index()
ranking.columns = ["Ticker","Total no ano (R$)"]
ranking["Total no ano (R$)"] = ranking["Total no ano (R$)"].map(fmt_moeda)
st.dataframe(ranking, use_container_width=True, hide_index=True)

# === DOWNLOAD (openpyxl -> CSV fallback) ===
def to_excel_or_csv_bytes(df):
    try:
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Calendario")
        return out.getvalue(), "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        return df.to_csv(index=False).encode("utf-8"), "csv", "text/csv"

data_bytes, ext, mime = to_excel_or_csv_bytes(tabela)
st.download_button(f"⬇️ Baixar Calendário ({ext.upper()})",
                   data=data_bytes,
                   file_name=f"calendario_proventos_{ano_sel}.{ext}",
                   mime=mime)

st.download_button("⬇️ Baixar Ranking (CSV)",
                   data=ranking.to_csv(index=False).encode("utf-8"),
                   file_name=f"ranking_proventos_{ano_sel}.csv",
                   mime="text/csv")

st.caption("O app usa **Total_Final_R$** (planilha se existir; senão Qtd×Unitário−IRRF). Use a Auditoria para comparar com o total original.")
