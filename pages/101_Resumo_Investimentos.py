# 3_Proventos_e_Calendario.py
# MVP Proventos + Calendário — conectado às abas "3. Proventos" e "1. Meus Ativos"

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from urllib.parse import quote
from datetime import datetime, timedelta

st.set_page_config(page_title="Proventos & Calendário", page_icon="💸", layout="wide")
st.title("💸 Proventos & Calendário")

# =========================
# CONFIG — edite aqui se precisar
# =========================
SHEET_ID = "1p9IzDr-5ZV0phUHfNA_9d5xNvZW1IRo84LA__JyiiQc"  # sua planilha de investimentos
ABA_PROVENTOS = "3. Proventos"
ABA_MEUS_ATIVOS = "1. Meus Ativos"

PLOTLY_TEMPLATE = "plotly_dark"

# =========================
# HELPERS
# =========================
MESES_PT = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril",
    5: "maio", 6: "junho", 7: "julho", 8: "agosto",
    9: "setembro", 10: "outubro", 11: "novembro", 12: "dezembro"
}
COL_ORDEM_MESES = list(MESES_PT.values())

def csv_url(sheet_id: str, sheet_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(sheet_name)}"

def to_float_br(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan
    # remove símbolo de moeda e espaços
    s = s.replace("R$", "").replace("$", "").replace("US$", "").replace("€", "").strip()
    # pt-BR: separador milhar "." e decimal ","
    # mas cuide para casos já com ponto decimal
    # 1.234,56 -> 1234.56 ; 1234,56 -> 1234.56 ; 1234.56 -> 1234.56
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def to_int_safe(x):
    try:
        if pd.isna(x) or x == "":
            return np.nan
        return int(float(str(x).replace(",", ".").strip()))
    except Exception:
        return np.nan

def to_date_br(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    s = str(x).strip()
    # aceita dd/mm/aaaa ou aaaa-mm-dd
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # fallback: tenta pandas
    try:
        return pd.to_datetime(s, dayfirst=True).date()
    except Exception:
        return pd.NaT

def pick_col(df: pd.DataFrame, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in cols:
            return cols[k]
    # tenta aproximação por começa-com
    for cand in candidates:
        for key in cols:
            if key.startswith(cand.lower().strip()):
                return cols[key]
    return None

# =========================
# CARREGAMENTO
# =========================
@st.cache_data(ttl=300)
def load_csv(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    url = csv_url(sheet_id, sheet_name)
    return pd.read_csv(url, dtype=str)  # lemos tudo como string; limpamos depois

@st.cache_data(ttl=300)
def load_proventos(sheet_id: str) -> pd.DataFrame:
    df = load_csv(sheet_id, ABA_PROVENTOS)
    # mapeia colunas esperadas
    col_ticker    = pick_col(df, ["Ticker"])
    col_tipo      = pick_col(df, ["Tipo Provento", "Tipo", "Provento"])
    col_data      = pick_col(df, ["Data"])
    col_qtd       = pick_col(df, ["Quantidade"])
    col_unit      = pick_col(df, ["Unitário R$", "Unitario R$", "Unitário", "Unitario"])
    col_tot_liq   = pick_col(df, ["Total Líquido R$", "Total Liquido R$", "Total Líquido", "Total Liquido"])
    col_irrf      = pick_col(df, ["IRRF"])
    col_ptax      = pick_col(df, ["PTAX"])
    col_tot_bruto = pick_col(df, ["Total Bruto", "Total Bruto R$"])
    col_mes       = pick_col(df, ["Mês", "Mes"])
    col_ano       = pick_col(df, ["Ano"])
    col_classe    = pick_col(df, ["Classe do Ativo", "Classe"])

    rename = {}
    if col_ticker:    rename[col_ticker] = "Ticker"
    if col_tipo:      rename[col_tipo] = "Tipo"
    if col_data:      rename[col_data] = "Data"
    if col_qtd:       rename[col_qtd] = "Quantidade"
    if col_unit:      rename[col_unit] = "Unitario_R$"
    if col_tot_liq:   rename[col_tot_liq] = "Total_Liquido_R$"
    if col_irrf:      rename[col_irrf] = "IRRF"
    if col_ptax:      rename[col_ptax] = "PTAX"
    if col_tot_bruto: rename[col_tot_bruto] = "Total_Bruto_R$"
    if col_mes:       rename[col_mes] = "Mes"
    if col_ano:       rename[col_ano] = "Ano"
    if col_classe:    rename[col_classe] = "Classe"

    df = df.rename(columns=rename)

    # limpeza
    if "Data" in df.columns:
        df["Data"] = df["Data"].map(to_date_br)
    if "Quantidade" in df.columns:
        df["Quantidade"] = df["Quantidade"].map(to_int_safe)
    for c in ["Unitario_R$", "Total_Liquido_R$", "Total_Bruto_R$", "IRRF", "PTAX"]:
        if c in df.columns:
            df[c] = df[c].map(to_float_br)

    # mês/ano derivados, se necessário
    if "Data" in df.columns:
        df["Ano"] = df.get("Ano", pd.Series(dtype=object))
        df.loc[df["Ano"].isna(), "Ano"] = pd.Series([d.year if pd.notna(d) else np.nan for d in df["Data"]])
        df["Mes"] = df.get("Mes", pd.Series(dtype=object))
        df.loc[df["Mes"].isna(), "Mes"] = pd.Series([MESES_PT.get(d.month) if pd.notna(d) else np.nan for d in df["Data"]])

    # calcula Total Líquido se vier vazio (Qtd*Unitário - IRRF)
    if "Total_Liquido_R$" not in df.columns:
        df["Total_Liquido_R$"] = np.nan
    mask_calc = df["Total_Liquido_R$"].isna() & df["Quantidade"].notna() & df["Unitario_R$"].notna()
    df.loc[mask_calc, "Total_Liquido_R$"] = df.loc[mask_calc, "Quantidade"] * df.loc[mask_calc, "Unitario_R$"]
    if "IRRF" in df.columns:
        df.loc[mask_calc & df["IRRF"].notna(), "Total_Liquido_R$"] -= df.loc[mask_calc & df["IRRF"].notna(), "IRRF"]

    # normaliza classe vazia
    if "Classe" not in df.columns:
        df["Classe"] = ""

    # remove linhas totalmente vazias
    df = df[~df.get("Ticker", "").isna()]
    df = df[df.get("Ticker", "").astype(str).str.strip() != ""]
    return df

@st.cache_data(ttl=300)
def load_meus_ativos(sheet_id: str) -> pd.DataFrame:
    df = load_csv(sheet_id, ABA_MEUS_ATIVOS)

    col_ticker   = pick_col(df, ["Ticker"])
    col_pct_cart = pick_col(df, ["% na Carteira", "% na carteira"])
    col_posicao  = pick_col(df, ["Posição", "Posicao"])
    col_classe   = pick_col(df, ["Classe"])
    col_qtd      = pick_col(df, ["Quantidade (Líquida)", "Quantidade", "Qtd"])
    col_pm_adj   = pick_col(df, ["Preço Médio Ajustado (R$)", "Preço Médio Ajustado", "PM Ajustado"])
    col_cot_hoje = pick_col(df, ["Cotação de Hoje (R$)", "Cotação de Hoje", "Cotacao de Hoje"])

    rename = {}
    if col_ticker:   rename[col_ticker] = "Ticker"
    if col_pct_cart: rename[col_pct_cart] = "Pct_Carteira"
    if col_posicao:  rename[col_posicao] = "Posicao"
    if col_classe:   rename[col_classe] = "Classe"
    if col_qtd:      rename[col_qtd] = "Qtd_Liquida"
    if col_pm_adj:   rename[col_pm_adj] = "PM_Ajustado_R$"
    if col_cot_hoje: rename[col_cot_hoje] = "Cotacao_R$"

    df = df.rename(columns=rename)

    # limpeza numérica
    if "Pct_Carteira" in df.columns:
        df["Pct_Carteira"] = df["Pct_Carteira"].astype(str).str.replace("%","").map(to_float_br)/100.0
    if "Qtd_Liquida" in df.columns:
        df["Qtd_Liquida"] = df["Qtd_Liquida"].map(to_int_safe)
    if "PM_Ajustado_R$" in df.columns:
        df["PM_Ajustado_R$"] = df["PM_Ajustado_R$"].map(to_float_br)
    if "Cotacao_R$" in df.columns:
        df["Cotacao_R$"] = df["Cotacao_R$"].map(to_float_br)

    return df

# =========================
# DADOS
# =========================
prov = load_proventos(SHEET_ID)
ativos = load_meus_ativos(SHEET_ID)

anos_disponiveis = sorted(list({int(a) for a in prov["Ano"].dropna().astype(int)}))
ano_sel = st.sidebar.selectbox("Ano", anos_disponiveis[::-1], index=0)

# Filtros adicionais (classe/posição/ticker)
classes = ["(todas)"] + sorted([c for c in prov["Classe"].dropna().unique() if str(c).strip() != ""])
classe_sel = st.sidebar.selectbox("Classe", classes, index=0)

posicoes = ["(todas)", "Ativa", "Encerrada"]
pos_sel = st.sidebar.selectbox("Posição", posicoes, index=0)

tickers_all = ["(todos)"] + sorted(prov["Ticker"].dropna().unique())
ticker_sel = st.sidebar.selectbox("Ticker", tickers_all, index=0)

# aplica filtros
df = prov.copy()
df = df[df["Ano"].astype(int) == int(ano_sel)]
if classe_sel != "(todas)":
    df = df[df["Classe"] == classe_sel]
if pos_sel != "(todas)":
    # junta com Meus Ativos para saber posição
    df = df.merge(ativos[["Ticker","Posicao"]], on="Ticker", how="left")
    df = df[df["Posicao"].fillna("(desconhecida)") == pos_sel]
if ticker_sel != "(todos)":
    df = df[df["Ticker"] == ticker_sel]

# =========================
# CALENDÁRIO (pivot)
# =========================
if df.empty:
    st.warning("Sem registros para os filtros escolhidos.")
    st.stop()

df["Mes"] = pd.Categorical(df["Mes"], categories=COL_ORDEM_MESES, ordered=True)
g_mes_ticker = df.groupby(["Ticker","Mes"], dropna=False)["Total_Liquido_R$"].sum().reset_index()

pivot = g_mes_ticker.pivot(index="Ticker", columns="Mes", values="Total_Liquido_R$").fillna(0.0)
pivot = pivot.reindex(columns=COL_ORDEM_MESES, fill_value=0.0)

# totais do ano e média mensal (12 meses)
pivot["Total no ano"] = pivot.sum(axis=1)
total_ano = pivot["Total no ano"].sum()
media_mensal = total_ano / 12.0
melhor_mes_idx = pivot.drop(columns=["Total no ano"]).sum().idxmax()
melhor_mes_val = pivot.drop(columns=["Total no ano"]).sum().max()

# junta metadados da esquerda (classe, % carteira, posição)
meta = ativos[["Ticker","Classe","Pct_Carteira","Posicao"]].copy()
meta["Pct_Carteira"] = meta["Pct_Carteira"].fillna(0.0)
tabela = meta.merge(pivot.reset_index(), on="Ticker", how="right").fillna(0.0)

# =========================
# YIELDS (DY 12m, YOC, Yield do mês)
# =========================
# janela de 12m a partir do fim do ano selecionado
jan1 = datetime(int(ano_sel), 1, 1).date()
dez31 = datetime(int(ano_sel), 12, 31).date()
ultimos_12m_ini = (datetime(int(ano_sel), 12, 31) - timedelta(days=365)).date()

prov_12m = prov[(prov["Data"].notna()) &
                (prov["Data"] > ultimos_12m_ini) &
                (prov["Data"] <= dez31)]

agg_12m = prov_12m.groupby("Ticker")["Total_Liquido_R$"].sum().reset_index().rename(columns={"Total_Liquido_R$":"Prov_12m_R$"})

# último provento por cota no ano (Yield do mês ~ último evento)
ult_unit = (df.sort_values("Data")
              .dropna(subset=["Unitario_R$"])
              .groupby("Ticker")["Unitario_R$"]
              .last()
              .reset_index()
              .rename(columns={"Unitario_R$":"Ultimo_Unitario_R$"}))

yields = ativos[["Ticker","Qtd_Liquida","PM_Ajustado_R$","Cotacao_R$"]].copy()
yields = yields.merge(agg_12m, on="Ticker", how="left").merge(ult_unit, on="Ticker", how="left")
yields["Valor_Atual_R$"] = yields["Cotacao_R$"] * yields["Qtd_Liquida"]
yields["DY_12m"]  = np.where(yields["Valor_Atual_R$"]>0, yields["Prov_12m_R$"] / yields["Valor_Atual_R$"], np.nan)
yields["YOC"]     = np.where(yields["PM_Ajustado_R$"]>0, yields["Ultimo_Unitario_R$"] / yields["PM_Ajustado_R$"], np.nan)
yields["Yield_mensal"] = yields["YOC"]  # interpretamos como yield-on-cost do último pagamento

tabela = tabela.merge(yields[["Ticker","DY_12m","YOC","Yield_mensal"]], on="Ticker", how="left")

# =========================
# KPIs
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Total no ano selecionado", f"R$ {total_ano:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
c2.metric("Média mensal (12 meses)", f"R$ {media_mensal:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
c3.metric(f"Melhor mês ({melhor_mes_idx})", f"R$ {melhor_mes_val:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

st.markdown("### Calendário por Ticker (Jan–Dez)")
fmt_cols_moeda = COL_ORDEM_MESES + ["Total no ano"]

def formatar_moeda(v):
    try:
        return "R$ " + f"{float(v):,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")
    except Exception:
        return v

tabela_view = tabela.copy()
for col in fmt_cols_moeda:
    if col in tabela_view.columns:
        tabela_view[col] = tabela_view[col].map(formatar_moeda)
for col in ["Pct_Carteira","DY_12m","YOC","Yield_mensal"]:
    if col in tabela_view.columns:
        tabela_view[col] = tabela_view[col].apply(lambda x: "-" if pd.isna(x) else f"{100*float(x):.2f}%")

st.dataframe(tabela_view, use_container_width=True, hide_index=True)

# =========================
# Ranking por Ticker (ano)
# =========================
st.markdown("### Ranking de Proventos no Ano")
ranking = pivot["Total no ano"].sort_values(ascending=False).reset_index()
ranking.columns = ["Ticker","Total no ano (R$)"]
ranking["Total no ano (R$)"] = ranking["Total no ano (R$)"].map(formatar_moeda)
st.dataframe(ranking, use_container_width=True, hide_index=True)

# =========================
# Exportação
# =========================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Calendario")
    return out.getvalue()

col_b1, col_b2 = st.columns(2)
with col_b1:
    st.download_button(
        "⬇️ Baixar Calendário (Excel)",
        data=to_excel_bytes(tabela),
        file_name=f"calendario_proventos_{ano_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_b2:
    st.download_button(
        "⬇️ Baixar Ranking (CSV)",
        data=ranking.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_proventos_{ano_sel}.csv",
        mime="text/csv"
    )

st.caption("Fonte: abas **3. Proventos** e **1. Meus Ativos** da sua planilha. O app respeita exatamente os valores lançados por você (não recalcula impostos).")
