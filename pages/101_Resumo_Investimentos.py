# pages/1_Resumo_Investimentos.py
# Resumo de Investimentos — lendo direto de Excel (.xlsx)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="💼 Resumo de Investimentos", page_icon="💼", layout="wide")
TEMPLATE = "plotly_dark"
st.title("💼 Resumo de Investimentos")

# =========================
# Helpers
# =========================
def moeda(v) -> str:
    try: v = float(v)
    except Exception: v = 0.0
    return f"R$ {v:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")

def to_float_br(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except: return np.nan

def guess_sheet(xl: pd.ExcelFile, keys):
    """Acha aba por palavras‑chave (casefold/PT)."""
    names = xl.sheet_names
    low = [n.lower() for n in names]
    for i, n in enumerate(low):
        if any(k in n for k in keys):
            return names[i]
    return None

def normalize_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# =========================
# Entrada do arquivo
# =========================
st.sidebar.header("Arquivo de dados")
up = st.sidebar.file_uploader("Carregar planilha Excel (.xlsx)", type=["xlsx"])
# Também aceito um caminho local (útil no dev/local)
path_local = st.sidebar.text_input("ou caminho local (opcional)", value="")

if up is not None:
    excel = up
elif path_local:
    try:
        excel = path_local
    except Exception:
        st.error("Caminho inválido.")
        st.stop()
else:
    st.info("Envie o Excel nas opções da esquerda para carregar os dados.")
    st.stop()

# =========================
# Leitura das abas
# =========================
try:
    xl = pd.ExcelFile(excel, engine="openpyxl")
except Exception as e:
    st.error(f"Não consegui abrir o Excel. Detalhe: {e}")
    st.stop()

# Tentativa de descoberta de nomes de abas
sheet_mov = guess_sheet(xl, ["mov", "lanç", "lanc", "transa", "opera"])
sheet_pos = guess_sheet(xl, ["posi", "consol", "carteira", "posição", "consolid"])
sheet_pro = guess_sheet(xl, ["prov", "divid", "rend", "provento"])

colA, colB, colC = st.columns(3)
with colA: st.caption("Aba Movimentações"); st.code(sheet_mov or "—", language="bash")
with colB: st.caption("Aba Posição/Consolidado"); st.code(sheet_pos or "—", language="bash")
with colC: st.caption("Aba Proventos"); st.code(sheet_pro or "—", language="bash")

if not sheet_mov:
    st.error("Não encontrei a aba de Movimentações. Renomeie a aba ou escolha um arquivo com essa aba.")
    st.stop()

# Carrega cada aba encontrada
df_mov = normalize_cols(pd.read_excel(xl, sheet_name=sheet_mov, engine="openpyxl"))
df_pos = normalize_cols(pd.read_excel(xl, sheet_name=sheet_pos, engine="openpyxl")) if sheet_pos else pd.DataFrame()
df_prov = normalize_cols(pd.read_excel(xl, sheet_name=sheet_pro, engine="openpyxl")) if sheet_pro else pd.DataFrame()

# =========================
# Normalizações — MOV
# =========================
# Data
for c in ["data", "data operação", "data operacao", "data_op", "date"]:
    if c in df_mov.columns:
        df_mov["data"] = pd.to_datetime(df_mov[c], errors="coerce")
        break
if "data" not in df_mov.columns:
    df_mov["data"] = pd.NaT

# Total (valor da linha). Se não tiver "total", tenta preço * quantidade.
if "total" in df_mov.columns:
    df_mov["total"] = df_mov["total"].apply(to_float_br)
else:
    preco = None
    for c in ["preco", "preço", "preco unit.", "preço unit.", "preco_unit"]:
        if c in df_mov.columns:
            preco = df_mov[c].apply(to_float_br); break
    qtd = None
    for c in ["quantidade", "qtd", "qte", "qde"]:
        if c in df_mov.columns:
            qtd = pd.to_numeric(df_mov[c], errors="coerce"); break
    df_mov["total"] = preco * qtd if (preco is not None and qtd is not None) else 0.0

# Tipo (texto da operação)
if "tipo" not in df_mov.columns:
    for c in ["movimento", "operation", "operacao", "operação", "tipo operação"]:
        if c in df_mov.columns:
            df_mov["tipo"] = df_mov[c]; break
if "tipo" not in df_mov.columns:
    df_mov["tipo"] = ""
df_mov["tipo_norm"] = df_mov["tipo"].astype(str).str.lower().str.strip()

# =========================
# Normalizações — POSIÇÃO
# =========================
if not df_pos.empty:
    if "valor_atual" not in df_pos.columns:
        qcol = next((c for c in ["quantidade", "qtd"] if c in df_pos.columns), None)
        pcol = next((c for c in ["preco_atual", "preço_atual", "ultimo_preco", "preco", "preço"] if c in df_pos.columns), None)
        if qcol and pcol:
            df_pos["valor_atual"] = pd.to_numeric(df_pos[qcol], errors="coerce") * df_pos[pcol].apply(to_float_br)
        else:
            df_pos["valor_atual"] = 0.0

# =========================
# Normalizações — PROVENTOS
# =========================
if not df_prov.empty:
    if "data_com" not in df_prov.columns:
        for c in ["data com", "com_data"]:
            if c in df_prov.columns:
                df_prov["data_com"] = pd.to_datetime(df_prov[c], errors="coerce"); break
    else:
        df_prov["data_com"] = pd.to_datetime(df_prov["data_com"], errors="coerce")

    if "pagamento" in df_prov.columns:
        df_prov["pagamento"] = pd.to_datetime(df_prov["pagamento"], errors="coerce")

    if "valor_cota" in df_prov.columns:
        df_prov["valor_cota"] = df_prov["valor_cota"].apply(to_float_br)
    if "qtde" in df_prov.columns:
        df_prov["qtde"] = pd.to_numeric(df_prov["qtde"], errors="coerce")
    if "total" not in df_prov.columns and {"valor_cota", "qtde"} <= set(df_prov.columns):
        df_prov["total"] = df_prov["valor_cota"] * df_prov["qtde"]

# =========================
# Lógica financeira
# =========================
DEPOSITOS = {"aporte", "depósito", "deposito", "transferência recebida", "transferencia recebida"}
RETIRADAS = {"retirada", "saque", "transferência enviada", "transferencia enviada"}
COMPRAS   = {"compra", "compra - fracionário", "compra - mercado"}
VENDAS    = {"venda", "venda - fracionário", "venda - mercado"}

st.sidebar.header("Filtros")
metodo_aporte = st.sidebar.radio(
    "Como calcular o Aportado (bolso)?",
    options=["Depósito - Retirada", "Compra - Venda"],
    index=0,
)

df_mov["mes"] = pd.to_datetime(df_mov["data"], errors="coerce").dt.to_period("M").dt.to_timestamp()

if metodo_aporte == "Depósito - Retirada":
    df_mov["aporte_sinal"] = np.select(
        [df_mov["tipo_norm"].isin(DEPOSITOS), df_mov["tipo_norm"].isin(RETIRADAS)],
        [df_mov["total"].fillna(0.0), -df_mov["total"].fillna(0.0)],
        default=0.0
    )
else:
    df_mov["aporte_sinal"] = np.select(
        [df_mov["tipo_norm"].isin(COMPRAS), df_mov["tipo_norm"].isin(VENDAS)],
        [df_mov["total"].abs().fillna(0.0), -df_mov["total"].abs().fillna(0.0)],
        default=0.0
    )

aportes_m = df_mov.groupby("mes", as_index=False)["aporte_sinal"].sum()
aportes_m["aportado_acum"] = aportes_m["aporte_sinal"].clip(lower=0).cumsum()

aportado_total = float(aportes_m["aporte_sinal"].sum())
patrimonio_atual = float(pd.to_numeric(df_pos.get("valor_atual", 0), errors="coerce").sum()) if not df_pos.empty else None
variacao = (patrimonio_atual - aportado_total) if patrimonio_atual is not None else None
perc_var = (variacao / aportado_total * 100) if (variacao is not None and aportado_total) else None

if not df_prov.empty and {"pagamento", "total"} <= set(df_prov.columns):
    prov12 = df_prov.set_index("pagamento").sort_index()
    prov12 = prov12.last("365D") if len(prov12) else prov12
    media12 = float(prov12["total"].sum()) / 12 if len(prov12) else 0.0
    total_prov = float(df_prov["total"].sum())
else:
    media12, total_prov = None, None

# Filtro de ano
anos = sorted(aportes_m["mes"].dt.year.unique())
ano_sel = st.sidebar.multiselect("Ano", options=anos, default=anos[-1:] if anos else None)
aportes_plot = aportes_m[aportes_m["mes"].dt.year.isin(ano_sel)] if ano_sel else aportes_m.copy()
prov_plot = df_prov[df_prov.get("pagamento", pd.NaT).dt.year.isin(ano_sel)] if (not df_prov.empty and "pagamento" in df_prov) else pd.DataFrame()

# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.caption("Variação Patrimonial")
    if variacao is None:
        st.metric(" ", "—", "adicione aba de Posição")
    else:
        st.metric(" ", moeda(variacao), f"{(perc_var or 0):.2f}%".replace(".", ","))

with c2:
    st.caption("Aportado (bolso)")
    st.metric(" ", moeda(aportado_total))

with c3:
    st.caption("Patrimônio atual")
    st.metric(" ", "—" if patrimonio_atual is None else moeda(patrimonio_atual))

with c4:
    st.caption("Proventos (média 12m)")
    if media12 is None:
        st.metric(" ", "—", "adicione aba de Proventos")
    else:
        st.metric(" ", moeda(media12), f"Total {moeda(total_prov)}")

st.markdown("---")

# =========================
# Evolução x Calendário
# =========================
a, b = st.columns([2, 1])

with a:
    st.subheader("Evolução Patrimonial")
    fig = px.bar(aportes_plot, x="mes", y="aportado_acum",
                 labels={"mes": "", "aportado_acum": "Aportado (Acumulado)"},
                 template=TEMPLATE)
    fig.add_scatter(x=aportes_plot["mes"], y=aportes_plot["aportado_acum"],
                    mode="lines+markers", name="Evolução")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with b:
    st.subheader("Calendário de eventos")
    if not prov_plot.empty and {"pagamento","ticker"}.issubset(prov_plot.columns):
        hoje = pd.to_datetime(date.today())
        prox = prov_plot[pd.to_datetime(prov_plot["pagamento"], errors="coerce") >= hoje].copy()
        prox = prox.sort_values("pagamento").head(8)
        if len(prox) == 0:
            st.info("Sem pagamentos futuros nos filtros.")
        for _, r in prox.iterrows():
            total = r["total"] if "total" in r else (float(r.get("valor_cota",0))*float(r.get("qtde",0)))
            st.write(f"**{r.get('ticker','-')}** — {moeda(total)}")
            dcom = r.get("data_com", pd.NaT)
            dcom_txt = pd.to_datetime(dcom).date() if pd.notna(dcom) else "-"
            dpg  = pd.to_datetime(r.get("pagamento", pd.NaT))
            dpg_txt = dpg.date() if pd.notna(dpg) else "-"
            st.caption(f"data com: {dcom_txt} | pagamento: {dpg_txt}")
            st.divider()
    else:
        st.info("Proventos indisponíveis (adicione/ajuste a aba).")

# =========================
# Conferência rápida (opcional)
# =========================
with st.expander("🛠️ Conferência de dados"):
    st.write("Aportes por mês (fonte do bolso):")
    st.dataframe(aportes_m)
    if not df_pos.empty:
        st.write("Posição (resumo):")
        st.dataframe(df_pos.head(20))
    if not df_prov.empty:
        st.write("Proventos (amostra):")
        st.dataframe(df_prov.head(20))
