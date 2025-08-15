# pages/1_Resumo_Investimentos.py — v2
# Dashboard estilo "md" com cálculos fiéis (Patrimônio, Aportado, Variação)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="💼 Resumo de Investimentos", page_icon="💼", layout="wide")
TEMPLATE = "plotly_dark"
st.title("💼 Resumo de Investimentos")

# =========================
# CONFIG PLANILHA
# =========================
SHEET_ID = ""        # ex: "1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GID_MOV  = ""        # Movimentações / Lançamentos
GID_POS  = ""        # Posição / Consolidado
GID_PROV = ""        # Proventos

def csv_url(gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

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

# =========================
# MOCK (fallback)
# =========================
@st.cache_data(ttl=300)
def mock_data():
    hoje = date.today()
    datas = pd.date_range(hoje - pd.DateOffset(months=11), periods=12, freq="MS")
    aportes = np.linspace(1500, 2600, len(datas))
    df_mov = pd.DataFrame({"data": datas, "valor": aportes, "tipo": "Depósito"})
    df_mov["total"] = df_mov["valor"]
    df_pos = pd.DataFrame({
        "ticker": ["HGLG11","MXRF11","BBAS3"],
        "classe": ["FII","FII","Ação"],
        "quantidade": [20,300,100],
        "preco_atual": [180.0, 10.2, 35.0]
    })
    df_pos["valor_atual"] = df_pos["quantidade"] * df_pos["preco_atual"]
    prov_datas = [hoje + timedelta(days=d) for d in (3,8,15,25)]
    df_prov = pd.DataFrame({
        "ticker": ["BBAS3","CXSE3","HGLG11","MXRF11"],
        "valor_cota": [0.35, 0.31, 1.10, 0.12],
        "qtde": [100, 50, 20, 300],
        "data_com": pd.to_datetime([d - timedelta(days=10) for d in prov_datas]),
        "pagamento": pd.to_datetime(prov_datas)
    })
    df_prov["total"] = df_prov["valor_cota"] * df_prov["qtde"]
    return df_mov, df_pos, df_prov

# =========================
# CARREGAMENTO
# =========================
def carregar():
    if SHEET_ID and GID_MOV and GID_POS and GID_PROV:
        try:
            mov  = pd.read_csv(csv_url(GID_MOV))
            pos  = pd.read_csv(csv_url(GID_POS))
            prov = pd.read_csv(csv_url(GID_PROV))
            mov.columns  = [c.strip().lower() for c in mov.columns]
            pos.columns  = [c.strip().lower() for c in pos.columns]
            prov.columns = [c.strip().lower() for c in prov.columns]

            # Datas
            for c in ["data","data operação","data_op","date"]:
                if c in mov.columns: 
                    mov["data"] = pd.to_datetime(mov[c], errors="coerce"); break
            if "data" not in mov: mov["data"] = pd.NaT

            # Normaliza valores (total/preço * quantidade)
            for c in ["total","valor total","valor_total","valor"]:
                if c in mov.columns:
                    mov["total"] = mov[c].apply(to_float_br)
                    break
            if "total" not in mov.columns:
                preco = None
                for c in ["preco","preço","preco unit.","preco_unit"]:
                    if c in mov.columns: preco = mov[c].apply(to_float_br); break
                qtd = None
                for c in ["quantidade","qtd","qte","qde"]:
                    if c in mov.columns: qtd = pd.to_numeric(mov[c], errors="coerce"); break
                if preco is not None and qtd is not None:
                    mov["total"] = preco * qtd
                else:
                    mov["total"] = 0.0

            # Tipo
            if "tipo" not in mov.columns:
                for c in ["movimento","operation","operacao","operação"]:
                    if c in mov.columns: mov["tipo"] = mov[c]; break
            if "tipo" not in mov.columns: mov["tipo"] = ""

            # Posição: valor_atual
            if "valor_atual" not in pos.columns:
                qcol = next((c for c in ["quantidade","qtd"] if c in pos.columns), None)
                pcol = next((c for c in ["preco_atual","preço_atual","ultimo_preco","preco"] if c in pos.columns), None)
                if qcol and pcol:
                    pos["valor_atual"] = pd.to_numeric(pos[qcol], errors="coerce") * pos[pcol].apply(to_float_br)
                else:
                    pos["valor_atual"] = 0.0

            # Proventos
            if "data_com" not in prov.columns:
                for c in ["data com","data_com","com_data"]:
                    if c in prov.columns: prov["data_com"] = pd.to_datetime(prov[c], errors="coerce"); break
            else:
                prov["data_com"] = pd.to_datetime(prov["data_com"], errors="coerce")
            if "pagamento" in prov.columns:
                prov["pagamento"] = pd.to_datetime(prov["pagamento"], errors="coerce")
            if "valor_cota" in prov.columns:
                prov["valor_cota"] = prov["valor_cota"].apply(to_float_br)
            if "qtde" in prov.columns:
                prov["qtde"] = pd.to_numeric(prov["qtde"], errors="coerce")
            if "total" not in prov.columns and {"valor_cota","qtde"} <= set(prov.columns):
                prov["total"] = prov["valor_cota"] * prov["qtde"]

            return mov, pos, prov, False
        except Exception as e:
            st.warning(f"Falha ao ler Sheets ({e}). Usando MOCK.")
            m, p, pr = mock_data(); return m, p, pr, True
    else:
        m, p, pr = mock_data(); return m, p, pr, True

df_mov, df_pos, df_prov, is_mock = carregar()
if is_mock:
    st.info("🧪 Dados: MOCK (preencha SHEET_ID + GIDs para usar sua planilha).")

# =========================
# LÓGICA FINANCEIRA
# =========================
DEPOSITOS = {"aporte","depósito","deposito","transferência recebida","transferencia recebida"}
RETIRADAS = {"retirada","saque","transferência enviada","transferencia enviada"}
COMPRAS   = {"compra","compra - fracionário","compra - mercado"}
VENDAS    = {"venda","venda - fracionário","venda - mercado"}

st.sidebar.header("Filtros")
metodo_aporte = st.sidebar.radio(
    "Como calcular o Aportado (bolso)?",
    options=["Depósito - Retirada", "Compra - Venda"],
    index=0,
    help="Escolha a fonte de caixa: (Depósitos/Retiradas) OU (Compras/Vendas)."
)

# Aportado por mês
mov = df_mov.copy()
mov["tipo_norm"] = mov["tipo"].astype(str).str.lower().str.strip()
mov["mes"] = pd.to_datetime(mov["data"], errors="coerce").dt.to_period("M").dt.to_timestamp()

if metodo_aporte == "Depósito - Retirada":
    mov["aporte_sinal"] = np.select(
        [
            mov["tipo_norm"].isin(DEPOSITOS),
            mov["tipo_norm"].isin(RETIRADAS)
        ],
        [mov["total"].fillna(0.0), -mov["total"].fillna(0.0)],
        default=0.0
    )
else:
    # Compra positiva (sai do bolso), Venda negativa (volta p/ bolso)
    mov["aporte_sinal"] = np.select(
        [
            mov["tipo_norm"].isin(COMPRAS),
            mov["tipo_norm"].isin(VENDAS)
        ],
        [mov["total"].abs().fillna(0.0), -mov["total"].abs().fillna(0.0)],
        default=0.0
    )

aportes_m = mov.groupby("mes", as_index=False)["aporte_sinal"].sum()
aportes_m["aportado_acum"] = aportes_m["aporte_sinal"].clip(lower=0).cumsum()

# Patrimônio atual (posição)
patrimonio_atual = float(pd.to_numeric(df_pos.get("valor_atual", 0), errors="coerce").sum())

# Aportado total (bolso)
aportado_total = float(aportes_m["aporte_sinal"].sum())

# KPIs
variacao = patrimonio_atual - aportado_total
perc_var = (variacao / aportado_total * 100) if aportado_total else 0.0

# Proventos média 12m
if {"pagamento","total"} <= set(df_prov.columns):
    prov12 = df_prov.set_index("pagamento").sort_index()
    prov12 = prov12.last("365D") if len(prov12) else prov12
    media12 = float(prov12["total"].sum()) / 12 if len(prov12) else 0.0
    total_prov = float(df_prov["total"].sum())
else:
    media12, total_prov = 0.0, 0.0

# FILTRO DE ANO
anos = sorted(aportes_m["mes"].dt.year.unique())
ano_sel = st.sidebar.multiselect("Ano", options=anos, default=anos[-1:] if anos else None)
if ano_sel:
    aportes_plot = aportes_m[aportes_m["mes"].dt.year.isin(ano_sel)]
    prov_plot = df_prov[df_prov.get("pagamento", pd.NaT).dt.year.isin(ano_sel)] if "pagamento" in df_prov else df_prov.iloc[0:0]
else:
    aportes_plot = aportes_m.copy()
    prov_plot = df_prov.copy()

# =========================
# KPIs (linha superior)
# =========================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.caption("Variação Patrimonial")
    st.metric(" ", moeda(variacao), f"{perc_var:.2f}%".replace(".",","))

with c2:
    st.caption("Aportado (bolso)")
    st.metric(" ", moeda(aportado_total))

with c3:
    st.caption("Patrimônio atual")
    st.metric(" ", moeda(patrimonio_atual))

with c4:
    st.caption("Proventos (média 12m)")
    st.metric(" ", moeda(media12), f"Total {moeda(total_prov)}")

st.markdown("---")

# =========================
# Evolução x Calendário
# =========================
a, b = st.columns([2, 1])

with a:
    st.subheader("Evolução Patrimonial")
    # Barras = aportado acumulado; Linha = patrimônio aproximado (aqui = aportado_acum)
    fig = px.bar(aportes_plot, x="mes", y="aportado_acum",
                 labels={"mes": "", "aportado_acum": "Aportado (Acumulado)"},
                 template=TEMPLATE)
    fig.add_scatter(x=aportes_plot["mes"], y=aportes_plot["aportado_acum"],
                    mode="lines+markers", name="Evolução")
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with b:
    st.subheader("Calendário de eventos")
    if {"pagamento","ticker"} <= set(prov_plot.columns):
        hoje = pd.to_datetime(date.today())
        prox = prov_plot[pd.to_datetime(prov_plot["pagamento"], errors="coerce") >= hoje].copy()
        prox = prox.sort_values("pagamento").head(8)
        if len(prox)==0: st.info("Sem pagamentos futuros nos filtros.")
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
        st.info("Mapeie colunas de proventos para ver o calendário.")

# =========================
# Debug opcional
# =========================
with st.expander("🛠️ Conferência (opcional)"):
    st.write("Aportes por mês (fonte do bolso):")
    st.dataframe(aportes_m)

