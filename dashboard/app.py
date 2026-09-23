"""
ColdRoute Intelligence Platform — Dashboard Ejecutivo
Hosting: Streamlit Community Cloud (gratuito)
URL: coldroute.streamlit.app
Metodología JP Morgan Fusion aplicada a puertos del Biobío
"""
import sys
sys.path.insert(0, '/home/claude/coldroute')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json, os

# ── Config página ────────────────────────────────────────────
st.set_page_config(
    page_title="ColdRoute Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0B1829; }
    .block-container { padding-top: 1rem; }
    h1 { color: #00D4FF; font-size: 1.8rem; }
    h2 { color: #7EC8E3; }
    h3 { color: #B0D4E8; }
    .metric-card {
        background: #0D2137;
        border: 1px solid #1A4A6C;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px;
    }
    .verde { border-left: 4px solid #00FF88; }
    .amarillo { border-left: 4px solid #FFD700; }
    .rojo { border-left: 4px solid #FF4444; }
    .stMetric label { color: #7EC8E3 !important; }
    .stMetric > div { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
col_logo, col_title, col_time = st.columns([1,6,2])
with col_logo:
    st.markdown("## 🌊")
with col_title:
    st.markdown("# ColdRoute Intelligence Platform")
    st.markdown("**Puerto Coronel** · Metodología JP Morgan Fusion · "
                "Mecanismo CASST © Cristián G. Villar Neira C.I. 12.323.957-1")
with col_time:
    st.markdown(f"**Actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("⚡ Ciclo IoT: 15 min")

st.divider()

# ── Cargar / generar datos ───────────────────────────────────
@st.cache_data(ttl=900)  # 15 minutos
def cargar_datos():
    from data.simulador_iot import SimuladorReeferIoT, SimuladorOperacionesPortuarias
    from models.anomaly_detector import AgenteDetectorAnomalias
    from models.demand_forecaster import AgenteForecastDemanda
    from models.risk_engine import AgenteMotoRiesgo

    sim_iot = SimuladorReeferIoT()
    sim_ops = SimuladorOperacionesPortuarias()

    df_meta, df_series = sim_iot.generar_portafolio_puerto(
        n_normales=120, n_anomalos=6
    )
    df_dem = sim_ops.generar_demanda_reefer(meses=24)
    df_ms = sim_ops.generar_historico_market_share()
    df_front = sim_ops.generar_tiempos_frontera_neuquen(n=200)

    # Agente 1 — War Room
    agente_an = AgenteDetectorAnomalias()
    war_room = agente_an.resumen_portafolio(df_meta, df_series)

    # Agente 2 — Forecast
    agente_fc = AgenteForecastDemanda()
    agente_fc.entrenar(df_dem)
    forecast = agente_fc.predecir(6)

    # Agente 3 — Riesgo
    agente_rk = AgenteMotoRiesgo()
    val_res = agente_rk.simular_perdidas_temporada(n_contenedores=120)
    stress = agente_rk.stress_testing()
    val_tipo = agente_rk.val_por_tipo_carga()

    return {
        "war_room": war_room,
        "forecast": forecast,
        "df_ms": df_ms,
        "df_dem": df_dem,
        "df_front": df_front,
        "val": val_res,
        "stress": stress,
        "val_tipo": val_tipo,
        "df_meta": df_meta,
    }

with st.spinner("🔄 Procesando ciclo de inteligencia portuaria..."):
    datos = cargar_datos()

wr = datos["war_room"]
fc = datos["forecast"]
df_ms = datos["df_ms"]
df_dem = datos["df_dem"]
val = datos["val"]
stress = datos["stress"]

# ══════════════════════════════════════════════════════════════
# FILA 1 — KPIs PRINCIPALES
# ══════════════════════════════════════════════════════════════
st.markdown("### 📊 KPIs Ejecutivos — Tiempo Real")
c1, c2, c3, c4, c5 = st.columns(5)

ms_actual = df_ms[df_ms["año"]==2025]["market_share_pct"].mean()
with c1:
    delta_ms = ms_actual - 44.68
    st.metric("Market Share Reefer", f"{ms_actual:.1f}%",
              f"{delta_ms:+.1f} pp vs 2025")

with c2:
    st.metric("Contenedores VERDE", f"{wr['semaforo']['VERDE']}",
              f"de {wr['total_contenedores']} totales")

with c3:
    col = "🔴" if wr['semaforo']['ROJO'] > 0 else "🟢"
    st.metric(f"{col} Alertas ROJO", wr['semaforo']['ROJO'],
              "EMERGENCIA" if wr['semaforo']['ROJO'] > 0 else "Sin alertas críticas")

with c4:
    st.metric("Pérdida Esperada", f"US${wr['perdida_total_esperada_usd']:,.0f}",
              "portafolio actual")

with c5:
    mrr_est = 12_500_000 + 5 * 3_800_000
    st.metric("MRR Estimado", f"${mrr_est/1e6:.1f}M CLP",
              "+Rev.Share pendiente")

st.divider()

# ══════════════════════════════════════════════════════════════
# FILA 2 — WAR ROOM + MARKET SHARE
# ══════════════════════════════════════════════════════════════
col_wr, col_ms = st.columns([1, 2])

with col_wr:
    st.markdown("### 🚨 War Room Cadena de Frío")

    # Gauge semáforo
    total = wr["total_contenedores"]
    verde = wr["semaforo"]["VERDE"]
    amarillo = wr["semaforo"]["AMARILLO"]
    rojo = wr["semaforo"]["ROJO"]

    fig_gauge = go.Figure(go.Pie(
        values=[verde, amarillo, rojo],
        labels=["VERDE", "AMARILLO", "ROJO"],
        hole=0.5,
        marker_colors=["#00FF88","#FFD700","#FF4444"],
        textinfo="label+value",
        textfont_color="white",
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=280,
        margin=dict(t=20,b=0,l=0,r=0),
        annotations=[dict(
            text=f"<b>{total}</b><br>TEUs",
            x=0.5, y=0.5,
            font_size=20,
            font_color="white",
            showarrow=False
        )],
        showlegend=False,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Lista alertas críticas
    if wr["contenedores_criticos"]:
        st.markdown("**🚨 Contenedores ROJO:**")
        for c in wr["contenedores_criticos"][:4]:
            st.markdown(
                f"<div class='metric-card rojo'>"
                f"<b>{c['contenedor_id']}</b> | {c['tipo_carga']}<br>"
                f"T={c['temperatura_actual']}°C | Δ={c['delta_temp']:+.1f}°C<br>"
                f"<small>US${c['perdida_esperada_usd']:,.0f} riesgo</small>"
                f"</div>",
                unsafe_allow_html=True
            )

with col_ms:
    st.markdown("### 📉 Market Share Reefer — Puerto Coronel vs Objetivo ColdRoute")

    # Datos verificados 2023-2025 + proyección con ColdRoute
    años_verificados = [2023, 2024, 2025]
    ms_verificado = [64.83, 48.91, 44.68]
    años_proyec = [2025, 2026, 2027]
    ms_proyec = [44.68, 50.2, 56.5]  # con ColdRoute

    fig_ms = go.Figure()
    fig_ms.add_trace(go.Scatter(
        x=años_verificados, y=ms_verificado,
        mode="lines+markers",
        name="Real (verificado)",
        line=dict(color="#FF4444", width=3),
        marker=dict(size=10),
    ))
    fig_ms.add_trace(go.Scatter(
        x=años_proyec, y=ms_proyec,
        mode="lines+markers",
        name="Proyección con ColdRoute",
        line=dict(color="#00FF88", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
    ))
    fig_ms.add_hline(y=50, line_dash="dot",
                      line_color="#FFD700",
                      annotation_text="Meta piloto 50%")

    fig_ms.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,33,55,0.8)",
        font_color="white",
        height=300,
        xaxis=dict(title="Año", gridcolor="#1A3A5C"),
        yaxis=dict(title="Market Share %", gridcolor="#1A3A5C",
                   range=[30, 75]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_ms, use_container_width=True)

    # Tabla de pérdida por 8.900 contenedores desviados
    tcu_usd = 200
    teu_clp = 200 * 930
    st.markdown(
        f"**💸 Costo de la información asimétrica:** "
        f"8.900 contenedores × US${tcu_usd}/TEU = "
        f"**US$1.780.000/año** en ingresos perdidos por Valparaíso"
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# FILA 3 — FORECAST REEFER + STRESS TESTING
# ══════════════════════════════════════════════════════════════
col_fc, col_st = st.columns([3, 2])

with col_fc:
    st.markdown("### 📅 Forecast Demanda Reefer — 42 Días Forward")
    st.caption("Ensemble Prophet + XGBoost | IC 90% | Calibrado con datos Puerto Coronel")

    fig_fc = go.Figure()
    fechas = [r["fecha"] for r in fc.to_dict("records")]
    dem = [r["demanda_forecast"] for r in fc.to_dict("records")]
    low = [r["ic90_bajo"] for r in fc.to_dict("records")]
    high = [r["ic90_alto"] for r in fc.to_dict("records")]
    gap = [r["gap_escasez"] for r in fc.to_dict("records")]

    fig_fc.add_trace(go.Scatter(
        x=fechas, y=high,
        fill=None, mode="lines",
        line=dict(color="rgba(0,212,255,0)"),
        showlegend=False,
    ))
    fig_fc.add_trace(go.Scatter(
        x=fechas, y=low,
        fill="tonexty",
        mode="lines",
        line=dict(color="rgba(0,212,255,0)"),
        fillcolor="rgba(0,212,255,0.15)",
        name="IC 90%",
    ))
    fig_fc.add_trace(go.Scatter(
        x=fechas, y=dem,
        mode="lines+markers",
        name="Demanda forecast",
        line=dict(color="#00D4FF", width=3),
        marker=dict(size=8),
    ))
    fig_fc.add_trace(go.Bar(
        x=fechas, y=gap,
        name="Gap escasez",
        marker_color="#FF4444",
        opacity=0.6,
        yaxis="y2",
    ))

    fig_fc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,33,55,0.8)",
        font_color="white",
        height=320,
        yaxis=dict(title="Reefers demandados", gridcolor="#1A3A5C"),
        yaxis2=dict(title="Gap escasez", overlaying="y",
                    side="right", gridcolor="#1A3A5C"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=40, l=60, r=60),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Tabla resumen
    st.dataframe(
        fc[["fecha","demanda_forecast","disponibles_estimados",
            "gap_escasez","nor_recomendado","alerta_escasez"]].rename(columns={
            "fecha": "Fecha", "demanda_forecast": "Demanda",
            "disponibles_estimados": "Disponibles",
            "gap_escasez": "Gap",
            "nor_recomendado": "NOR Rec.",
            "alerta_escasez": "⚠️ Escasez",
        }),
        use_container_width=True,
        hide_index=True,
    )

with col_st:
    st.markdown("### ⚡ Stress Testing — Escenarios Históricos")
    st.caption("Metodología JP Morgan: pérdida vs protección ColdRoute")

    nombres = list(stress.keys())
    sin_sistema = [v["perdida_sin_sistema_clp"]/1e9 for v in stress.values()]
    con_sistema = [v["perdida_con_coldroute_clp"]/1e9 for v in stress.values()]

    fig_st = go.Figure()
    fig_st.add_trace(go.Bar(
        name="Sin ColdRoute",
        x=nombres,
        y=sin_sistema,
        marker_color="#FF4444",
    ))
    fig_st.add_trace(go.Bar(
        name="Con ColdRoute",
        x=nombres,
        y=con_sistema,
        marker_color="#00FF88",
    ))
    fig_st.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,33,55,0.8)",
        font_color="white",
        height=280,
        yaxis=dict(title="Pérdida (Miles MM CLP)", gridcolor="#1A3A5C"),
        xaxis=dict(tickangle=-30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=80, l=60, r=20),
    )
    st.plotly_chart(fig_st, use_container_width=True)

    # VaL resumen
    st.markdown(
        f"<div class='metric-card verde'>"
        f"<b>VaL(95%) Temporada:</b> US${val['VaL_95_usd']:,.0f}<br>"
        f"<b>CVaL(95%):</b> US${val['CVaL_95_usd']:,.0f}<br>"
        f"<b>Pérdida media:</b> US${val['perdida_media_usd']:,.0f}<br>"
        f"<small>Monte Carlo Sobol QMC N=50.000</small>"
        f"</div>",
        unsafe_allow_html=True
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# FILA 4 — PROYECCIÓN DE INGRESOS + SIDEBAR
# ══════════════════════════════════════════════════════════════
st.markdown("### 💰 Proyección de Ingresos ColdRoute — 36 Meses")

meses = list(range(1, 37))
saas_proj = [0]*5 + [12.5, 12.5, 20, 25, 35, 35, 36, 37, 38, 40,
             45, 50, 55, 60, 65, 70, 78, 85, 90, 100, 108, 112, 115, 120,
             122, 125, 128, 130, 132][:31]
rev_share_proj = [0]*6 + [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22,
                  25, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42][:30]
total_proj = [s + r for s, r in zip(saas_proj, rev_share_proj)]

fig_rev = go.Figure()
fig_rev.add_trace(go.Scatter(
    x=meses, y=saas_proj,
    mode="lines", name="SaaS MRR",
    line=dict(color="#00D4FF", width=2),
    fill="tozeroy",
    fillcolor="rgba(0,212,255,0.1)",
))
fig_rev.add_trace(go.Scatter(
    x=meses, y=total_proj,
    mode="lines", name="MRR Total",
    line=dict(color="#00FF88", width=3),
))
fig_rev.add_hline(y=110, line_dash="dash",
                   line_color="#FFD700",
                   annotation_text="Objetivo $110M CLP/mes")
fig_rev.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,33,55,0.8)",
    font_color="white",
    height=280,
    xaxis=dict(title="Mes desde inicio", gridcolor="#1A3A5C"),
    yaxis=dict(title="MRR (millones CLP)", gridcolor="#1A3A5C"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=10, b=40, l=60, r=20),
)
st.plotly_chart(fig_rev, use_container_width=True)

# Footer
st.divider()
st.markdown(
    "<small>ColdRoute Intelligence Platform v1.0 · "
    "Mecanismo CASST © Cristián G. Villar Neira C.I. 12.323.957-1 · "
    "Ley N°17.336 · INAPI Modelo de Utilidad 2026 · "
    "Sub-Fondo Startup Capital Biobío AGF · "
    "Hosting: Streamlit Community Cloud (gratuito)</small>",
    unsafe_allow_html=True
)
