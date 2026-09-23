"""
ColdRoute Intelligence Platform — Configuración Maestra
Metodología JP Morgan Fusion aplicada a puertos del Biobío
Autor del mecanismo CASST: Cristián G. Villar Neira C.I. 12.323.957-1
"""
from pydantic_settings import BaseSettings
from typing import Dict, List
import os

# ══════════════════════════════════════════════════════════════
# PUERTOS OBJETIVO (datos verificados)
# ══════════════════════════════════════════════════════════════
PUERTOS = {
    "CORONEL": {
        "nombre": "Puerto Coronel",
        "cppi_2025": 26,
        "cppi_2024": 40,
        "cppi_2023": 97,
        "market_share_reefer_2023": 64.83,
        "market_share_reefer_2024": 48.91,
        "market_share_reefer_2025": 44.68,
        "contenedores_desviados_valparaiso": 8900,
        "teu_salmon_2025": 21205,
        "toneladas_neuquen_2024": 64000,
        "conexiones_reefer": 1500,
        "muelles": 3,
        "gruas": 9,
        "bodega_m2": 130000,
        "gerente_general": "Patricio Román Lois",
        "gerente_comercial": "Javier Lobo Salazar",
        "gerente_ops": "Lukas Buckel Ocqueteau",
    },
    "TALCAHUANO": {
        "nombre": "Puerto Talcahuano San Vicente",
        "toneladas_2024": 25805825,
        "toneladas_svti_ttpsa_2025": 6157000,
        "gerente_general": "Cristian Wulf",
        "problema_consolidacion": "carga forestal suelta sin sistema predictivo",
    }
}

# ══════════════════════════════════════════════════════════════
# UMBRALES DE TEMPERATURA POR TIPO DE CARGA
# Fuente: USDA APHIS / FDA 21 CFR / SENASA Argentina
# ══════════════════════════════════════════════════════════════
TEMP_THRESHOLDS = {
    "fruta_fresca": {
        "setpoint_min": -0.5,
        "setpoint_max": 1.0,
        "alerta_amarilla_delta": 1.5,  # °C sobre setpoint por 12 min
        "alerta_roja_delta": 3.0,      # °C sobre setpoint por 6 min
        "tiempo_amarillo_min": 12,
        "tiempo_rojo_min": 6,
        "productos": ["manzanas","peras","cerezas","arandanos","kiwis","uvas"],
        "costo_reclamo_usd": {"min": 8000, "max": 45000},
    },
    "salmon_fresco": {
        "setpoint_min": 0.0,
        "setpoint_max": 2.0,
        "alerta_amarilla_delta": 3.0,
        "alerta_roja_delta": 5.0,
        "tiempo_amarillo_min": 10,
        "tiempo_rojo_min": 5,
        "productos": ["salmon_atlantico","trucha","salmon_pacifico"],
        "costo_reclamo_usd": {"min": 25000, "max": 80000},
    },
    "congelado": {
        "setpoint_min": -20.0,
        "setpoint_max": -18.0,
        "alerta_amarilla_delta": 3.0,
        "alerta_roja_delta": 5.0,
        "tiempo_amarillo_min": 20,
        "tiempo_rojo_min": 10,
        "productos": ["jurel_congelado","salmon_congelado","camaron"],
        "costo_reclamo_usd": {"min": 5000, "max": 20000},
    },
    "celulosa": {
        "setpoint_min": 5.0,
        "setpoint_max": 35.0,
        "alerta_amarilla_delta": 15.0,
        "alerta_roja_delta": 25.0,
        "tiempo_amarillo_min": 60,
        "tiempo_rojo_min": 30,
        "productos": ["celulosa_blanqueada","celulosa_kraft"],
        "costo_reclamo_usd": {"min": 1000, "max": 8000},
    },
}

# ══════════════════════════════════════════════════════════════
# MODELO DE PRICING (5 fuentes de ingreso)
# ══════════════════════════════════════════════════════════════
PRICING = {
    # FUENTE 1 — SaaS mensual (CLP)
    "saas_mensual": {
        "puerto_suite_completa": 12_500_000,
        "exportador_grande": 3_800_000,
        "exportador_mediano": 1_400_000,
        "empresa_acuicola": 4_500_000,
        "naviera": 2_200_000,
    },
    # FUENTE 2 — Revenue Share
    "revenue_share": {
        "comision_teu_recuperado_clp": 25_000,
        "certificado_usda_senasa_clp": 18_000,
        "contenedor_asegurado_sin_reclamo_clp": 8_000,
    },
    # FUENTE 3 — Data Products (CLP/mes)
    "data_products": {
        "indice_escasez_reefer_naviera": 2_800_000,
        "indice_escasez_reefer_exportador": 1_200_000,
        "score_riesgo_logistico_banco": 4_500_000,
        "benchmark_operacional_corfo": 1_500_000,
    },
    # FUENTE 4 — Professional Services (CLP único)
    "professional_services": {
        "implementacion_puerto": 45_000_000,
        "consultoria_captura_carga": 18_000_000,
        "protocolo_neuquen": 24_000_000,
        "onboarding_exportador": 6_500_000,
        "auditoria_cadena_frio": 12_000_000,
    },
    # FUENTE 5 — CASST MRR (USD → CLP ~930)
    "casst_mrr_usd": 16_667,
}

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE MODELOS ML
# ══════════════════════════════════════════════════════════════
ML_CONFIG = {
    "seed": 2026,
    "isolation_forest": {
        "contamination": 0.02,
        "n_estimators": 200,
        "max_samples": "auto",
    },
    "prophet": {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.05,
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 2026,
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 2026,
    },
    "monte_carlo": {
        "n_scenarios": 50_000,
        "alpha_var": 0.05,
        "seed": 2026,
    },
    "garch": {
        "p": 1, "q": 1,
        "vol": "GARCH",
        "dist": "normal",
    },
}

# ══════════════════════════════════════════════════════════════
# STRESS TESTING — 4 escenarios históricos verificados
# ══════════════════════════════════════════════════════════════
STRESS_SCENARIOS = {
    "corte_energia_72h": {
        "descripcion": "Apagón Biobío 72h — reefers sin conexión",
        "n_reefers_afectados": 340,
        "perdida_estimada_clp": 2_800_000_000,
        "duracion_horas": 72,
    },
    "nevada_pino_hachado": {
        "descripcion": "Nevada extrema Pino Hachado 8 días — corredor Neuquén bloqueado",
        "camiones_bloqueados": 180,
        "perdida_estimada_clp": 4_200_000_000,
        "duracion_dias": 8,
    },
    "escasez_reefer_2024": {
        "descripcion": "Escasez extrema reefer temporada 2024",
        "contenedores_desviados": 8_900,
        "perdida_ingreso_clp": 1_690_000_000,
    },
    "huelga_portuaria": {
        "descripcion": "Huelga 5 días — operación paralizada",
        "duracion_dias": 5,
        "perdida_estimada_clp": 890_000_000,
    },
}

# ══════════════════════════════════════════════════════════════
# HOSTING GRATUITO — Configuración Railway/Render/Streamlit Cloud
# ══════════════════════════════════════════════════════════════
HOSTING = {
    "api":       "Railway.app (FastAPI — tier gratuito 500h/mes)",
    "dashboard": "Streamlit Community Cloud (gratis ilimitado)",
    "db":        "Supabase PostgreSQL (gratis 500MB)",
    "redis":     "Upstash Redis (gratis 10k req/día)",
    "monitor":   "Grafana Cloud (gratis 10k métricas)",
    "dominio":   "coldroute.streamlit.app + coldroute-api.railway.app",
}

class Settings(BaseSettings):
    """Configuración dinámica vía variables de entorno"""
    APP_NAME: str = "ColdRoute Intelligence Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ANTHROPIC_API_KEY: str = ""
    DATABASE_URL: str = "sqlite:///./coldroute_dev.db"
    REDIS_URL: str = "redis://localhost:6379"
    SECRET_KEY: str = "coldroute-dev-secret-2026"
    SEED: int = 2026

    class Config:
        env_file = ".env"

settings = Settings()
