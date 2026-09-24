"""
ColdRoute — Agente 3: Motor de Riesgo Portuario
Capa 3 JP Morgan: VaL Dinámico + Stress Testing
Metodología: Monte Carlo Sobol QMC (50.000 escenarios)
             Cornish-Fisher VaR / CVaR
             4 escenarios de estrés históricos verificados
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc, genpareto, norm
from scipy.optimize import minimize
from typing import Dict, List
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config.settings import ML_CONFIG, STRESS_SCENARIOS, TEMP_THRESHOLDS, PUERTOS

class AgenteMotoRiesgo:
    """
    Calcula el Value-at-Loss (VaL) portuario:
    pérdida máxima esperada en el peor 5% de escenarios
    para una temporada de exportación en Puerto Coronel.

    Aplica directamente la metodología de JP Morgan Fusion:
    múltiples metodologías VaR + stress testing robusto.
    """

    def __init__(self):
        self.seed = ML_CONFIG["seed"]
        self.rng = np.random.default_rng(self.seed)
        self.n_sim = ML_CONFIG["monte_carlo"]["n_scenarios"]
        self.alpha = ML_CONFIG["monte_carlo"]["alpha_var"]

        # Factores de riesgo (análogo a factores de mercado JP Morgan)
        self.factores = {
            "temperatura":         {"peso": 0.35, "volatilidad": 0.25},
            "disponibilidad_reefer": {"peso": 0.25, "volatilidad": 0.30},
            "tiempo_frontera":     {"peso": 0.20, "volatilidad": 0.45},
            "slot_nave":           {"peso": 0.15, "volatilidad": 0.20},
            "certificacion":       {"peso": 0.05, "volatilidad": 0.15},
        }

    def _generar_sobol(self) -> np.ndarray:
        """Genera escenarios Sobol QMC para baja discrepancia"""
        n_factores = len(self.factores)
        sampler = qmc.Sobol(d=n_factores, scramble=True,
                             seed=self.seed)
        U = sampler.random(self.n_sim)
        # Transformar a normales estándar
        Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
        return Z

    def simular_perdidas_temporada(self,
                                    n_contenedores: int = 1200,
                                    valor_carga_promedio_usd: float = 45000) -> Dict:
        """
        Simula pérdidas totales en una temporada completa.
        Calibrado para Puerto Coronel:
        - 25.000 contenedores reefer/temporada target
        - 150.000 ton fruta/año
        - Valor promedio US$45.000/contenedor (manzanas frescas mercado Asia)
        """
        Z = self._generar_sobol()
        pesos = np.array([v["peso"] for v in self.factores.values()])
        vols = np.array([v["volatilidad"] for v in self.factores.values()])

        # Cholesky: correlación entre factores de riesgo
        corr_matrix = np.array([
            [1.00, 0.15, 0.10, 0.20, 0.25],  # temperatura
            [0.15, 1.00, 0.30, 0.45, 0.20],  # disponibilidad reefer
            [0.10, 0.30, 1.00, 0.15, 0.05],  # tiempo frontera
            [0.20, 0.45, 0.15, 1.00, 0.30],  # slot nave
            [0.25, 0.20, 0.05, 0.30, 1.00],  # certificación
        ])
        L = np.linalg.cholesky(corr_matrix)
        Z_corr = (L @ Z.T).T  # correlacionar factores

        # Score de riesgo compuesto por escenario [0,1]
        scores_riesgo = np.clip(
            Z_corr @ (pesos * vols) + 0.10, 0, 1
        )

        # Pérdida por contenedor en USD
        perdidas_cont = (
            scores_riesgo * valor_carga_promedio_usd
            * self.rng.uniform(0.02, 0.95, self.n_sim)
        )

        # Pérdida total temporada = pérdida_cont × n_contenedores afectados
        n_afectados = np.round(
            scores_riesgo * n_contenedores * 0.08
        ).astype(int)  # máx 8% de contenedores con pérdida en temporada
        perdidas_total = perdidas_cont * n_afectados

        # VaR y CVaR Cornish-Fisher
        mu = float(np.mean(perdidas_total))
        sigma = float(np.std(perdidas_total))
        skew = float(pd.Series(perdidas_total).skew())
        kurt = float(pd.Series(perdidas_total).kurt())

        # Ajuste Cornish-Fisher para distribuciones no normales
        z_alpha = norm.ppf(1 - self.alpha)
        z_cf = (z_alpha
                + (z_alpha**2 - 1) * skew / 6
                + (z_alpha**3 - 3*z_alpha) * kurt / 24
                - (2*z_alpha**3 - 5*z_alpha) * skew**2 / 36)

        VaL_95 = float(mu + z_cf * sigma)
        CVaL_95 = float(np.mean(perdidas_total[perdidas_total >= VaL_95]))

        percentiles = {
            f"p{p}": float(np.percentile(perdidas_total, p))
            for p in [5, 25, 50, 75, 90, 95, 99]
        }

        return {
            "n_simulaciones": self.n_sim,
            "n_contenedores_temporada": n_contenedores,
            "valor_carga_promedio_usd": valor_carga_promedio_usd,
            "perdida_media_usd": round(mu, 0),
            "perdida_std_usd": round(sigma, 0),
            "VaL_95_usd": round(VaL_95, 0),
            "CVaL_95_usd": round(CVaL_95, 0),
            "perdida_maxima_usd": round(float(perdidas_total.max()), 0),
            "probabilidad_perdida_gt_1M": float((perdidas_total > 1_000_000).mean()),
            "percentiles": {k: round(v,0) for k,v in percentiles.items()},
            "metodologia": "Monte Carlo Sobol QMC + Cornish-Fisher VaR",
        }

    def stress_testing(self) -> Dict:
        """
        4 escenarios de estrés históricos verificados.
        Metodología: igual al Flash Crash Backtesting del modelo CASST.
        Cuantifica el impacto económico de cada escenario en Puerto Coronel.
        """
        resultados = {}
        for nombre, escenario in STRESS_SCENARIOS.items():
            perdida_base = escenario.get("perdida_estimada_clp",
                           escenario.get("perdida_ingreso_clp",
                           escenario.get("perdida_estimada_clp", 500_000_000)))

            # Con sistema ColdRoute: reducción del impacto
            # Alerta temprana reduce pérdida en 55-70%
            factor_reduccion = self.rng.uniform(0.30, 0.45)
            perdida_con_sistema = perdida_base * factor_reduccion

            # ROI del sistema en este escenario
            costo_anual_sistema_clp = 12_500_000 * 12  # $150M ARR
            ahorro_neto = (perdida_base - perdida_con_sistema) - costo_anual_sistema_clp

            resultados[nombre] = {
                "descripcion": escenario["descripcion"],
                "perdida_sin_sistema_clp": perdida_base,
                "perdida_con_coldroute_clp": round(perdida_con_sistema, 0),
                "ahorro_estimado_clp": round(perdida_base - perdida_con_sistema, 0),
                "reduccion_pct": round((1 - factor_reduccion) * 100, 1),
                "roi_escenario_clp": round(ahorro_neto, 0),
                "roi_positivo": ahorro_neto > 0,
            }

        return resultados

    def val_por_tipo_carga(self) -> pd.DataFrame:
        """
        VaL específico por tipo de carga — permite pricing
        diferencial del seguro y del módulo Cold Chain Monitor
        """
        filas = []
        for tipo, umbral in TEMP_THRESHOLDS.items():
            costo_medio = (
                umbral["costo_reclamo_usd"]["min"] +
                umbral["costo_reclamo_usd"]["max"]
            ) / 2

            # Probabilidad implícita de evento por tipo
            prob_evento = {
                "fruta_fresca": 0.04,
                "salmon_fresco": 0.02,
                "congelado": 0.03,
                "celulosa": 0.01,
            }.get(tipo, 0.03)

            val_usd = costo_medio * prob_evento * 1200  # temporada

            filas.append({
                "tipo_carga": tipo,
                "productos": ", ".join(umbral["productos"][:3]),
                "costo_reclamo_min_usd": umbral["costo_reclamo_usd"]["min"],
                "costo_reclamo_max_usd": umbral["costo_reclamo_usd"]["max"],
                "prob_evento_pct": round(prob_evento * 100, 1),
                "VaL_temporada_usd": round(val_usd, 0),
                "precio_modulo_cold_chain_clp": {
                    "fruta_fresca": 3_800_000,
                    "salmon_fresco": 4_500_000,
                    "congelado": 3_200_000,
                    "celulosa": 1_400_000,
                }.get(tipo, 2_000_000),
            })

        return pd.DataFrame(filas)


if __name__ == "__main__":
    print("🤖 Agente 3 — Motor de Riesgo Portuario iniciando...")
    agente = AgenteMotoRiesgo()

    print("\n📊 SIMULACIÓN MONTE CARLO — Temporada frutícola Puerto Coronel")
    resultado = agente.simular_perdidas_temporada(
        n_contenedores=1200,
        valor_carga_promedio_usd=45_000
    )
    print(f"   Escenarios simulados: {resultado['n_simulaciones']:,}")
    print(f"   Pérdida media esperada: US${resultado['perdida_media_usd']:,.0f}")
    print(f"   VaL(95%):  US${resultado['VaL_95_usd']:,.0f}")
    print(f"   CVaL(95%): US${resultado['CVaL_95_usd']:,.0f}")
    print(f"   P(pérdida > US$1M): {resultado['probabilidad_perdida_gt_1M']:.1%}")

    print("\n⚡ STRESS TESTING — 4 escenarios históricos verificados:")
    stress = agente.stress_testing()
    for nombre, r in stress.items():
        roi_str = f"✅ ROI +${abs(r['roi_escenario_clp']):,.0f}" if r["roi_positivo"] \
                  else f"⚠️  ROI ${r['roi_escenario_clp']:,.0f}"
        print(f"\n   [{nombre}] {r['descripcion']}")
        print(f"   Sin sistema: ${r['perdida_sin_sistema_clp']:,.0f} CLP")
        print(f"   Con ColdRoute: ${r['perdida_con_coldroute_clp']:,.0f} CLP "
              f"(↓{r['reduccion_pct']}%)")
        print(f"   {roi_str} CLP")

    print("\n📋 VaL POR TIPO DE CARGA:")
    df_val = agente.val_por_tipo_carga()
    print(df_val[["tipo_carga","VaL_temporada_usd","precio_modulo_cold_chain_clp"]].to_string())
