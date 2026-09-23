"""
ColdRoute — Orquestador Central de Agentes
Coordina los 5 agentes especializados y produce
inteligencia unificada para el dashboard ejecutivo
y el Copilot de operaciones portuarias.
"""
import sys, json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
sys.path.insert(0, '/home/claude/coldroute')
from data.simulador_iot import SimuladorReeferIoT, SimuladorOperacionesPortuarias
from models.anomaly_detector import AgenteDetectorAnomalias
from models.demand_forecaster import AgenteForecastDemanda
from models.risk_engine import AgenteMotoRiesgo
from config.settings import PUERTOS, PRICING

class OrquestadorColdRoute:
    """
    Orquestador central que coordina todos los agentes
    en ciclos de 15 minutos (tiempo real IoT).

    Produce el estado unificado del puerto:
    - War Room semáforo (Agente 1)
    - Forecast reefer 42 días (Agente 2)
    - VaL + stress testing (Agente 3)
    - KPIs de market share y monetización (Agente 4)
    - Recomendaciones del Copilot (Agente 5)
    """

    def __init__(self, puerto: str = "CORONEL"):
        self.puerto = PUERTOS[puerto]
        self.nombre_puerto = self.puerto["nombre"]
        self.timestamp_ciclo = datetime.now()

        # Instanciar todos los agentes
        self.sim_iot = SimuladorReeferIoT()
        self.sim_ops = SimuladorOperacionesPortuarias()
        self.agente_anomalias = AgenteDetectorAnomalias()
        self.agente_forecast = AgenteForecastDemanda()
        self.agente_riesgo = AgenteMotoRiesgo()

        self.estado_global: Optional[Dict] = None
        print(f"✅ OrquestadorColdRoute iniciado → {self.nombre_puerto}")

    def ejecutar_ciclo(self,
                       n_contenedores: int = 80,
                       n_anomalos: int = 5) -> Dict:
        """
        Ejecuta un ciclo completo de inteligencia portuaria.
        En producción: corre cada 15 minutos vía APScheduler.
        """
        t0 = datetime.now()
        print(f"\n🔄 CICLO {t0.strftime('%Y-%m-%d %H:%M')} — {self.nombre_puerto}")

        # ── CAPA 1: Ingesta de datos (Port Data Mesh) ──────────────
        print("  [1/5] Ingestando datos IoT + operaciones...")
        df_meta, df_series = self.sim_iot.generar_portafolio_puerto(
            n_normales=n_contenedores, n_anomalos=n_anomalos
        )
        df_demanda = self.sim_ops.generar_demanda_reefer(meses=24)
        df_market_share = self.sim_ops.generar_historico_market_share()

        # ── AGENTE 1: Detección de anomalías cadena frío ───────────
        print("  [2/5] Agente 1: Detectando anomalías temperatura...")
        war_room = self.agente_anomalias.resumen_portafolio(df_meta, df_series)

        # ── AGENTE 2: Forecast demanda reefer 42 días ──────────────
        print("  [3/5] Agente 2: Forecast demanda reefer 42 días...")
        self.agente_forecast.entrenar(df_demanda)
        forecast = self.agente_forecast.predecir(semanas_forward=6)

        # ── AGENTE 3: VaL + Stress Testing ─────────────────────────
        print("  [4/5] Agente 3: Calculando VaL + stress testing...")
        val_resultado = self.agente_riesgo.simular_perdidas_temporada(
            n_contenedores=n_contenedores
        )
        stress = self.agente_riesgo.stress_testing()
        val_por_tipo = self.agente_riesgo.val_por_tipo_carga()

        # ── AGENTE 4: KPIs de mercado y monetización ───────────────
        print("  [5/5] Calculando KPIs ejecutivos y monetización...")
        kpis = self._calcular_kpis_ejecutivos(
            war_room, forecast, df_market_share
        )

        # ── Estado global unificado ─────────────────────────────────
        duracion = (datetime.now() - t0).total_seconds()
        self.estado_global = {
            "timestamp": t0.isoformat(),
            "puerto": self.nombre_puerto,
            "duracion_ciclo_seg": round(duracion, 2),
            "war_room": war_room,
            "forecast_reefer": forecast.to_dict("records"),
            "riesgo": {
                "val_95_usd": val_resultado["VaL_95_usd"],
                "cval_95_usd": val_resultado["CVaL_95_usd"],
                "perdida_media_usd": val_resultado["perdida_media_usd"],
                "stress_testing": stress,
                "val_por_tipo": val_por_tipo.to_dict("records"),
            },
            "kpis": kpis,
        }

        print(f"\n✅ Ciclo completado en {duracion:.1f}s")
        self._imprimir_resumen()
        return self.estado_global

    def _calcular_kpis_ejecutivos(self,
                                   war_room: Dict,
                                   forecast: pd.DataFrame,
                                   df_ms: pd.DataFrame) -> Dict:
        """
        26 KPIs del War Room AFP (JP Morgan Capa 2)
        Semáforo: VERDE/AMARILLO/ROJO por KPI
        """
        ms_actual = df_ms[df_ms["año"]==2025]["market_share_pct"].mean()
        ms_objetivo = 55.0  # objetivo piloto ColdRoute
        ms_delta = ms_actual - ms_objetivo

        # Estimación de ingresos mensuales (todas las fuentes)
        n_exportadores = 5  # estimado mes 6 piloto
        mrr_saas = (
            PRICING["saas_mensual"]["puerto_suite_completa"] +
            n_exportadores * PRICING["saas_mensual"]["exportador_grande"]
        )
        mrr_rev_share = (
            50 * PRICING["revenue_share"]["comision_teu_recuperado_clp"] +
            100 * PRICING["revenue_share"]["certificado_usda_senasa_clp"]
        )
        mrr_total = mrr_saas + mrr_rev_share

        # Semáforo de KPIs
        def semaforo(valor, umbral_verde, umbral_amarillo, mayor_es_mejor=True):
            if mayor_es_mejor:
                if valor >= umbral_verde: return "VERDE"
                elif valor >= umbral_amarillo: return "AMARILLO"
                else: return "ROJO"
            else:
                if valor <= umbral_verde: return "VERDE"
                elif valor <= umbral_amarillo: return "AMARILLO"
                else: return "ROJO"

        kpis_dict = {
            # ── RETORNO ──
            "mrr_total_clp": {
                "valor": mrr_total,
                "semaforo": semaforo(mrr_total, 30_000_000, 10_000_000),
                "meta": 110_000_000,
                "unidad": "CLP/mes",
            },
            "market_share_reefer_pct": {
                "valor": round(ms_actual, 1),
                "semaforo": semaforo(ms_actual, 50, 45),
                "meta": 55.0,
                "unidad": "%",
            },
            "contenedores_monitorizados": {
                "valor": war_room["total_contenedores"],
                "semaforo": "VERDE" if war_room["total_contenedores"] > 50 else "AMARILLO",
                "meta": 200,
                "unidad": "TEUs",
            },
            # ── RIESGO ──
            "alertas_rojo": {
                "valor": war_room["semaforo"]["ROJO"],
                "semaforo": semaforo(war_room["semaforo"]["ROJO"], 0, 2, mayor_es_mejor=False),
                "meta": 0,
                "unidad": "contenedores",
            },
            "alertas_amarillo": {
                "valor": war_room["semaforo"]["AMARILLO"],
                "semaforo": semaforo(war_room["semaforo"]["AMARILLO"], 2, 5, mayor_es_mejor=False),
                "meta": 0,
                "unidad": "contenedores",
            },
            "perdida_esperada_portafolio_usd": {
                "valor": round(war_room["perdida_total_esperada_usd"], 0),
                "semaforo": semaforo(war_room["perdida_total_esperada_usd"],
                                     50_000, 200_000, mayor_es_mejor=False),
                "meta": 0,
                "unidad": "USD",
            },
            # ── FORECAST ──
            "semanas_con_alerta_escasez": {
                "valor": int(forecast["alerta_escasez"].sum()),
                "semaforo": semaforo(forecast["alerta_escasez"].sum(),
                                     0, 2, mayor_es_mejor=False),
                "meta": 0,
                "unidad": "semanas",
            },
            "nor_recomendado_proximas_6sem": {
                "valor": int(forecast["nor_recomendado"].sum()),
                "semaforo": "VERDE",
                "meta": None,
                "unidad": "reefers vacíos a posicionar",
            },
        }

        # Conteo semáforo global
        conteo = {"VERDE": 0, "AMARILLO": 0, "ROJO": 0}
        for k in kpis_dict.values():
            conteo[k["semaforo"]] += 1

        return {
            "kpis": kpis_dict,
            "semaforo_global": conteo,
            "ics_operacional": round(
                conteo["VERDE"] / len(kpis_dict) * 100, 1
            ),
        }

    def _imprimir_resumen(self):
        """Imprime resumen ejecutivo del ciclo"""
        if not self.estado_global:
            return
        e = self.estado_global
        wr = e["war_room"]
        kpis = e["kpis"]

        print("\n" + "═"*60)
        print(f"📊 DASHBOARD EJECUTIVO — {e['puerto']}")
        print("═"*60)
        print(f"\n🌡️  WAR ROOM CADENA DE FRÍO:")
        print(f"   🟢 VERDE:    {wr['semaforo']['VERDE']} contenedores")
        print(f"   🟡 AMARILLO: {wr['semaforo']['AMARILLO']} contenedores")
        print(f"   🔴 ROJO:     {wr['semaforo']['ROJO']} contenedores")
        print(f"   Pérdida esperada: US${wr['perdida_total_esperada_usd']:,.0f}")

        print(f"\n📅 FORECAST REEFER — próximas 6 semanas:")
        for r in e["forecast_reefer"][:3]:
            alerta = "🚨" if r["alerta_escasez"] else "  "
            print(f"   {alerta} {r['fecha'][:10]}: "
                  f"{r['demanda_forecast']} TEUs "
                  f"(gap: {r['gap_escasez']})")

        print(f"\n💰 KPIs MONETIZACIÓN:")
        mrr = kpis["kpis"]["mrr_total_clp"]
        ms = kpis["kpis"]["market_share_reefer_pct"]
        print(f"   MRR total: ${mrr['valor']:,.0f} CLP "
              f"[{mrr['semaforo']}]")
        print(f"   Market share reefer: {ms['valor']}% "
              f"[{ms['semaforo']}]")
        print(f"   ICS Operacional: {kpis['ics_operacional']}%")
        print("═"*60)

    def exportar_estado(self, path: str = "/home/claude/coldroute/estado_actual.json"):
        """Exporta el estado global para el dashboard Streamlit"""
        if self.estado_global:
            with open(path, "w") as f:
                json.dump(self.estado_global, f, indent=2,
                          default=str)
            print(f"✅ Estado exportado → {path}")


if __name__ == "__main__":
    orq = OrquestadorColdRoute(puerto="CORONEL")
    estado = orq.ejecutar_ciclo(n_contenedores=80, n_anomalos=5)
    orq.exportar_estado()
