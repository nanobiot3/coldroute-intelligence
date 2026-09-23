"""
ColdRoute — Agente 1: Detector de Anomalías de Cadena de Frío
Capa 2 Metodología JP Morgan: Analítica en Tiempo Real
Algoritmo: Isolation Forest (Liu, Ting & Zhou, 2008)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '/home/claude/coldroute')
from config.settings import TEMP_THRESHOLDS, ML_CONFIG

class AgenteDetectorAnomalias:
    """
    Detecta rupturas de cadena de frío en contenedores reefer
    mediante aprendizaje no supervisado + reglas de dominio físico.
    Opera en tiempo real sobre lecturas IoT cada 15 minutos.
    """

    def __init__(self):
        self.seed = ML_CONFIG["seed"]
        cfg = ML_CONFIG["isolation_forest"]
        self.modelo = IsolationForest(
            contamination=cfg["contamination"],
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            random_state=self.seed,
        )
        self.scaler = StandardScaler()
        self.entrenado = False
        self.historial_alertas: List[Dict] = []

    def _extraer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Features de ingeniería para detección de anomalías:
        - temperatura actual
        - delta respecto al setpoint
        - variación entre lecturas consecutivas (velocidad de cambio)
        - media móvil 4 lecturas (1 hora)
        - desviación de media móvil
        """
        df = df.sort_values("timestamp").copy()
        df["delta_temp"] = df["temperatura"] - df["setpoint"]
        df["velocidad"] = df["temperatura"].diff().fillna(0)
        df["media_movil_4"] = df["temperatura"].rolling(4, min_periods=1).mean()
        df["desv_media"] = df["temperatura"] - df["media_movil_4"]

        features = df[[
            "temperatura", "delta_temp",
            "velocidad", "desv_media"
        ]].fillna(0).values

        return features

    def entrenar(self, df_historico: pd.DataFrame) -> Dict:
        """
        Entrena el modelo con historial de lecturas normales
        """
        X = self._extraer_features(df_historico)
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled)
        self.entrenado = True

        scores = self.modelo.score_samples(X_scaled)
        return {
            "estado": "entrenado",
            "n_muestras": len(X),
            "score_medio": float(np.mean(scores)),
            "score_min": float(np.min(scores)),
            "contamination": ML_CONFIG["isolation_forest"]["contamination"],
        }

    def detectar(self, df_contenedor: pd.DataFrame,
                  tipo_carga: str) -> Dict:
        """
        Evalúa las últimas lecturas de un contenedor específico.
        Combina Isolation Forest + reglas físicas de temperatura.
        Devuelve nivel de alerta: VERDE / AMARILLO / ROJO
        """
        if not self.entrenado:
            # Auto-entrenamiento con datos del mismo contenedor
            self.entrenar(df_contenedor)

        X = self._extraer_features(df_contenedor)
        X_scaled = self.scaler.transform(X)
        predicciones = self.modelo.predict(X_scaled)  # 1=normal, -1=anomalía
        scores = self.modelo.score_samples(X_scaled)

        # Análisis de las últimas 4 lecturas (1 hora)
        ultimas = df_contenedor.tail(4).copy()
        umbral = TEMP_THRESHOLDS[tipo_carga]
        setpoint = df_contenedor["setpoint"].iloc[-1]
        temp_actual = df_contenedor["temperatura"].iloc[-1]
        delta_actual = temp_actual - setpoint

        # === REGLAS FÍSICAS DE DOMINIO (primero que Isolation Forest) ===
        alerta_fisica = "VERDE"
        tiempo_exceso_min = 0

        # Contar lecturas consecutivas fuera de rango
        for _, row in ultimas.iterrows():
            if abs(row["temperatura"] - setpoint) > umbral["alerta_roja_delta"]:
                tiempo_exceso_min += 15

        if tiempo_exceso_min >= umbral["tiempo_rojo_min"]:
            alerta_fisica = "ROJO"
        elif tiempo_exceso_min >= umbral["tiempo_amarillo_min"]:
            alerta_fisica = "AMARILLO"
        elif abs(delta_actual) > umbral["alerta_amarilla_delta"]:
            alerta_fisica = "AMARILLO"

        # === ISOLATION FOREST (refuerzo estadístico) ===
        n_anomalas_recientes = int((predicciones[-4:] == -1).sum())
        score_reciente = float(np.mean(scores[-4:]))

        # Nivel final: lo peor entre regla física y modelo ML
        if alerta_fisica == "ROJO" or (n_anomalas_recientes >= 3 and alerta_fisica != "VERDE"):
            nivel_final = "ROJO"
        elif alerta_fisica == "AMARILLO" or n_anomalas_recientes >= 2:
            nivel_final = "AMARILLO"
        else:
            nivel_final = "VERDE"

        # Calcular pérdida estimada
        rango_costo = umbral["costo_reclamo_usd"]
        prob_reclamo = {"VERDE": 0.01, "AMARILLO": 0.25, "ROJO": 0.72}
        perdida_esperada_usd = (
            (rango_costo["min"] + rango_costo["max"]) / 2
            * prob_reclamo[nivel_final]
        )

        alerta = {
            "contenedor_id": df_contenedor["contenedor_id"].iloc[-1],
            "tipo_carga": tipo_carga,
            "nivel": nivel_final,
            "temperatura_actual": round(float(temp_actual), 2),
            "setpoint": round(float(setpoint), 2),
            "delta_temp": round(float(delta_actual), 2),
            "tiempo_exceso_min": tiempo_exceso_min,
            "n_lecturas_anomalas_1h": n_anomalas_recientes,
            "score_isolation_forest": round(score_reciente, 4),
            "perdida_esperada_usd": round(perdida_esperada_usd, 0),
            "accion_recomendada": self._accion(nivel_final, tipo_carga),
            "timestamp": df_contenedor["timestamp"].iloc[-1],
        }

        self.historial_alertas.append(alerta)
        return alerta

    def _accion(self, nivel: str, tipo_carga: str) -> str:
        acciones = {
            "VERDE": "Monitoreo ordinario. Sin acción requerida.",
            "AMARILLO": (
                "VERIFICAR conexión eléctrica del rack. "
                "Notificar supervisor de turno. "
                "Re-evaluar en próxima lectura (15 min)."
            ),
            "ROJO": (
                "EMERGENCIA CADENA DE FRÍO. "
                "Notificar exportador + operador portuario + aseguradora. "
                "Activar protocolo de emergencia inmediato. "
                "Registrar evento para certificado USDA/SENASA. "
                "Evaluar traslado a reefer de respaldo."
            ),
        }
        if nivel == "ROJO" and tipo_carga == "salmon_fresco":
            return acciones["ROJO"] + " PRIORIDAD MÁXIMA: salmón fresco tolerancia cero."
        return acciones[nivel]

    def resumen_portafolio(self,
                            df_meta: pd.DataFrame,
                            df_series: pd.DataFrame) -> Dict:
        """
        Evalúa todos los contenedores del patio y devuelve
        el resumen del War Room (Capa JP Morgan — 26 KPIs)
        """
        alertas_totales = []

        # Entrenar con datos normales
        df_normales = df_series[
            ~df_series["contenedor_id"].str.startswith("ANOM")
        ]
        if len(df_normales) > 100:
            self.entrenar(df_normales)

        for _, cont in df_meta.iterrows():
            df_cont = df_series[
                df_series["contenedor_id"] == cont["contenedor_id"]
            ].copy()
            if len(df_cont) < 4:
                continue
            alerta = self.detectar(df_cont, cont["tipo_carga"])
            alertas_totales.append(alerta)

        df_alertas = pd.DataFrame(alertas_totales)

        resumen = {
            "total_contenedores": len(alertas_totales),
            "semaforo": {
                "VERDE":    int((df_alertas["nivel"] == "VERDE").sum()),
                "AMARILLO": int((df_alertas["nivel"] == "AMARILLO").sum()),
                "ROJO":     int((df_alertas["nivel"] == "ROJO").sum()),
            },
            "perdida_total_esperada_usd": float(df_alertas["perdida_esperada_usd"].sum()),
            "contenedores_criticos": df_alertas[
                df_alertas["nivel"] == "ROJO"
            ][["contenedor_id","tipo_carga","temperatura_actual",
               "delta_temp","perdida_esperada_usd","accion_recomendada"]].to_dict("records"),
            "temperatura_media_patio": float(df_alertas["temperatura_actual"].mean()),
            "score_if_medio": float(df_alertas["score_isolation_forest"].mean()),
        }
        return resumen


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🤖 Agente 1 — Detector de Anomalías iniciando...")
    from data.simulador_iot import SimuladorReeferIoT
    sim = SimuladorReeferIoT()
    df_meta, df_series = sim.generar_portafolio_puerto(n_normales=60, n_anomalos=5)

    agente = AgenteDetectorAnomalias()
    resumen = agente.resumen_portafolio(df_meta, df_series)

    print(f"\n📊 WAR ROOM — Puerto Coronel (tiempo real)")
    print(f"   Total contenedores monitorizados: {resumen['total_contenedores']}")
    s = resumen["semaforo"]
    print(f"   🟢 VERDE:    {s['VERDE']} contenedores")
    print(f"   🟡 AMARILLO: {s['AMARILLO']} contenedores")
    print(f"   🔴 ROJO:     {s['ROJO']} contenedores")
    print(f"   📉 Pérdida esperada portafolio: US${resumen['perdida_total_esperada_usd']:,.0f}")
    print(f"   🌡️  Temperatura media patio: {resumen['temperatura_media_patio']:.1f}°C")
    if resumen["contenedores_criticos"]:
        print(f"\n🚨 ALERTAS CRÍTICAS:")
        for c in resumen["contenedores_criticos"][:3]:
            print(f"   {c['contenedor_id']} | {c['tipo_carga']} | "
                  f"T={c['temperatura_actual']}°C | "
                  f"Δ={c['delta_temp']:+.1f}°C | "
                  f"Pérdida est. US${c['perdida_esperada_usd']:,.0f}")
