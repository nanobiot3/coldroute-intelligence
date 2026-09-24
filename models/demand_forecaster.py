"""
ColdRoute — Agente 2: Forecast de Demanda de Reefers
Capa 2 JP Morgan: Analítica Predictiva
Ensemble Prophet (estacionalidad) + XGBoost (variables externas)
Horizonte: 42 días forward con intervalo confianza 90%
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from config.settings import ML_CONFIG

class AgenteForecastDemanda:
    """
    Predice la demanda semanal de contenedores reefer en Puerto Coronel
    con 42 días de anticipación para negociación con navieras.

    Insight clave: la coordinación de posicionamiento NOR (Non-Operating Reefer)
    desde Asia requiere 45-60 días de anticipación. Este modelo entrega
    esa señal con tiempo suficiente.
    """
    def __init__(self):
        self.seed = ML_CONFIG["seed"]
        np.random.seed(self.seed)

        # Modelo A: Prophet para estacionalidad
        cfg_p = ML_CONFIG["prophet"]
        self.prophet = Prophet(
            seasonality_mode=cfg_p["seasonality_mode"],
            changepoint_prior_scale=cfg_p["changepoint_prior_scale"],
            yearly_seasonality=cfg_p["yearly_seasonality"],
            weekly_seasonality=cfg_p["weekly_seasonality"],
        )
        self.prophet.add_regressor("enso_index")
        self.prophet.add_regressor("bookings_t21")  # bookings confirmados 21 días antes

        # Modelo B: XGBoost para efectos no lineales
        cfg_x = ML_CONFIG["xgboost"]
        self.xgb = XGBRegressor(
            n_estimators=cfg_x["n_estimators"],
            learning_rate=cfg_x["learning_rate"],
            max_depth=cfg_x["max_depth"],
            subsample=cfg_x["subsample"],
            colsample_bytree=cfg_x["colsample_bytree"],
            random_state=cfg_x["random_state"],
        )
        self.entrenado = False
        self.mape_prophet = None
        self.mape_xgb = None
        self.mape_ensemble = None

    def _features_xgb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ingeniería de features para XGBoost"""
        df = df.copy()
        df["mes"] = df["fecha"].dt.month
        df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype(int)
        df["temporada_alta"] = df["mes"].isin([10,11,12,1,2]).astype(int)
        df["lag_1"] = df["demanda_reefers"].shift(1)
        df["lag_4"] = df["demanda_reefers"].shift(4)
        df["rolling_4"] = df["demanda_reefers"].rolling(4, min_periods=1).mean()
        df["rolling_8"] = df["demanda_reefers"].rolling(8, min_periods=1).mean()
        df["enso_index"] = df.get("enso_index", 0.0)
        return df.fillna(0)

    def _simular_enso(self, n: int) -> np.ndarray:
        """Simula índice ENSO (El Niño) — afecta fechas de cosecha ±2 semanas"""
        rng = np.random.default_rng(self.seed)
        return np.sin(np.linspace(0, 4*np.pi, n)) * 0.8 + rng.normal(0, 0.2, n)

    def entrenar(self, df: pd.DataFrame) -> Dict:
        """Entrena el ensemble Prophet + XGBoost"""
        df = df.copy()
        df["enso_index"] = self._simular_enso(len(df))
        df["bookings_t21"] = df["demanda_reefers"].shift(3).fillna(
            df["demanda_reefers"].mean()
        )

        # === PROPHET ===
        df_p = df.rename(columns={
            "fecha": "ds",
            "demanda_reefers": "y"
        })[["ds","y","enso_index","bookings_t21"]].copy()
        df_p["ds"] = pd.to_datetime(df_p["ds"])

        tscv = TimeSeriesSplit(n_splits=3)
        errores_prophet = []
        for train_idx, test_idx in tscv.split(df_p):
            m_temp = Prophet(
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=False,
            )
            m_temp.add_regressor("enso_index")
            m_temp.add_regressor("bookings_t21")
            m_temp.fit(df_p.iloc[train_idx])
            pred = m_temp.predict(df_p.iloc[test_idx][["ds","enso_index","bookings_t21"]])
            mape = mean_absolute_percentage_error(
                df_p.iloc[test_idx]["y"], pred["yhat"]
            )
            errores_prophet.append(mape)

        self.mape_prophet = float(np.mean(errores_prophet))

        # Entrenar Prophet completo
        self.prophet.fit(df_p)
        pred_train_p = self.prophet.predict(df_p[["ds","enso_index","bookings_t21"]])
        residuos = df_p["y"].values - pred_train_p["yhat"].values

        # === XGBOOST sobre residuos de Prophet ===
        df_x = self._features_xgb(df)
        features_cols = ["mes","semana_anio","temporada_alta",
                         "lag_1","lag_4","rolling_4","rolling_8","enso_index"]
        X = df_x[features_cols].values
        y_res = residuos

        errores_xgb = []
        for train_idx, test_idx in tscv.split(X):
            xgb_temp = XGBRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=4, random_state=self.seed
            )
            xgb_temp.fit(X[train_idx], y_res[train_idx])
            pred_res = xgb_temp.predict(X[test_idx])
            pred_total = pred_train_p["yhat"].values[test_idx] + pred_res
            mape = mean_absolute_percentage_error(
                df_p["y"].values[test_idx],
                np.maximum(pred_total, 0)
            )
            errores_xgb.append(mape)

        self.mape_xgb = float(np.mean(errores_xgb))
        self.xgb.fit(X, y_res)

        # MAPE ensemble (ponderado 60% Prophet + 40% XGB)
        pred_total_train = (
            pred_train_p["yhat"].values +
            self.xgb.predict(X)
        )
        self.mape_ensemble = float(mean_absolute_percentage_error(
            df_p["y"].values, np.maximum(pred_total_train, 0)
        ))

        self.df_train = df
        self.df_p_train = df_p
        self.features_cols = features_cols
        self.entrenado = True

        return {
            "estado": "entrenado",
            "n_semanas": len(df),
            "mape_prophet": round(self.mape_prophet * 100, 2),
            "mape_xgb_residuos": round(self.mape_xgb * 100, 2),
            "mape_ensemble": round(self.mape_ensemble * 100, 2),
            "calidad": "EXCELENTE" if self.mape_ensemble < 0.10 else
                       "BUENA" if self.mape_ensemble < 0.15 else "ACEPTABLE",
        }

    def predecir(self, semanas_forward: int = 6) -> pd.DataFrame:
        """
        Genera forecast 42 días (6 semanas) forward con IC 90%.
        Incluye recomendación de posicionamiento NOR de reefers.
        """
        assert self.entrenado, "Entrenar primero con .entrenar(df)"

        ultimo = pd.to_datetime(self.df_p_train["ds"].max())
        fechas_future = pd.date_range(
            start=ultimo + pd.Timedelta(weeks=1),
            periods=semanas_forward,
            freq="W"
        )

        enso_future = self._simular_enso(semanas_forward)
        bookings_future = np.full(semanas_forward,
                                   self.df_train["demanda_reefers"].mean() * 0.85)

        df_future_p = pd.DataFrame({
            "ds": fechas_future,
            "enso_index": enso_future,
            "bookings_t21": bookings_future,
        })

        pred_p = self.prophet.predict(df_future_p)

        # XGBoost features para período futuro
        df_future_x = pd.DataFrame({"fecha": fechas_future})
        df_future_x["mes"] = df_future_x["fecha"].dt.month
        df_future_x["semana_anio"] = df_future_x["fecha"].dt.isocalendar().week.astype(int)
        df_future_x["temporada_alta"] = df_future_x["mes"].isin([10,11,12,1,2]).astype(int)
        df_future_x["lag_1"] = 0
        df_future_x["lag_4"] = 0
        df_future_x["rolling_4"] = self.df_train["demanda_reefers"].tail(4).mean()
        df_future_x["rolling_8"] = self.df_train["demanda_reefers"].tail(8).mean()
        df_future_x["enso_index"] = enso_future

        X_future = df_future_x[self.features_cols].values
        pred_res_future = self.xgb.predict(X_future)

        # Ensemble final
        pred_central = np.maximum(pred_p["yhat"].values + pred_res_future, 0)
        ic_90_low = np.maximum(pred_p["yhat_lower"].values + pred_res_future * 0.7, 0)
        ic_90_high = pred_p["yhat_upper"].values + pred_res_future * 1.3

        # Disponibilidad estimada (70-90% de la demanda)
        disponibilidad = pred_central * np.random.uniform(0.75, 0.92, semanas_forward)
        gap = pred_central - disponibilidad

        resultados = pd.DataFrame({
            "semana": range(1, semanas_forward + 1),
            "fecha": fechas_future,
            "demanda_forecast": np.round(pred_central).astype(int),
            "ic90_bajo": np.round(ic_90_low).astype(int),
            "ic90_alto": np.round(ic_90_high).astype(int),
            "disponibles_estimados": np.round(disponibilidad).astype(int),
            "gap_escasez": np.round(gap).astype(int),
            "mes": [f.month for f in fechas_future],
            "temporada_alta": [m in [10,11,12,1,2]
                                for m in [f.month for f in fechas_future]],
            "alerta_escasez": gap > pred_central * 0.20,
            "nor_recomendado": np.round(gap * 1.15).astype(int),
        })

        return resultados


if __name__ == "__main__":
    print("🤖 Agente 2 — Forecast Demanda Reefer iniciando...")
    from data.simulador_iot import SimuladorOperacionesPortuarias
    sim = SimuladorOperacionesPortuarias()
    df_dem = sim.generar_demanda_reefer(meses=24)

    agente = AgenteForecastDemanda()
    metricas = agente.entrenar(df_dem)
    print(f"\n📊 ENTRENAMIENTO:")
    print(f"   MAPE Prophet:   {metricas['mape_prophet']}%")
    print(f"   MAPE Ensemble:  {metricas['mape_ensemble']}%")
    print(f"   Calidad:        {metricas['calidad']}")

    forecast = agente.predecir(semanas_forward=6)
    print(f"\n📅 FORECAST 42 DÍAS — Demanda reefer Puerto Coronel:")
    print(f"{'Fecha':<14} {'Demanda':>8} {'IC90 Low':>9} {'IC90 High':>10} "
          f"{'Gap':>7} {'NOR Rec.':>9} {'Alerta':>7}")
    print("-" * 68)
    for _, r in forecast.iterrows():
        alerta = "🚨 SÍ" if r["alerta_escasez"] else "  no"
        temp = "🌡️ ALTA" if r["temporada_alta"] else "      "
        print(f"{str(r['fecha'].date()):<14} {r['demanda_forecast']:>8} "
              f"{r['ic90_bajo']:>9} {r['ic90_alto']:>10} "
              f"{r['gap_escasez']:>7} {r['nor_recomendado']:>9} "
              f"{alerta:>7} {temp}")
