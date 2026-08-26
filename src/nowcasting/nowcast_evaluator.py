"""
Avaliador de Performance do Nowcast vs. Forecast Puro (Pseudo Real-Time Backtest).

Compara o ganho de precisão preditiva obtido ao incluir dados alternativos de alta frequência
no mês corrente (Nowcast) em relação aos modelos de projeção tradicionais (Forecast).
"""

import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from .midas_bridge_model import MIDASBridgeModel, XGBoostNowcaster

logger = logging.getLogger(__name__)


class NowcastEvaluator:
    """
    Avaliador empírico do ganho do Nowcasting via Pseudo Real-Time Backtesting.
    """

    def __init__(self, test_size: int = 12) -> None:
        """
        Inicializa o avaliador.

        Parâmetros:
            test_size: Número de meses no conjunto de teste out-of-sample.
        """
        self.test_size = test_size

    def evaluate_pseudo_realtime(
        self,
        X_nowcast: pd.DataFrame,
        X_forecast: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executa o teste comparativo entre o Nowcast e o Forecast Puro.

        Parâmetros:
            X_nowcast: Matriz de atributos contendo dados de alta frequência alinhados.
            X_forecast: Matriz de atributos contendo apenas lags da série-alvo (Forecast puro).
            y: Série temporal observada (ground truth).

        Retorno:
            Tuple (df_metrics, df_predictions)
        """
        logger.info(f"Iniciando Pseudo Real-Time Backtest (Últimos {self.test_size} meses)...")

        # Split Treino / Teste temporal rígido
        n_obs = len(y)
        train_idx = n_obs - self.test_size

        X_nc_train, X_nc_test = X_nowcast.iloc[:train_idx], X_nowcast.iloc[train_idx:]
        X_fc_train, X_fc_test = X_forecast.iloc[:train_idx], X_forecast.iloc[train_idx:]
        y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

        # 1. Treinar modelos de Nowcast
        bridge_nc = MIDASBridgeModel(alpha=0.5, l1_ratio=0.5)
        bridge_nc.fit(X_nc_train, y_train)
        y_pred_nc_bridge = bridge_nc.predict(X_nc_test)

        xgb_nc = XGBoostNowcaster(max_depth=4, learning_rate=0.03)
        xgb_nc.fit(X_nc_train, y_train)
        y_pred_nc_xgb = xgb_nc.predict(X_nc_test)

        # 2. Treinar modelo de Forecast Puro (Baseline AR)
        bridge_fc = MIDASBridgeModel(alpha=1.0, l1_ratio=0.0)  # Ridge apenas sobre lags
        bridge_fc.fit(X_fc_train, y_train)
        y_pred_fc_baseline = bridge_fc.predict(X_fc_test)

        # 3. Calcular Métricas de Avaliação
        metrics_dict = {}

        models_preds = {
            "Forecast_Puro_Baseline": y_pred_fc_baseline,
            "Nowcast_Bridge_Equation": y_pred_nc_bridge,
            "Nowcast_XGBoost": y_pred_nc_xgb,
        }

        base_rmse = root_mean_squared_error(y_test, y_pred_fc_baseline)

        for name, preds in models_preds.items():
            rmse = root_mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mape = mean_absolute_percentage_error(y_test, preds) * 100
            
            # Ganho de precisão relativo ao Forecast Puro
            rmse_reduction_pct = ((base_rmse - rmse) / base_rmse) * 100.0

            metrics_dict[name] = {
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE (%)": round(mape, 4),
                "Ganho RMSE vs Forecast (%)": round(rmse_reduction_pct, 2),
            }

        df_metrics = pd.DataFrame(metrics_dict).T

        # DataFrame de Previsões
        df_predictions = pd.DataFrame(
            {
                "y_real": y_test,
                "Forecast_Puro_Baseline": y_pred_fc_baseline,
                "Nowcast_Bridge_Equation": y_pred_nc_bridge,
                "Nowcast_XGBoost": y_pred_nc_xgb,
            },
            index=y_test.index,
        )

        logger.info(f">> Resultados da avaliação do Nowcast:\n{df_metrics}")
        return df_metrics, df_predictions

    def export_results(
        self,
        df_metrics: pd.DataFrame,
        df_predictions: pd.DataFrame,
        output_dir: str = "results",
    ) -> Tuple[Path, Path]:
        """
        Salva os DataFrames de métricas e previsões em arquivo CSV.
        """
        path_dir = Path(output_dir)
        path_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = path_dir / "nowcast_vs_forecast_comparison.csv"
        preds_file = path_dir / "nowcast_predictions_detail.csv"

        df_metrics.to_csv(metrics_file, encoding="utf-8")
        df_predictions.to_csv(preds_file, encoding="utf-8")

        logger.info(f"Resultados do Nowcasting salvos em: {metrics_file} e {preds_file}")
        return metrics_file, preds_file
