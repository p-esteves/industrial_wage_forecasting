"""
Módulo de Nowcasting de Alta Frequência (Alternative Data & Mixed Frequency Econometrics).

Oferece coletores de dados de alta frequência (Google Trends, ONS), alinhador de frequências,
modelos de regressão ponte (Bridge Equations, U-MIDAS), XGBoost Nowcaster e avaliador de pseudo real-time.
"""

from .google_trends_collector import GoogleTrendsCollector
from .energy_data_loader import EnergyDataLoader
from .feature_aligner import FeatureAligner
from .midas_bridge_model import MIDASBridgeModel, XGBoostNowcaster
from .nowcast_evaluator import NowcastEvaluator
from .nowcast_visualization import plot_nowcast_vs_forecast

__all__ = [
    "GoogleTrendsCollector",
    "EnergyDataLoader",
    "FeatureAligner",
    "MIDASBridgeModel",
    "XGBoostNowcaster",
    "NowcastEvaluator",
    "plot_nowcast_vs_forecast",
]
