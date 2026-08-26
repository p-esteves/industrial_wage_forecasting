"""
Suíte de Testes Unitários para o Módulo de Nowcasting de Alta Frequência.

Execução:
    pytest tests/test_nowcasting.py -v
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.nowcasting import (
    GoogleTrendsCollector,
    EnergyDataLoader,
    FeatureAligner,
    MIDASBridgeModel,
    XGBoostNowcaster,
    NowcastEvaluator,
    plot_nowcast_vs_forecast,
)


@pytest.fixture
def sample_monthly_data():
    """Fixture contendo série mensal da massa salarial para testes."""
    dates = pd.date_range(start="2020-01-01", end="2024-12-01", freq="MS")
    np.random.seed(42)
    values = 100 + 0.5 * np.arange(len(dates)) + np.random.normal(0, 2, len(dates))
    return pd.DataFrame({"massa_salarial_real": values}, index=dates)


@pytest.fixture
def sample_daily_energy():
    """Fixture contendo carga de energia diária."""
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "carga_energia_se": 50 + np.random.normal(0, 5, len(dates)),
            "carga_energia_ind": 100 + np.random.normal(0, 10, len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def sample_weekly_trends():
    """Fixture contendo tendências de busca semanais do Google Trends."""
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="W-SUN")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "vagas de emprego": np.random.uniform(20, 80, len(dates)),
            "seguro desemprego": np.random.uniform(10, 60, len(dates)),
        },
        index=dates,
    )


def test_google_trends_collector(sample_weekly_trends):
    """Testa o coletor de tendências de busca do Google Trends com fallback."""
    collector = GoogleTrendsCollector(keywords=["vagas de emprego", "seguro desemprego"])
    df_trends = collector.fetch_weekly_trends(start_date="2023-01-01", end_date="2023-12-31")

    assert isinstance(df_trends, pd.DataFrame)
    assert not df_trends.empty
    assert "vagas de emprego" in df_trends.columns
    assert "seguro desemprego" in df_trends.columns


def test_energy_data_loader():
    """Testa o carregador de consumo de energia elétrica industrial."""
    loader = EnergyDataLoader(subsystems=["SE", "S"])
    df_energy = loader.fetch_energy_data(start_date="2023-01-01", end_date="2023-12-31")

    assert isinstance(df_energy, pd.DataFrame)
    assert not df_energy.empty
    assert "carga_energia_se" in df_energy.columns
    assert "carga_energia_ind" in df_energy.columns


def test_feature_aligner(sample_daily_energy, sample_weekly_trends, sample_monthly_data):
    """Testa o alinhamento de frequências mistas e tratamento de borda irregular."""
    aligner = FeatureAligner()
    df_hf = aligner.align_high_frequency_to_monthly(
        df_daily=sample_daily_energy, df_weekly=sample_weekly_trends, cutoff_day=20
    )

    assert isinstance(df_hf, pd.DataFrame)
    assert not df_hf.empty

    X, y = aligner.create_nowcast_dataset(
        df_monthly_target=sample_monthly_data,
        df_hf_monthly=df_hf,
        target_col="massa_salarial_real",
        lags_target=(1, 2),
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert "massa_salarial_real_lag1" in X.columns


def test_midas_bridge_model(sample_monthly_data, sample_daily_energy, sample_weekly_trends):
    """Testa os modelos Bridge Equation e XGBoost Nowcaster."""
    aligner = FeatureAligner()
    df_hf = aligner.align_high_frequency_to_monthly(
        df_daily=sample_daily_energy, df_weekly=sample_weekly_trends
    )
    X, y = aligner.create_nowcast_dataset(
        df_monthly_target=sample_monthly_data, df_hf_monthly=df_hf, target_col="massa_salarial_real"
    )

    # Bridge Model
    bridge = MIDASBridgeModel(alpha=0.5)
    bridge.fit(X, y)
    preds, low, high = bridge.predict_with_interval(X)

    assert len(preds) == len(X)
    assert len(low) == len(X)
    assert len(high) == len(X)
    assert (high >= preds).all()
    assert (low <= preds).all()

    # XGBoost Nowcaster
    xgb_nowcaster = XGBoostNowcaster(max_depth=3, n_estimators=50)
    xgb_nowcaster.fit(X, y)
    xgb_preds = xgb_nowcaster.predict(X)

    assert len(xgb_preds) == len(X)
    importances = xgb_nowcaster.get_feature_importance(list(X.columns))
    assert len(importances) == len(X.columns)


def test_nowcast_evaluator(sample_monthly_data, sample_daily_energy, sample_weekly_trends):
    """Testa o avaliador de pseudo real-time backtesting."""
    aligner = FeatureAligner()
    df_hf = aligner.align_high_frequency_to_monthly(
        df_daily=sample_daily_energy, df_weekly=sample_weekly_trends
    )
    X_nowcast, y = aligner.create_nowcast_dataset(
        df_monthly_target=sample_monthly_data, df_hf_monthly=df_hf, target_col="massa_salarial_real"
    )

    lag_cols = [c for c in X_nowcast.columns if "lag" in c]
    X_forecast = X_nowcast[lag_cols]

    evaluator = NowcastEvaluator(test_size=6)
    df_metrics, df_predictions = evaluator.evaluate_pseudo_realtime(
        X_nowcast=X_nowcast, X_forecast=X_forecast, y=y
    )

    assert isinstance(df_metrics, pd.DataFrame)
    assert isinstance(df_predictions, pd.DataFrame)
    assert "RMSE" in df_metrics.columns
    assert "Ganho RMSE vs Forecast (%)" in df_metrics.columns
    assert len(df_predictions) == 6


def test_nowcast_visualization(sample_monthly_data, tmp_path):
    """Testa a geração do gráfico Plotly para Nowcasting vs Forecasting."""
    y_hist = sample_monthly_data["massa_salarial_real"]
    nowcast_val = 125.0
    nowcast_low = 120.0
    nowcast_high = 130.0
    fc_dates = pd.date_range(start="2025-01-01", periods=6, freq="MS")
    s_forecast = pd.Series([126, 127, 128, 129, 130, 131], index=fc_dates)

    output_html = tmp_path / "test_nowcast_plot.html"
    fig = plot_nowcast_vs_forecast(
        y_historical=y_hist,
        nowcast_value=nowcast_val,
        nowcast_lower=nowcast_low,
        nowcast_upper=nowcast_high,
        forecast_series=s_forecast,
        filename=str(output_html),
    )

    assert output_html.exists()
    assert os.path.getsize(output_html) > 0
