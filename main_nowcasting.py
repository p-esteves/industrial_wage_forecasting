"""
Script Principal de Orquestração do Módulo de Nowcasting de Alta Frequência.

Execução:
    python main_nowcasting.py

Fluxo:
    1. Ingestão do histórico mensal oficial
    2. Coleta de proxies de alta frequência (Google Trends + Consumo de Energia ONS)
    3. Alinhamento de frequências mistas e tratamento da borda irregular (Ragged Edge)
    4. Execução de Pseudo Real-Time Backtest (Nowcast vs. Forecast Puro)
    5. Exportação de métricas comparativas em CSV (results/)
    6. Cálculo do Nowcast do Mês Corrente (com intervalo de confiança)
    7. Geração de gráfico interativo Plotly (outputs/06_nowcasting_vs_forecasting.html)
"""

import logging
import logging.config
import sys
from pathlib import Path

# Garantir que a raiz do projeto esteja no sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from config import DATA_CONFIG, LOGGING_CONFIG, NOWCASTING_CONFIG
from data_pipeline import ingest_data
from src.nowcasting import (
    GoogleTrendsCollector,
    EnergyDataLoader,
    FeatureAligner,
    MIDASBridgeModel,
    XGBoostNowcaster,
    NowcastEvaluator,
    plot_nowcast_vs_forecast,
)

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("nowcasting_main")

Path("results").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def run_nowcasting_pipeline() -> None:
    """
    Função principal de orquestração do pipeline de Nowcasting.
    """
    logger.info("=" * 70)
    logger.info("        MÓDULO DE NOWCASTING DE ALTA FREQUÊNCIA (MIXED FREQUENCY)       ")
    logger.info("=" * 70)

    try:
        # ============================================================
        # ETAPA 1: INGESTÃO DO HISTÓRICO MENSAL
        # ============================================================
        logger.info("\n[1/6] Carregando Séries Históricas Mensais...")
        df_monthly = ingest_data(use_real_data=True)
        target_col = "massa_salarial_real"

        # ============================================================
        # ETAPA 2: COLETA DE DADOS ALTERNATIVOS DE ALTA FREQUÊNCIA
        # ============================================================
        logger.info("\n[2/6] Coletando Proxies de Alta Frequência (Google Trends + ONS)...")
        start_date = df_monthly.index[0].strftime("%Y-%m-%d")
        end_date = df_monthly.index[-1].strftime("%Y-%m-%d")

        trends_collector = GoogleTrendsCollector(
            keywords=NOWCASTING_CONFIG.TRENDS_KEYWORDS,
            geo=NOWCASTING_CONFIG.TRENDS_GEO,
        )
        df_weekly_trends = trends_collector.fetch_weekly_trends(
            start_date=start_date, end_date=end_date
        )

        energy_loader = EnergyDataLoader(subsystems=NOWCASTING_CONFIG.ONS_SUBSYSTEMS)
        df_daily_energy = energy_loader.fetch_energy_data(
            start_date=start_date, end_date=end_date
        )

        # ============================================================
        # ETAPA 3: ALINHAMENTO DE FREQUÊNCIAS MISTAS & RAGGED EDGE
        # ============================================================
        logger.info("\n[3/6] Alinhando Frequências Mistas (Dia/Semana -> Mês)...")
        aligner = FeatureAligner()
        
        # Simular Nowcast no dia 20 do mês corrente
        df_hf_monthly = aligner.align_high_frequency_to_monthly(
            df_daily=df_daily_energy,
            df_weekly=df_weekly_trends,
            cutoff_day=20,
        )

        # Criar matriz de atributos de Nowcast (alta frequência + lags da meta)
        X_nowcast, y_target = aligner.create_nowcast_dataset(
            df_monthly_target=df_monthly,
            df_hf_monthly=df_hf_monthly,
            target_col=target_col,
            lags_target=(1, 2, 12),
        )

        # Criar matriz de atributos de Forecast Puro (somente lags da meta, sem alta frequência)
        lag_cols = [c for c in X_nowcast.columns if target_col in c and "lag" in c]
        X_forecast = X_nowcast[lag_cols].copy()

        # ============================================================
        # ETAPA 4: PSEUDO REAL-TIME BACKTEST & AVALIAÇÃO DE GANHO
        # ============================================================
        logger.info("\n[4/6] Executando Pseudo Real-Time Backtest (12 Meses Out-of-Sample)...")
        evaluator = NowcastEvaluator(test_size=12)
        df_metrics, df_predictions = evaluator.evaluate_pseudo_realtime(
            X_nowcast=X_nowcast,
            X_forecast=X_forecast,
            y=y_target,
        )

        metrics_path, preds_path = evaluator.export_results(
            df_metrics=df_metrics,
            df_predictions=df_predictions,
            output_dir=NOWCASTING_CONFIG.RESULTS_DIR,
        )

        # ============================================================
        # ETAPA 5: ESTIMATIVA DO NOWCAST DO MÊS CORRENTE (HOJE)
        # ============================================================
        logger.info("\n[5/6] Gerando Estimativa do Mês Corrente (Nowcast t)...")
        nowcaster_full = XGBoostNowcaster(
            max_depth=NOWCASTING_CONFIG.NOWCAST_XGB_MAX_DEPTH,
            learning_rate=NOWCASTING_CONFIG.NOWCAST_XGB_LEARNING_RATE,
            n_estimators=NOWCASTING_CONFIG.NOWCAST_XGB_N_ESTIMATORS,
        )
        nowcaster_full.fit(X_nowcast, y_target)

        # Obter última observação disponível para prever o mês t
        latest_X = X_nowcast.iloc[[-1]]
        nowcast_val, lower_bound, upper_bound = nowcaster_full.predict_with_interval(latest_X)

        nowcast_point = float(nowcast_val[0])
        nowcast_low = float(lower_bound[0])
        nowcast_high = float(upper_bound[0])
        nowcast_date = y_target.index[-1] + pd.DateOffset(months=1)

        logger.info(f"  >> Data do Nowcast: {nowcast_date.strftime('%Y-%m')}")
        logger.info(f"  >> Estimativa Pontual: R$ {nowcast_point:.2f}")
        logger.info(f"  >> Intervalo de Confiança (95%): [R$ {nowcast_low:.2f}, R$ {nowcast_high:.2f}]")

        # ============================================================
        # ETAPA 6: GERAÇÃO DE FORECAST DE MÉDIO PRAZO E VISUALIZAÇÃO
        # ============================================================
        logger.info("\n[6/6] Gerando Projeção de Médio Prazo e Gráfico Interativo...")
        
        # Simular curva de forecast de 12 meses simples a partir do Nowcast
        forecast_dates = pd.date_range(
            start=nowcast_date + pd.DateOffset(months=1),
            periods=NOWCASTING_CONFIG.NOWCAST_CUTOFF_DAYS[0] if False else 12,
            freq="MS",
        )
        
        # Trajetória baseada em tendência média recente
        growth_rate = (nowcast_point - y_target.iloc[-12]) / y_target.iloc[-12]
        monthly_drift = (1 + growth_rate) ** (1 / 12) - 1
        forecast_vals = [nowcast_point * ((1 + monthly_drift) ** i) for i in range(1, 13)]
        s_forecast = pd.Series(forecast_vals, index=forecast_dates)

        # Gerar gráfico interativo Plotly
        fig = plot_nowcast_vs_forecast(
            y_historical=y_target,
            nowcast_value=nowcast_point,
            nowcast_lower=nowcast_low,
            nowcast_upper=nowcast_high,
            forecast_series=s_forecast,
            nowcast_date=nowcast_date,
            filename="outputs/06_nowcasting_vs_forecasting.html",
        )

        # ============================================================
        # RESUMO EXECUTIVO
        # ============================================================
        best_nowcast_model = df_metrics.index[df_metrics["RMSE"].argmin()]
        best_rmse = df_metrics.loc[best_nowcast_model, "RMSE"]
        gain_pct = df_metrics.loc[best_nowcast_model, "Ganho RMSE vs Forecast (%)"]

        summary = f"""
======================================================================
              RESUMO DO MÓDULO DE NOWCASTING DE ALTA FREQUÊNCIA
======================================================================
  Último Dado Oficial:  {y_target.index[-1].strftime('%Y-%m')} (R$ {y_target.iloc[-1]:.2f})
  Data do Nowcast (t):  {nowcast_date.strftime('%Y-%m')}
  
  ESTIMATIVA DO MÊS CORRENTE (NOWCAST):
    Valor Pontual:      R$ {nowcast_point:.2f}
    Intervalo 95%:      [R$ {nowcast_low:.2f}, R$ {nowcast_high:.2f}]

  DESEMPENHO RELATIVO (PSEUDO REAL-TIME BACKTEST):
    Modelo Vencedor:    {best_nowcast_model}
    RMSE Nowcast:       {best_rmse:.4f}
    Ganho de Precisão:  {gain_pct:.2f}% de redução de erro vs. Forecast Puro
    
  ARQUIVOS GERADOS:
    Métricas CSV:       {metrics_path}
    Gráfico Plotly:     outputs/06_nowcasting_vs_forecasting.html
======================================================================
"""
        logger.info(summary)
        logger.info("[OK] Pipeline de Nowcasting concluído com sucesso!")

    except Exception as e:
        logger.critical(f"Erro crítico no pipeline de Nowcasting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_nowcasting_pipeline()
