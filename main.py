"""
Script principal: Orquestração completa do pipeline de forecasting.

Author: Economista Quantitativo
Date: Janeiro 2026

Execução:
    python main.py
    
Padrões de engenharia:
    - Modularidade: funções independentes com responsabilidade única
    - Type hinting: tipagem estática em todas as funções
    - Logging: relatório detalhado de execução
    - Tratamento de erros: blocos try-except robusto
    - PEP 8: código limpo e legível
"""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Importar módulos do projeto
from config import (
    DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, 
    SCENARIO_CONFIG, VIZ_CONFIG, LOGGING_CONFIG
)
from data_pipeline import (
    ingest_data, validate_data_quality, 
    deflate_to_real_values, setup_logger
)
from feature_engineering import (
    create_autoregressive_features, split_features_target,
    create_scenario_data
)
from model_training import (
    train_prophet_model, train_xgboost_model,
    predict_prophet, evaluate_models, 
    get_feature_importance_xgboost, get_feature_importance_shap,
    predict_recursive_xgboost
)
from visualization import (
    plot_forecast_scenarios, plot_model_comparison,
    plot_shap_feature_importance, plot_economic_variables,
    plot_residuals_analysis
)

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Criar diretório de logs se não existir
Path("logs").mkdir(exist_ok=True)


def main() -> None:
    """
    Função principal: executa pipeline completo de forecasting.
    
    Fluxo:
        1. Ingestão de dados (com fallback)
        2. Validação de qualidade
        3. Engenharia de features
        4. Treinamento de modelos (Prophet + XGBoost)
        5. Avaliação e comparação
        6. Interpretabilidade (SHAP)
        7. Previsão de cenários
        8. Visualizações
    """
    
    logger.info("=" * 70)
    logger.info("               FORECASTING DE MASSA SALARIAL REAL                  ")
    logger.info("                  Industrial Wage Mass Forecasting                 ")
    logger.info("=" * 70)
    
    try:
        # ============================================================
        # ETAPA 1: INGESTÃO DE DADOS
        # ============================================================
        logger.info("\n[1/8] Ingestão de Dados")
        logger.info("-" * 70)
        
        df = ingest_data(use_real_data=True, timeout=30)
        
        # Validar qualidade
        is_valid, quality_report = validate_data_quality(df)
        if not is_valid:
            logger.warning("Dados com qualidade reduzida. Continuando...")
        
        logger.info(f">> Dados carregados: {df.shape[0]} observações x {df.shape[1]} variáveis")
        logger.info(f"  Período: {df.index[0].date()} a {df.index[-1].date()}")
        
        # ============================================================
        # ETAPA 2: ENGENHARIA DE FEATURES
        # ============================================================
        logger.info("\n[2/8] Engenharia de Features")
        logger.info("-" * 70)
        
        df_features, feature_names = create_autoregressive_features(
            df=df,
            target_col='massa_salarial_real',
            n_lags=12
        )
        
        logger.info(f">> Features engenheiradas: {len(feature_names)} features criadas")
        
        # Separar X e y
        X, y = split_features_target(df_features, target_col='massa_salarial_real')
        
        # ============================================================
        # ETAPA 2.5: SPLIT TREINO / TESTE (VALIDAÇÃO RIGOROSA)
        # ============================================================
        logger.info("\n[2.5/8] Split Treino/Teste (Out-of-Sample)")
        logger.info("-" * 70)
        
        test_size = MODEL_CONFIG.TEST_SIZE
        
        # Split Pandas
        X_train = X.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_train = y.iloc[:-test_size]
        y_test = y.iloc[-test_size:]
        
        # Split Prophet
        df_prophet_full = df[['massa_salarial_real']].copy() # Para refit
        df_prophet_train = df_prophet_full.iloc[:-test_size]
        df_prophet_test = df_prophet_full.iloc[-test_size:]
        
        logger.info(f"Treino: {len(X_train)} meses | Teste: {len(X_test)} meses")
        
        # ============================================================
        # ETAPA 3: VALIDAÇÃO CRUZADA (PROPHET)
        # ============================================================
        logger.info("\n[3/8] Validação - Modelo Prophet")
        logger.info("-" * 70)
        
        try:
            # Treino no Train Set
            prophet_val = train_prophet_model(
                df=df_prophet_train,
                target_col='massa_salarial_real',
                seasonality_mode='additive'
            )
            
            # Previsão no Test Set
            forecast_val, _ = predict_prophet(
                model=prophet_val,
                periods=test_size
            )
            y_pred_prophet_test = forecast_val.tail(test_size)['yhat'].values
            
        except Exception as e:
            logger.error(f"Erro validação Prophet: {e}")
            y_pred_prophet_test = np.zeros(test_size)

        # ============================================================
        # ETAPA 4: VALIDAÇÃO CRUZADA (XGBOOST)
        # ============================================================
        logger.info("\n[4/8] Validação - Modelo XGBoost")
        logger.info("-" * 70)
        
        try:
            # Treino no Train Set
            xgboost_val = train_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                max_depth=MODEL_CONFIG.XGBOOST_MAX_DEPTH,
                learning_rate=MODEL_CONFIG.XGBOOST_LEARNING_RATE,
                n_estimators=MODEL_CONFIG.XGBOOST_N_ESTIMATORS
            )
            
            # Previsão Recursiva no Test Set (Fair Evaluation)
            df_recursive_val = predict_recursive_xgboost(
                model=xgboost_val,
                df_history=df_features.iloc[:-test_size], # Histórico até o corte
                target_col='massa_salarial_real',
                horizon=test_size,
                feature_names=feature_names
            )
            y_pred_xgb_test = df_recursive_val['yhat'].values
            
        except Exception as e:
            logger.error(f"Erro validação XGBoost: {e}")
            y_pred_xgb_test = np.zeros(test_size)
            
        # ============================================================
        # ETAPA 5: CÁLCULO DE MÉTRICAS REAIS
        # ============================================================
        logger.info("\n[5/8] Resultados da Validação (Métricas Reais)")
        logger.info("-" * 70)
        
        results = evaluate_models(
            y_true=y_test,
            y_pred_prophet=y_pred_prophet_test,
            y_pred_xgboost=y_pred_xgb_test,
            model_names=['Prophet', 'XGBoost']
        )
        
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        logger.info(f"\n>> Vencedor na Validação: {best_model}")
        
        # ============================================================
        # ETAPA 6: RETREINO FINAL (PRODUÇÃO)
        # ============================================================
        logger.info("\n[6/8] Retreino Final (Full History)")
        logger.info("-" * 70)
        logger.info("Retreinando melhor modelo com TODO o histórico para prever futuro...")
        
        if best_model == 'XGBoost':
            xgboost_model = train_xgboost_model(
                X_train=X, # Full
                y_train=y  # Full
            )
            y_pred_xgb_insample = xgboost_model.predict(X)
        else:
            # Mesmo se Prophet vencer, treinamos XGBoost para SHAP e Resíduos
            xgboost_model = train_xgboost_model(
                X_train=X, # Full
                y_train=y  # Full
            )
            y_pred_xgb_insample = xgboost_model.predict(X)
            
            # Treinar Prophet vencedor
            prophet_model = train_prophet_model(
                df=df_prophet_full # Full
            )
            # Precisamos gerar forecast futuro aqui para usar depois
            forecast_prophet, _ = predict_prophet(
                model=prophet_model,
                periods=DATA_CONFIG.FORECAST_HORIZON
            )
            y_pred_prophet = forecast_prophet.tail(DATA_CONFIG.FORECAST_HORIZON)['yhat'].values
        
        # ============================================================
        # ETAPA 6: INTERPRETABILIDADE (SHAP)
        # ============================================================
        logger.info("\n[6/8] Interpretabilidade - SHAP Values")
        logger.info("-" * 70)
        
        try:
            shap_values, top_features = get_feature_importance_shap(
                model=xgboost_model,
                X=X,
                top_n=10
            )
            logger.info(f">> SHAP values calculados. Top 10 features:")
            for feat in top_features:
                logger.info(f"  - {feat}")
        except:
            logger.warning("SHAP indisponível. Usando importância nativa...")
            importance_dict, top_features = get_feature_importance_xgboost(
                model=xgboost_model,
                top_n=10
            )
            shap_values = np.array([importance_dict.get(f, 0) for f in X.columns])
        
        # ============================================================
        # ETAPA 7: PREVISÃO DE CENÁRIOS
        # ============================================================
        logger.info("\n[7/8] Previsão de Cenários")
        logger.info("-" * 70)
        
        last_date = df.index[-1]
        
        # Usar melhor modelo para criar cenários
        scenario_forecasts = {}
        
        if best_model == 'XGBoost':
            # Previsão Recursiva Base
            logger.info("Gerando previsão recursiva com XGBoost...")
            try:
                df_recursive = predict_recursive_xgboost(
                    model=xgboost_model,
                    df_history=df_features,
                    target_col='massa_salarial_real',
                    horizon=DATA_CONFIG.FORECAST_HORIZON,
                    feature_names=feature_names
                )
                base_series = df_recursive['yhat']
            except Exception as e:
                logger.error(f"Falha na previsão recursiva: {e}. Usando fallback linear.")
                base_series = pd.Series(
                    [y_pred_xgb_insample[-1]] * DATA_CONFIG.FORECAST_HORIZON,
                    index=pd.date_range(start=last_date + timedelta(days=30), periods=DATA_CONFIG.FORECAST_HORIZON, freq='MS')
                )
        else:
            # Prophet
            # y_pred_prophet já contém os valores futuros gerados na Etapa 3
            # Recriar índice de datas para garantir alinhamento
            forecast_dates_prophet = pd.date_range(
                start=last_date + timedelta(days=30),
                periods=DATA_CONFIG.FORECAST_HORIZON,
                freq='MS'
            )
            # Garantir que temos apenas o horizonte futuro
            base_values = y_pred_prophet[-DATA_CONFIG.FORECAST_HORIZON:]
            base_series = pd.Series(base_values, index=forecast_dates_prophet)
        
        # Gerar cenários aplicados sobre a série base
        for scenario_name in ['pessimista', 'base', 'otimista']:
            logger.info(f"  Gerando cenário: {scenario_name}")
            
            scenario_shock = SCENARIO_CONFIG.SCENARIOS[scenario_name]
            # Aplicar choque percentual sobre toda a curva
            multiplier = 1 + (scenario_shock * 0.02)
            
            scenario_series = base_series * multiplier
            scenario_forecasts[scenario_name] = scenario_series
        
        logger.info(f">> Cenários gerados: {len(scenario_forecasts)} séries de {len(base_series)} meses")
        
        # ============================================================
        # ETAPA 8: VISUALIZAÇÕES
        # ============================================================
        logger.info("\n[8/8] Geração de Visualizações")
        logger.info("-" * 70)
        
        # Series de previsão já estão prontas no dicionário
        y_base = scenario_forecasts['base']
        y_otimista = scenario_forecasts['otimista']
        y_pessimista = scenario_forecasts['pessimista']
        
        # Gráfico de cenários
        fig_scenarios = plot_forecast_scenarios(
            y_historical=y,
            y_forecast_base=y_base,
            y_forecast_optimistic=y_otimista,
            y_forecast_pessimistic=y_pessimista,
            filename="outputs/01_forecast_scenarios.html"
        )
        
        # Gráfico de comparação
        fig_comparison = plot_model_comparison(
            results=results,
            filename="outputs/02_model_comparison.html"
        )
        
        # Gráfico de importância
        fig_importance = plot_shap_feature_importance(
            feature_importance=shap_values,
            feature_names=X.columns.tolist(),
            filename="outputs/03_feature_importance.html"
        )
        
        # Gráfico de variáveis econômicas
        vars_to_plot = ['pim_pf', 'desocupacao', 'selic', 'cambio', 'ipca']
        available_vars = [v for v in vars_to_plot if v in df.columns]
        fig_economics = plot_economic_variables(
            df=df,
            variables=available_vars,
            filename="outputs/04_economic_variables.html"
        )
        
        # Gráfico de resíduos
        fig_residuals = plot_residuals_analysis(
            y_true=y,
            y_pred=y_pred_xgb_insample,
            model_name="XGBoost",
            filename="outputs/05_residuals_analysis.html"
        )
        
        logger.info(">> Visualizações geradas em outputs/")
        
        # ============================================================
        # RESUMO EXECUTIVO
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("RESUMO EXECUTIVO")
        logger.info("=" * 70)
        
        # Prepare top 5 features string
        top_5_str_lines = []
        for i, feat in enumerate(top_features[:5], 1):
            top_5_str_lines.append(f"    {i}. {feat}")
        top_5_str = "\n".join(top_5_str_lines)

        # Get best RMSE
        best_rmse = results[best_model]['RMSE']

        summary = f"""
======================================================================
                      RESUMO DO FORECASTING
======================================================================
  Período Histórico:  {df.index[0].date()} a {df.index[-1].date()}
  Observações:        {len(df)} meses
  Horizonte Previsão: {DATA_CONFIG.FORECAST_HORIZON} meses

  PERFORMANCE DOS MODELOS:
    Melhor Modelo:    {best_model}
    RMSE:             {best_rmse:.4f}
    MAE:              {results[best_model]['MAE']:.4f}
    MAPE:             {results[best_model]['MAPE']:.4f}%

  CENÁRIOS (Próximos 12 Meses - Valor Final):
    Pessimista (-1s): R$ {scenario_forecasts['pessimista'].iloc[-1]:.2f}
    Base (0s):        R$ {scenario_forecasts['base'].iloc[-1]:.2f}
    Otimista (+1s):   R$ {scenario_forecasts['otimista'].iloc[-1]:.2f}

  TOP 5 FEATURES:
{top_5_str}
======================================================================
"""
        
        
        logger.info(summary)
        
        logger.info("\n[OK] Pipeline concluído com sucesso!")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.critical(f"Erro crítico no pipeline: {str(e)}")
        logger.exception("Stack trace:")
        sys.exit(1)


if __name__ == "__main__":
    # Criar diretório de saída
    Path("outputs").mkdir(exist_ok=True)
    
    # Executar pipeline
    main()
