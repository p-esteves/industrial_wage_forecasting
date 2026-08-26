"""
GUIA PRÁTICO: Como usar o projeto Industrial Wage Mass Forecasting
====================================================================

Este arquivo mostra exemplos práticos de como usar cada módulo.
"""

# ============================================================================
# EXEMPLO 1: Usar apenas a ingestão de dados com fallback automático
# ============================================================================

from data_pipeline import ingest_data, validate_data_quality

# Carregar dados (tenta APIs, fallback para sintéticos)
df = ingest_data(use_real_data=True, timeout=30)

# Validar qualidade
is_valid, report = validate_data_quality(df)
print(f"Dados válidos: {is_valid}")
print(f"Relatório: {report}")

# ============================================================================
# EXEMPLO 2: Engenharia de features passo a passo
# ============================================================================

from feature_engineering import (
    engineer_lags, engineer_rolling_means, 
    engineer_shock_features, engineer_seasonal_features
)

# Criar lags
df_lags = engineer_lags(df, target_col='massa_salarial_real', 
                        lag_periods=[1, 3, 6, 12])

# Adicionar rolling means
df_rolling = engineer_rolling_means(
    df_lags, 
    cols_to_roll=['pim_pf', 'desocupacao', 'selic'],
    windows=[3, 6]
)

# Adicionar choques macroeconômicos
df_shock = engineer_shock_features(df_rolling, shock_col='selic', lag_period=12)

# Adicionar sazonalidade
df_seasonal = engineer_seasonal_features(df_shock, seasonal_type='cyclical')

# ============================================================================
# EXEMPLO 3: Treinar apenas XGBoost (sem Prophet)
# ============================================================================

from feature_engineering import split_features_target
from model_training import train_xgboost_model, get_feature_importance_xgboost

# Preparar dados
X, y = split_features_target(df_seasonal, target_col='massa_salarial_real')

# Treinar XGBoost com hiperparâmetros customizados
model_xgb = train_xgboost_model(
    X_train=X,
    y_train=y,
    max_depth=8,              # Árvores mais profundas
    learning_rate=0.03,       # Aprendizado mais lento (mais conservador)
    n_estimators=300,         # Mais estimadores
    subsample=0.7,
    colsample_bytree=0.7
)

# Obter importância de features
importance_dict, top_10_features = get_feature_importance_xgboost(
    model=model_xgb,
    top_n=10
)

print("Top 10 Features:")
for i, feat in enumerate(top_10_features, 1):
    print(f"  {i}. {feat}")

# ============================================================================
# EXEMPLO 4: Fazer previsão para próximos 12 meses
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Pegar última data do histórico
last_date = df.index[-1]

# Criar datas futuras (próximos 12 meses)
forecast_dates = pd.date_range(
    start=last_date + timedelta(days=30),
    periods=12,
    freq='MS'
)

# Fazer previsões (usando último ponto como referência)
last_features = X.iloc[-1:].copy()
forecast_values = []

for i in range(12):
    pred = model_xgb.predict(last_features)[0]
    forecast_values.append(pred)

# Criar series de previsão
y_forecast = pd.Series(forecast_values, index=forecast_dates)

print("\nPrevisões para próximos 12 meses:")
for date, value in y_forecast.items():
    print(f"  {date.date()}: R$ {value:.2f}")

# ============================================================================
# EXEMPLO 5: Gerar cenários pessimista/base/otimista
# ============================================================================

from feature_engineering import create_scenario_data

# Cenário base (sem choque)
df_base = create_scenario_data(df, scenario='base', shock_magnitude=0.0)

# Cenário otimista (+1 desvio padrão)
df_otimista = create_scenario_data(df, scenario='otimista', shock_magnitude=1.0)

# Cenário pessimista (-1 desvio padrão)
df_pessimista = create_scenario_data(df, scenario='pessimista', shock_magnitude=-1.0)

# Gerar previsões para cada cenário
y_base = model_xgb.predict(X.iloc[-1:].values)
y_otimista = y_base * 1.02  # +2% (otimismo)
y_pessimista = y_base * 0.98  # -2% (pessimismo)

print(f"\nCenários (mês que vem):")
print(f"  Pessimista: R$ {y_pessimista[0]:.2f}")
print(f"  Base:       R$ {y_base[0]:.2f}")
print(f"  Otimista:   R$ {y_otimista[0]:.2f}")

# ============================================================================
# EXEMPLO 6: Usar SHAP para explicar previsões
# ============================================================================

from model_training import get_feature_importance_shap

# Calcular SHAP values
shap_values, top_features = get_feature_importance_shap(
    model=model_xgb,
    X=X,
    top_n=15
)

# Entender o impacto de cada feature
if shap_values is not None:
    print("\nTop 15 Features por SHAP:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")

# ============================================================================
# EXEMPLO 7: Visualizar previsões
# ============================================================================

from visualization import (
    plot_forecast_scenarios,
    plot_economic_variables,
    plot_residuals_analysis
)

# Gráfico de cenários
fig_scenarios = plot_forecast_scenarios(
    y_historical=y,  # Série histórica
    y_forecast_base=y_forecast,
    y_forecast_optimistic=pd.Series([v * 1.02 for v in y_forecast.values], index=y_forecast.index),
    y_forecast_pessimistic=pd.Series([v * 0.98 for v in y_forecast.values], index=y_forecast.index),
    filename="meu_forecast.html"
)

# Gráfico de variáveis econômicas
fig_econ = plot_economic_variables(
    df=df,
    variables=['pim_pf', 'desocupacao', 'selic', 'cambio'],
    filename="variaveis_econ.html"
)

# Gráfico de resíduos
y_pred = model_xgb.predict(X)
fig_residuals = plot_residuals_analysis(
    y_true=y,
    y_pred=y_pred,
    model_name="XGBoost Customizado",
    filename="residuos.html"
)

print("\nGráficos salvos:")
print("  - meu_forecast.html")
print("  - variaveis_econ.html")
print("  - residuos.html")

# ============================================================================
# EXEMPLO 8: Validação temporal (como o código faz)
# ============================================================================

from model_training import temporal_cross_validation
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Setup validação temporal
tscv, X_data, metadata = temporal_cross_validation(
    X=X,
    y=y,
    n_splits=5,
    test_size=12
)

# Avaliar em cada split
rmse_scores = []
mape_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Treinar em cada fold
    model_fold = train_xgboost_model(X_train, y_train)
    
    # Avaliar
    y_pred = model_fold.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    rmse_scores.append(rmse)
    mape_scores.append(mape)
    
    print(f"Fold {fold + 1}: RMSE={rmse:.4f}, MAPE={mape:.4f}")

print(f"\nMédia RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
print(f"Média MAPE: {np.mean(mape_scores):.4f} (±{np.std(mape_scores):.4f})")

# ============================================================================
# EXEMPLO 9: Comparação Prophet vs XGBoost
# ============================================================================

from model_training import (
    train_prophet_model, 
    predict_prophet,
    evaluate_models
)

# Treinar Prophet
df_prophet = df[['massa_salarial_real']].copy()
prophet_model = train_prophet_model(df=df_prophet)

# Previsões Prophet
forecast_prophet, _ = predict_prophet(prophet_model, periods=12)
y_pred_prophet = forecast_prophet.tail(12)['yhat'].values

# Previsões XGBoost
y_pred_xgb = model_xgb.predict(X.tail(12))  # Simplificado

# Comparação
results = evaluate_models(
    y_true=y.tail(12),
    y_pred_prophet=y_pred_prophet,
    y_pred_xgboost=y_pred_xgb,
    model_names=['Prophet', 'XGBoost']
)

# Mostrar resultados
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# ============================================================================
# EXEMPLO 10: Usar configurações customizadas
# ============================================================================

from config import (
    DataConfig, FeatureConfig, ModelConfig,
    ScenarioConfig, VIZ_CONFIG
)

# Customizar configuração de dados
custom_data_config = DataConfig(
    START_DATE="2018-01-01",  # Começar mais recente
    END_DATE="2025-12-31",
    FORECAST_HORIZON=24  # 24 meses ao invés de 12
)

# Customizar features
custom_feature_config = FeatureConfig(
    LAG_PERIODS=[1, 2, 3, 6, 12],  # Mais lags
    ROLLING_WINDOWS=[2, 3, 6, 12],  # Mais windows
    SEASONAL_COMPONENTS=True,
    SHOCK_FEATURES=True
)

# Customizar modelagem
custom_model_config = ModelConfig(
    XGBOOST_MAX_DEPTH=10,  # Mais profundas
    XGBOOST_LEARNING_RATE=0.01,  # Mais conservador
    XGBOOST_N_ESTIMATORS=500  # Mais estimadores
)

print("Configurações customizadas criadas:")
print(f"  Período: {custom_data_config.START_DATE} a {custom_data_config.END_DATE}")
print(f"  Horizonte: {custom_data_config.FORECAST_HORIZON} meses")
print(f"  Lags: {custom_feature_config.LAG_PERIODS}")

# ============================================================================
# RESUMO DOS EXEMPLOS
# ============================================================================

"""
Conceitos demonstrados:

1. INGESTÃO: Carregar dados com fallback automático
2. FEATURES: Lags, rolling means, choques, sazonalidade
3. TREINAMENTO: XGBoost com hiperparâmetros customizados
4. FORECASTING: Gerar previsões para 12 meses
5. CENÁRIOS: Pessimista, Base, Otimista
6. INTERPRETABILIDADE: SHAP values explicam features
7. VISUALIZAÇÃO: Gráficos interativos Plotly
8. VALIDAÇÃO: TimeSeriesSplit sem data leakage
9. COMPARAÇÃO: Prophet vs XGBoost
10. CUSTOMIZAÇÃO: Ajustar configurações globais

Todos os conceitos estão implementados no main.py de forma integrada.
Este arquivo mostra como usar cada componente independentemente.
"""
