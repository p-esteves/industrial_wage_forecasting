"""
Treinamento de modelos: Prophet vs XGBoost com validação temporal.

Author: Economista Quantitativo
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def train_prophet_model(
    df: pd.DataFrame,
    target_col: str = 'massa_salarial_real',
    seasonality_mode: str = 'additive',
    interval_width: float = 0.95,
    yearly_seasonality: bool = True
) -> Any:
    """
    Treina modelo Prophet (Facebook).
    
    Parâmetros:
        df: DataFrame com coluna 'ds' (data) e 'y' (target)
        target_col: Nome da variável target
        seasonality_mode: 'additive' ou 'multiplicative'
        interval_width: Largura do intervalo de confiança (0-1)
        yearly_seasonality: Incluir componente anual
        
    Retorno:
        Modelo Prophet treinado
        
    Características:
        - Robusto a sazonalidade
        - Lida bem com séries curtas
        - Fácil interpretação de componentes
    """
    logger.info("Treinando modelo Prophet...")
    
    try:
        from prophet import Prophet
        
        # Preparar dados no formato esperado pelo Prophet
        df_prophet = df.reset_index()
        df_prophet.columns = ['ds', 'y']
        
        # Inicializar modelo
        model = Prophet(
            seasonality_mode=seasonality_mode,
            interval_width=interval_width,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Treinar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df_prophet)
        
        logger.info(">> Modelo Prophet treinado com sucesso")
        return model
        
    except ImportError:
        logger.error("Prophet não instalado. Install com: pip install prophet")
        raise
    except Exception as e:
        logger.error(f"Erro ao treinar Prophet: {str(e)}")
        raise


def predict_prophet(
    model: Any,
    periods: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza previsões com Prophet.
    
    Parâmetros:
        model: Modelo Prophet treinado
        periods: Horizonte de previsão (meses)
        
    Retorno:
        (forecast_df, components_df)
    """
    logger.info(f"Gerando previsões Prophet para {periods} períodos")
    
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
    # Extrair componentes
    components = model.plot_components(forecast)
    
    logger.debug(f"Forecast shape: {forecast.shape}")
    
    return forecast, components


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8
) -> Any:
    """
    Treina modelo XGBoost para forecasting.
    
    Parâmetros:
        X_train: Features de treinamento
        y_train: Target de treinamento
        max_depth: Profundidade máxima das árvores
        learning_rate: Taxa de aprendizado (shrinkage)
        n_estimators: Número de estimadores
        subsample: Fração de amostras por árvore
        colsample_bytree: Fração de features por árvore
        
    Retorno:
        Modelo XGBoost treinado
        
    Características:
        - Captura não-linearidades
        - Importância de features nativa
        - Rápido treinamento
    """
    logger.info("Treinando modelo XGBoost...")
    
    try:
        from xgboost import XGBRegressor
        
        model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        logger.info(">> Modelo XGBoost treinado com sucesso")
        return model
        
    except ImportError:
        logger.error("XGBoost não instalado. Install com: pip install xgboost")
        raise
    except Exception as e:
        logger.error(f"Erro ao treinar XGBoost: {str(e)}")
        raise


def temporal_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 12
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Validação cruzada temporal (TimeSeriesSplit) para evitar data leakage.
    
    Fluxo:
        Split 1: Train[0:48]    | Test[48:60]
        Split 2: Train[0:60]    | Test[60:72]
        Split 3: Train[0:72]    | Test[72:84]
        ...
    
    Parâmetros:
        X: Features (DatetimeIndex)
        y: Target (DatetimeIndex aligned)
        n_splits: Número de splits
        test_size: Tamanho do conjunto de teste
        
    Retorno:
        (train_indices, test_indices, metadata)
    """
    logger.info(f"Configurando validação temporal (n_splits={n_splits}, test_size={test_size})")
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    splits_info = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        splits_info.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_dates': f"{X.index[train_idx[0]]} to {X.index[train_idx[-1]]}",
            'test_dates': f"{X.index[test_idx[0]]} to {X.index[test_idx[-1]]}"
        })
        logger.debug(f"Fold {fold + 1}: Train[{len(train_idx)}] | Test[{len(test_idx)}]")
    
    metadata = {
        'n_splits': n_splits,
        'splits_info': splits_info
    }
    
    return tscv, X, metadata


def evaluate_models(
    y_true: pd.Series,
    y_pred_prophet: np.ndarray,
    y_pred_xgboost: np.ndarray,
    model_names: list[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Avalia performance de modelos usando múltiplas métricas.
    
    Métricas:
        - RMSE (Root Mean Squared Error): penaliza erros grandes
        - MAE (Mean Absolute Error): robusto a outliers
        - MAPE (Mean Absolute Percentage Error): erro relativo
        - MASE (Mean Absolute Scaled Error): normalizado pela série
        
    Parâmetros:
        y_true: Valores reais
        y_pred_prophet: Previsões Prophet
        y_pred_xgboost: Previsões XGBoost
        model_names: Nomes dos modelos
        
    Retorno:
        Dict com métricas de cada modelo
    """
    logger.info("Avaliando performance dos modelos...")
    
    model_names = model_names or ['Prophet', 'XGBoost']
    predictions = [y_pred_prophet, y_pred_xgboost]
    
    results = {}
    
    for name, y_pred in zip(model_names, predictions):
        # Alinhar comprimentos (em caso de diferença)
        min_len = min(len(y_true), len(y_pred))
        y_true_aligned = y_true.iloc[-min_len:]
        y_pred_aligned = y_pred[-min_len:] if isinstance(y_pred, np.ndarray) else y_pred.iloc[-min_len:]
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
        mae = np.mean(np.abs(y_true_aligned - y_pred_aligned))
        mape = mean_absolute_percentage_error(y_true_aligned, y_pred_aligned)
        
        # MASE (Mean Absolute Scaled Error)
        mae_naive = np.mean(np.abs(np.diff(y_true_aligned)))
        mase = mae / mae_naive if mae_naive > 0 else 0
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MASE': mase
        }
        
        logger.info(f"\n{name}:")
        logger.info(f"  RMSE:  {rmse:.4f}")
        logger.info(f"  MAE:   {mae:.4f}")
        logger.info(f"  MAPE:  {mape:.4f}")
        logger.info(f"  MASE:  {mase:.4f}")
    
    return results


def get_feature_importance_shap(
    model: Any,
    X: pd.DataFrame,
    top_n: int = 10
) -> Tuple[np.ndarray, list[str]]:
    """
    Extrai importância de features usando SHAP (interpretabilidade XAI).
    
    Parâmetros:
        model: Modelo XGBoost treinado
        X: Features
        top_n: Top N features a retornar
        
    Retorno:
        (shap_values, feature_names)
        
    Nota:
        SHAP (SHapley Additive exPlanations) fornece explicações
        locais e globais das previsões do modelo.
    """
    logger.info("Calculando SHAP values para interpretabilidade...")
    
    try:
        import shap
        
        # Criar explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Mean absolute SHAP values para importância global
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Top features
        top_indices = np.argsort(shap_importance)[-top_n:][::-1]
        top_features = [X.columns[i] for i in top_indices]
        top_values = shap_importance[top_indices]
        
        logger.info(f"Top {top_n} features por SHAP:")
        for feat, val in zip(top_features, top_values):
            logger.info(f"  {feat}: {val:.4f}")
        
        return shap_values, top_features
        
    except ImportError:
        logger.warning("SHAP não instalado. Install com: pip install shap")
        return None, None
    except Exception as e:
        logger.error(f"Erro ao calcular SHAP: {str(e)}")
        return None, None


def get_feature_importance_xgboost(
    model: Any,
    top_n: int = 10
) -> Tuple[Dict[str, float], list[str]]:
    """
    Extrai importância de features nativa do XGBoost.
    
    Parâmetros:
        model: Modelo XGBoost treinado
        top_n: Top N features
        
    Retorno:
        (feature_importance_dict, feature_names)
    """
    logger.info("Extraindo importância nativa do XGBoost...")
    
    try:
        importance = model.get_booster().get_score(importance_type='weight')
        
        # Sort e top-n
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:top_n]
        
        logger.info(f"Top {top_n} features (weight):")
        for feat, val in top_features:
            logger.info(f"  {feat}: {val}")
        
        return importance, [f[0] for f in top_features]
        
    except Exception as e:
        logger.error(f"Erro ao extrair importância: {str(e)}")
        return {}, []

def predict_recursive_xgboost(
    model: Any,
    df_history: pd.DataFrame,
    target_col: str,
    horizon: int,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Realiza previsões recursivas (step-by-step) para preservar estrutura temporal.
    
    Parâmetros:
        model: Modelo XGBoost treinado
        df_history: DataFrame com histórico completo (features + target)
        target_col: Nome da coluna target
        horizon: Horizonte de previsão (meses)
        feature_names: Lista ordenada de features esperada pelo modelo
        
    Retorno:
        DataFrame com as previsões futuras (features + y_pred)
    """
    logger.info(f"Iniciando previsão recursiva (Horizonte={horizon} meses)...")
    
    # Copiar histórico para não mutar original
    history = df_history.copy()
    
    # Identificar última data
    last_date = history.index[-1]
    freq = pd.infer_freq(history.index) or 'MS'
    
    # Data range futuro
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
    
    predictions = []
    
    try:
        for date in future_dates:
            # 1. Criar nova linha para o futuro
            new_row = pd.DataFrame(index=[date])
            
            # 2. Atualizar Features Sazonais (Manualmente para garantir consistência)
            new_row['month'] = date.month
            new_row['quarter'] = date.quarter
            
            # Ciclicas (se existirem nas features)
            if any('month_sin' in f for f in feature_names):
                new_row['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                new_row['month_cos'] = np.cos(2 * np.pi * date.month / 12)
            
            if any('quarter_sin' in f for f in feature_names):
                new_row['quarter_sin'] = np.sin(2 * np.pi * date.quarter / 4)
                new_row['quarter_cos'] = np.cos(2 * np.pi * date.quarter / 4)
                
            # Dummies (se existirem)
            for feat in feature_names:
                if feat.startswith('month_') and feat not in new_row.columns:
                    # month_1, month_2 etc
                    try:
                        m_num = int(feat.split('_')[1])
                        new_row[feat] = 1 if date.month == m_num else 0
                    except:
                        pass
            
            # 3. Atualizar Lags
            # Precisamos olhar para o histórico (que cresce a cada passo)
            # Para cada feature de lag (ex: target_lag1)
            for feat in feature_names:
                if f"{target_col}_lag" in feat:
                    # Extrair número do lag (ex: massa_salarial_real_lag1 -> 1)
                    try:
                        lag_num = int(feat.split('_lag')[-1])
                        
                        # Buscar valor no histórico (índice -lag_num)
                        # Como history está ordenado, iloc[-lag_num] pega o valor correto
                        # Se lag for maior que history disponível? (improvável c/ history grande)
                        val = history[target_col].iloc[-lag_num]
                        new_row[feat] = val
                    except Exception as e:
                        logger.warning(f"Erro ao calcular lag {feat}: {e}")
                        new_row[feat] = 0
            
            # 4. Atualizar Rolling Means
            # Similar aos lags, mas calculando média
            for feat in feature_names:
                if f"{target_col}_rolling_mean_" in feat:
                    try:
                        # Ex: ..._6m -> 6
                        window = int(feat.split('_rolling_mean_')[1].replace('m',''))
                        val = history[target_col].iloc[-window:].mean()
                        new_row[feat] = val
                    except:
                        pass
                    
            # 5. Outras features (Choques/Macroeconômicas)
            # Assumir constante ou tendência linear do último ano?
            # Simplificação: Repetir último valor conhecido para exógenas
            for feat in feature_names:
                if feat not in new_row.columns:
                    new_row[feat] = history[feat].iloc[-1]
            
            # Garantir ordem das colunas
            X_step = new_row[feature_names]
            
            # 6. Prever
            pred_value = model.predict(X_step)[0]
            
            # 7. Adicionar ao histórico (para próximos lags)
            # Precisamos preencher a coluna target na new_row para cálculos futuros
            new_row[target_col] = pred_value
            
            # Concatenar mantendo todas as colunas
            # Alinhar colunas
            new_row_full = new_row.reindex(columns=history.columns)
            # Preencher gaps se houver
            for col in history.columns:
                if col not in new_row_full.columns:
                    new_row_full[col] = history[col].iloc[-1]
            
            history = pd.concat([history, new_row_full])
            predictions.append(pred_value)
            
        # Retornar DataFrame com previsões
        df_forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions
        }).set_index('ds')
        
        logger.info(">> Previsão recursiva concluída com sucesso.")
        return df_forecast

    except Exception as e:
        logger.error(f"Falha durante a previsão recursiva temporal: {str(e)}")
        raise RuntimeError(f"Erro no pipeline de previsão recursiva do XGBoost: {e}")
