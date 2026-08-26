"""
Engenharia de features para forecasting de Massa Salarial.
Implementa lags, rolling means, variáveis de choque econômico e sazonalidade.

Author: Economista Quantitativo
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np
from config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


def engineer_lags(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: list[int] = None
) -> pd.DataFrame:
    """
    Cria features de defasagens (lags) do target.
    
    Parâmetros:
        df: DataFrame com série temporal
        target_col: Nome da coluna target (e.g., 'massa_salarial_real')
        lag_periods: Lista de períodos de lag [1, 3, 6, 12]
        
    Retorno:
        DataFrame com colunas de lags adicionadas
        
    Exemplo:
        lag_1, lag_3, lag_6, lag_12 para massa_salarial_real
    """
    logger.info(f"Criando features de lags para {target_col}")
    
    lag_periods = lag_periods or FEATURE_CONFIG.LAG_PERIODS
    df_lags = df.copy()
    
    for lag in lag_periods:
        col_name = f"{target_col}_lag{lag}"
        df_lags[col_name] = df[target_col].shift(lag)
        logger.debug(f"Criado: {col_name}")
    
    return df_lags


def engineer_rolling_means(
    df: pd.DataFrame,
    cols_to_roll: list[str],
    windows: list[int] = None
) -> pd.DataFrame:
    """
    Cria médias móveis (rolling means) para features selecionadas.
    
    Parâmetros:
        df: DataFrame
        cols_to_roll: Colunas para calcular rolling means
        windows: Tamanho das janelas [3, 6] (meses)
        
    Retorno:
        DataFrame com features de rolling means
        
    Intuição Econômica:
        - Suaviza volatilidade mensal
        - Captura tendência de curto prazo
    """
    logger.info(f"Criando rolling means com windows={windows}")
    
    windows = windows or FEATURE_CONFIG.ROLLING_WINDOWS
    df_rolling = df.copy()
    
    for col in cols_to_roll:
        for window in windows:
            col_name = f"{col}_rolling_mean_{window}m"
            df_rolling[col_name] = df[col].rolling(window=window, min_periods=1).mean()
            logger.debug(f"Criado: {col_name}")
    
    return df_rolling


def engineer_shock_features(
    df: pd.DataFrame,
    shock_col: str = 'selic',
    lag_period: int = 12
) -> pd.DataFrame:
    """
    Cria features de "choque econômico" (variação de taxa diretora).
    
    Parâmetros:
        df: DataFrame
        shock_col: Coluna com a variável de choque ('selic')
        lag_period: Período de defasagem (12 meses padrão)
        
    Retorno:
        DataFrame com feature de choque
        
    Detalhes:
        Captura mudança acumulada na Selic nos últimos 12 meses.
        Impacto defasado: mudança na Selic hoje afeta salários em ~12 meses.
    """
    logger.info(f"Criando features de choque para {shock_col}")
    
    df_shock = df.copy()
    
    # Variação da Selic em 12 meses (pontos percentuais)
    shock_feature = df[shock_col] - df[shock_col].shift(lag_period)
    df_shock[f"{shock_col}_12m_change"] = shock_feature
    
    # Magnitude do choque (valor absoluto)
    df_shock[f"{shock_col}_shock_magnitude"] = shock_feature.abs()
    
    logger.debug(f"Criadas features: {shock_col}_12m_change, {shock_col}_shock_magnitude")
    
    return df_shock


def engineer_seasonal_features(
    df: pd.DataFrame,
    seasonal_type: str = "cyclical"
) -> pd.DataFrame:
    """
    Cria features de sazonalidade (mês, trimestre, componentes harmônicos).
    
    Parâmetros:
        df: DataFrame com índice DatetimeIndex
        seasonal_type: 'cyclical' (sin/cos) ou 'categorical' (dummies)
        
    Retorno:
        DataFrame com features sazonais
        
    Nota:
        Cyclical (sin/cos) é mais apropriado para modelos de ML.
        Capturam periodicidade: picos em jul/ago e dez (fim de ano).
    """
    logger.info(f"Criando features sazonais ({seasonal_type})")
    
    df_seasonal = df.copy()
    
    # Extrair mês e trimestre
    df_seasonal['month'] = df.index.month
    df_seasonal['quarter'] = df.index.quarter
    df_seasonal['day_of_year'] = df.index.dayofyear
    
    if seasonal_type == "cyclical":
        # Representação cíclica (melhor para modelos ML)
        # Seno e cosseno do mês (periodicidade de 12 meses)
        df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['month'] / 12)
        df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['month'] / 12)
        
        # Seno e cosseno do trimestre (periodicidade de 4 trimestres)
        df_seasonal['quarter_sin'] = np.sin(2 * np.pi * df_seasonal['quarter'] / 4)
        df_seasonal['quarter_cos'] = np.cos(2 * np.pi * df_seasonal['quarter'] / 4)
        
        logger.debug("Features cíclicas criadas: month_sin, month_cos, quarter_sin, quarter_cos")
    
    elif seasonal_type == "categorical":
        # Dummies (para árvores)
        month_dummies = pd.get_dummies(df_seasonal['month'], prefix='month')
        df_seasonal = pd.concat([df_seasonal, month_dummies], axis=1)
        logger.debug("Features categóricas (dummies) criadas")
    
    return df_seasonal


def create_autoregressive_features(
    df: pd.DataFrame,
    target_col: str,
    n_lags: int = 12
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Cria suite completa de features autoregressivas.
    
    Combina:
        1. Lags do target (t-1, t-3, t-6, t-12)
        2. Médias móveis (3m, 6m)
        3. Choques macroeconômicos
        4. Sazonalidade
        
    Parâmetros:
        df: DataFrame com variáveis econômicas
        target_col: Nome do target ('massa_salarial_real')
        n_lags: Número máximo de lags
        
    Retorno:
        (df_engineered: DataFrame com todas as features,
         feature_names: Lista de nomes das features)
    """
    logger.info("=" * 70)
    logger.info("ENGENHARIA COMPLETA DE FEATURES")
    logger.info("=" * 70)
    
    try:
        df_feat = df.copy()
        
        # 1. Lags
        df_feat = engineer_lags(
            df_feat,
            target_col=target_col,
            lag_periods=[1, 3, 6, 12]
        )
        
        # 2. Rolling means
        rolling_cols = [col for col in df.columns if col != target_col]
        df_feat = engineer_rolling_means(
            df_feat,
            cols_to_roll=rolling_cols,
            windows=[3, 6]
        )
        
        # 3. Choques (se aplicável)
        if 'selic' in df.columns:
            df_feat = engineer_shock_features(df_feat, shock_col='selic')
        if 'cambio' in df.columns:
            df_feat = engineer_shock_features(df_feat, shock_col='cambio', lag_period=12)
        
        # 4. Sazonalidade
        if FEATURE_CONFIG.SEASONAL_COMPONENTS:
            df_feat = engineer_seasonal_features(df_feat, seasonal_type='cyclical')
        
        # Remover linhas com NaN (criadas pelos lags)
        n_rows_before = len(df_feat)
        df_feat = df_feat.dropna()
        n_rows_dropped = n_rows_before - len(df_feat)
        
        logger.info(f"Linhas removidas (NaN de lags): {n_rows_dropped}")
        logger.info(f"Shape final: {df_feat.shape}")
        
        # Lista de features (excluindo target)
        feature_names = [col for col in df_feat.columns if col != target_col]
        
        logger.info(f"Total de features criadas: {len(feature_names)}")
        logger.debug(f"Features: {feature_names}")
        logger.info("=" * 70)
        
        return df_feat, feature_names
        
    except Exception as e:
        logger.error(f"Erro na engenharia de features: {str(e)}")
        raise


def create_scenario_data(
    df: pd.DataFrame,
    scenario: str = "base",
    shock_magnitude: float = 1.0
) -> pd.DataFrame:
    """
    Cria dados cenários para projeção (pessimista, base, otimista).
    
    Parâmetros:
        df: DataFrame com dados históricos
        scenario: 'pessimista' (-1σ), 'base' (0σ), 'otimista' (+1σ)
        shock_magnitude: Magnitude do choque (desvios padrão)
        
    Retorno:
        DataFrame com features ajustadas para o cenário
        
    Intuição:
        - Pessimista: reduz PIM-PF e massa salarial, aumenta desocupação
        - Base: mantém médias históricas
        - Otimista: aumenta PIM-PF e massa salarial, reduz desocupação
    """
    logger.info(f"Criando dados para cenário: {scenario}")
    
    df_scenario = df.copy()
    
    # Obter desvio padrão das variáveis
    std_dict = df.std()
    
    scenario_map = {
        'pessimista': -1.0,
        'base': 0.0,
        'otimista': 1.0
    }
    
    if scenario not in scenario_map:
        logger.warning(f"Cenário '{scenario}' não reconhecido. Usando 'base'")
        scenario = 'base'
    
    shock = scenario_map[scenario]
    
    # Ajustar variáveis-chave
    if 'pim_pf' in df_scenario.columns:
        df_scenario['pim_pf'] += shock * std_dict['pim_pf'] * 0.5
    
    if 'desocupacao' in df_scenario.columns:
        df_scenario['desocupacao'] += shock * std_dict['desocupacao'] * (-0.5)  # Inverso
    
    if 'selic' in df_scenario.columns:
        df_scenario['selic'] += shock * std_dict['selic'] * 0.3
    
    logger.debug(f"Cenário {scenario} aplicado com shock={shock}")
    
    return df_scenario


def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) do target (y).
    
    Parâmetros:
        df: DataFrame com features e target
        target_col: Nome da coluna target
        
    Retorno:
        (X: DataFrame de features, y: Series do target)
    """
    logger.info(f"Separando features do target ({target_col})")
    
    if target_col not in df.columns:
        logger.error(f"Target '{target_col}' não encontrado em df.columns")
        raise ValueError(f"Target '{target_col}' não encontrado")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y
