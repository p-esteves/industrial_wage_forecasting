"""
Configurações gerais do projeto de forecasting de Massa Salarial.
Author: Economista Quantitativo
Date: Janeiro 2026
"""

from dataclasses import dataclass
from typing import Dict, List

# ==================== CONFIGURAÇÕES DE DADOS ====================

@dataclass
class DataConfig:
    """Configuração de fontes de dados e parâmetros econômicos."""
    
    # Períodos de coleta
    START_DATE: str = "2015-01-01"
    END_DATE: str = "2025-12-31"
    FORECAST_HORIZON: int = 12  # meses
    
    # Códigos SIDRA (IBGE)
    SIDRA_PIM_PF: str = "8888"      # Produção Física - Transformação
    SIDRA_IPCA: str = "69"           # IPCA - Inflação
    
    # Codes IPEADATA
    IPEADATA_SELIC: str = "BM12_TAXA12"
    IPEADATA_UNEMPLOYMENT: str = "SEADE12_TXDESOCUP12"
    IPEADATA_EXCHANGE: str = "EXR12_USA12"
    
    # Deflator
    IPCA_BASELINE_YEAR: int = 2020

# ==================== CONFIGURAÇÕES DE FEATURES ====================

@dataclass
class FeatureConfig:
    """Engenharia de features econômicas."""
    
    LAG_PERIODS: List[int] = None
    ROLLING_WINDOWS: List[int] = None
    SEASONAL_COMPONENTS: bool = True
    SHOCK_FEATURES: bool = True
    
    def __post_init__(self):
        self.LAG_PERIODS = self.LAG_PERIODS or [1, 3, 6, 12]
        self.ROLLING_WINDOWS = self.ROLLING_WINDOWS or [3, 6]

# ==================== CONFIGURAÇÕES DE MODELAGEM ====================

@dataclass
class ModelConfig:
    """Parâmetros de treino dos modelos."""
    
    # Prophet
    PROPHET_SEASONALITY_MODE: str = "additive"
    PROPHET_INTERVAL_WIDTH: float = 0.95
    PROPHET_YEARLY_SEASONALITY: bool = True
    
    # XGBoost
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_N_ESTIMATORS: int = 200
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8
    
    # Validação temporal
    TEST_SIZE: int = 12  # 12 meses para teste
    CV_SPLITS: int = 5

# ==================== CONFIGURAÇÕES DE CENÁRIOS ====================

@dataclass
class ScenarioConfig:
    """Cenários macroeconomicos para projeção."""
    
    SCENARIOS: Dict[str, float] = None
    
    def __post_init__(self):
        if self.SCENARIOS is None:
            self.SCENARIOS = {
                "pessimista": -1.0,      # -1 desvio padrão
                "base": 0.0,              # cenário base
                "otimista": 1.0           # +1 desvio padrão
            }

# ==================== CONFIGURAÇÕES DE VISUALIZAÇÃO ====================

@dataclass
class VisualizationConfig:
    """Parâmetros para gráficos Plotly."""
    
    TEMPLATE: str = "plotly_white"
    COLOR_PALETTE: Dict[str, str] = None
    HEIGHT: int = 600
    WIDTH: int = 1200
    FONT_SIZE: int = 12
    
    def __post_init__(self):
        if self.COLOR_PALETTE is None:
            self.COLOR_PALETTE = {
                "base": "#1f77b4",
                "otimista": "#2ca02c",
                "pessimista": "#d62728",
                "forecast": "#ff7f0e"
            }

# ==================== CONFIGURAÇÕES DE NOWCASTING ====================

@dataclass
class NowcastingConfig:
    """Parâmetros para o módulo de Nowcasting de Alta Frequência."""
    
    # Palavras-chave para Google Trends
    TRENDS_KEYWORDS: List[str] = None
    TRENDS_GEO: str = "BR"
    TRENDS_TIMEFRAME: str = "today 5-y"
    
    # Parâmetros de Carga de Energia (ONS)
    ONS_SUBSYSTEMS: List[str] = None
    
    # Frequência e Alinhamento
    NOWCAST_CUTOFF_DAYS: List[int] = None  # Dias de corte para pseudo real-time backtest (ex: dia 10, 20, 30 do mês)
    
    # Modelos de Nowcasting
    BRIDGE_ALPHA: float = 1.0
    BRIDGE_L1_RATIO: float = 0.5  # ElasticNet
    NOWCAST_XGB_MAX_DEPTH: int = 4
    NOWCAST_XGB_LEARNING_RATE: float = 0.03
    NOWCAST_XGB_N_ESTIMATORS: int = 150
    
    # Diretórios de Saída
    RESULTS_DIR: str = "results"
    OUTPUTS_DIR: str = "outputs"

    def __post_init__(self):
        if self.TRENDS_KEYWORDS is None:
            self.TRENDS_KEYWORDS = ["vagas de emprego", "vagas industria", "seguro desemprego", "demissao"]
        if self.ONS_SUBSYSTEMS is None:
            self.ONS_SUBSYSTEMS = ["SE", "S", "NE", "N"]
        if self.NOWCAST_CUTOFF_DAYS is None:
            self.NOWCAST_CUTOFF_DAYS = [10, 20, 30]

# ==================== LOGGING ====================

LOGGING_CONFIG: Dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.FileHandler",
            "filename": "logs/forecasting.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": True
        }
    }
}

# ==================== INSTÂNCIAS PADRÃO ====================

DATA_CONFIG = DataConfig()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
SCENARIO_CONFIG = ScenarioConfig()
VIZ_CONFIG = VisualizationConfig()
NOWCASTING_CONFIG = NowcastingConfig()
