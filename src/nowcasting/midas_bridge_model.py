"""
Modelos de Nowcasting de Frequências Mistas (Bridge Models & XGBoost Nowcaster).

Implementa modelos econométricos de regressão ponte (Bridge Equations / U-MIDAS)
e um modelo de Machine Learning supervisionado (XGBoost Nowcaster) projetados
para estimativa em tempo real com dados incompletos.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class MIDASBridgeModel:
    """
    Modelo de Regressão Ponte (Bridge Equation / U-MIDAS) com regularização ElasticNet.
    
    Relaciona agregados mensais de alta frequência e lags autorregressivos com a variável alvo.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, random_state: int = 42) -> None:
        """
        Inicializa o modelo Bridge.

        Parâmetros:
            alpha: Força da penalidade de regularização ElasticNet.
            l1_ratio: Balanço entre L1 (Lasso) e L2 (Ridge).
            random_state: Semente aleatória para reprodutibilidade.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state, max_iter=5000
        )
        self.is_fitted = False
        self.residual_std_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIDASBridgeModel":
        """
        Treina o modelo Bridge Equation.

        Parâmetros:
            X: Matriz de preditores de frequência mistos.
            y: Série temporal alvo mensal.
        """
        # Tratar eventuais NaNs em X por imputação no limite
        X_clean = X.ffill().bfill().fillna(0)
        
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y)
        
        # Calcular desvio padrão dos resíduos de treino para intervalos de confiança
        y_pred = self.model.predict(X_scaled)
        self.residual_std_ = float(np.std(y - y_pred))
        
        self.is_fitted = True
        logger.info(f"MIDASBridgeModel treinado com sucesso. RMSE treino: {self.residual_std_:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gera estimativas de Nowcast para a matriz X.
        """
        if not self.is_fitted:
            raise RuntimeError("O modelo deve ser treinado com .fit() antes de prever.")
            
        X_clean = X.ffill().bfill().fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

    def predict_with_interval(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gera a previsão pontual de Nowcast acompanhada do intervalo de confiança.

        Parâmetros:
            X: Matriz de atributos.
            confidence_level: Grau de confiança (ex: 0.95 para 95%).

        Retorno:
            Tuple (y_pred, lower_bound, upper_bound)
        """
        y_pred = self.predict(X)
        z = 1.96 if confidence_level == 0.95 else 1.645
        margin = z * self.residual_std_
        
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        return y_pred, lower_bound, upper_bound


class XGBoostNowcaster:
    """
    Modelo Nowcaster supervisionado baseado em XGBoost.
    
    Capaz de capturar não-linearidades entre variáveis de alta frequência e a série-alvo.
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        """
        Inicializa o XGBoost Nowcaster.
        """
        self.model = XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.is_fitted = False
        self.residual_std_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostNowcaster":
        """
        Treina o XGBoost Nowcaster.
        """
        X_clean = X.ffill().bfill().fillna(0)
        self.model.fit(X_clean, y)
        
        y_pred = self.model.predict(X_clean)
        self.residual_std_ = float(np.std(y - y_pred))
        self.is_fitted = True
        logger.info(f"XGBoostNowcaster treinado com sucesso. RMSE treino: {self.residual_std_:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gera estimativas de Nowcast.
        """
        if not self.is_fitted:
            raise RuntimeError("O modelo deve ser treinado antes de prever.")
            
        X_clean = X.ffill().bfill().fillna(0)
        return self.model.predict(X_clean)

    def predict_with_interval(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna a previsão pontual com intervalo de incerteza empírico.
        """
        y_pred = self.predict(X)
        z = 1.96 if confidence_level == 0.95 else 1.645
        margin = z * self.residual_std_
        return y_pred, y_pred - margin, y_pred + margin

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Retorna a importância relativa de cada variável de entrada.
        """
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(x) for x in importances]))
