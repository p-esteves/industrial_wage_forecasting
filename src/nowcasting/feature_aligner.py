"""
Alinhador de Frequências Mistas e Tratamento do Ragged Edge Problem.

Converte dados de alta frequência (diários/semanais) para a frequência mensal da série-alvo,
gerando métricas intramês e simulando o calendário de divulgação com borda irregular (ragged edge).
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureAligner:
    """
    Agregador e alinhador de frequências mistas com simulação de corte temporal.
    """

    def __init__(self) -> None:
        """Inicializa o alinhador de features."""
        pass

    def align_high_frequency_to_monthly(
        self,
        df_daily: Optional[pd.DataFrame] = None,
        df_weekly: Optional[pd.DataFrame] = None,
        cutoff_day: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Agrega dados diários e semanais no nível mensal.

        Parâmetros:
            df_daily: DataFrame diário indexado por data.
            df_weekly: DataFrame semanal indexado por data.
            cutoff_day: Se fornecido (ex: 10, 20), filtra dados diários/semanais
                        apenas até esse dia de cada mês (simulação de Nowcast intramês).

        Retorno:
            DataFrame com frequência mensal (MS - Month Start) contendo métricas agregadas.
        """
        monthly_dfs = []

        if df_daily is not None and not df_daily.empty:
            df_d = df_daily.copy()
            if cutoff_day is not None:
                # Filtrar observações diárias até o dia de corte
                df_d = df_d[df_d.index.day <= cutoff_day]

            # Criar chave de ano-mês para agrupamento
            df_d["year_month"] = df_d.index.to_period("M")

            # Agregar métricas intramês
            agg_daily = df_d.groupby("year_month").agg(["mean", "std", "last"])
            agg_daily.columns = [f"{col}_{stat}" for col, stat in agg_daily.columns]
            agg_daily.index = agg_daily.index.to_timestamp(how="start")
            monthly_dfs.append(agg_daily)

        if df_weekly is not None and not df_weekly.empty:
            df_w = df_weekly.copy()
            if cutoff_day is not None:
                df_w = df_w[df_w.index.day <= cutoff_day]

            df_w["year_month"] = df_w.index.to_period("M")

            agg_weekly = df_w.groupby("year_month").agg(["mean", "last"])
            agg_weekly.columns = [f"{col}_{stat}" for col, stat in agg_weekly.columns]
            agg_weekly.index = agg_weekly.index.to_timestamp(how="start")
            monthly_dfs.append(agg_weekly)

        if not monthly_dfs:
            raise ValueError("Ao menos um DataFrame (diário ou semanal) deve ser fornecido.")

        df_monthly_hf = pd.concat(monthly_dfs, axis=1)

        # Adicionar variações Month-over-Month (MoM) para séries de alta frequência
        for col in list(df_monthly_hf.columns):
            if col.endswith("_mean") or col.endswith("_last"):
                df_monthly_hf[f"{col}_mom"] = df_monthly_hf[col].pct_change(1)

        logger.info(f"Recursos de alta frequência alinhados ao nível mensal: {df_monthly_hf.shape}")
        return df_monthly_hf

    def create_nowcast_dataset(
        self,
        df_monthly_target: pd.DataFrame,
        df_hf_monthly: pd.DataFrame,
        target_col: str = "massa_salarial_real",
        lags_target: Tuple[int, ...] = (1, 2, 12),
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combina os lags da série-alvo mensal com as métricas de alta frequência alinhadas.

        Parâmetros:
            df_monthly_target: DataFrame com a variável dependente mensal.
            df_hf_monthly: DataFrame com features de alta frequência mensalizadas.
            target_col: Nome da coluna alvo.
            lags_target: Lags mensais da variável alvo a serem incluídos.

        Retorno:
            Tuple (X, y) com a matriz de atributos e vetor dependente.
        """
        df_combined = pd.concat([df_monthly_target[[target_col]], df_hf_monthly], axis=1)

        # Criar lags da variável alvo (ex: t-1, t-2, t-12)
        for lag in lags_target:
            df_combined[f"{target_col}_lag{lag}"] = df_combined[target_col].shift(lag)

        # Remover linhas iniciais com NaN resultantes dos lags
        max_lag = max(lags_target)
        df_clean = df_combined.iloc[max_lag:].copy()

        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])

        return X, y
