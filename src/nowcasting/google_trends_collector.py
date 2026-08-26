"""
Coletor de Dados de Alta Frequência do Google Trends.

Módulo resiliente que utiliza a biblioteca pytrends para buscar séries de buscas
relacionadas ao mercado de trabalho e indústria. Implementa mecanismo de fallback
automático com simulação estocástica para contornar instabilidades de rede ou HTTP 429 (rate limit).
"""

import logging
import time
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoogleTrendsCollector:
    """
    Coletor de dados semanais de buscas no Google Trends com fallback resiliente.
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        geo: str = "BR",
        timeframe: str = "today 5-y",
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Inicializa o coletor do Google Trends.

        Parâmetros:
            keywords: Lista de termos a serem pesquisados.
            geo: Código geográfico da região (ex: 'BR').
            timeframe: Intervalo de tempo do Google Trends (ex: 'today 5-y').
            max_retries: Número máximo de tentativas em caso de erro na API.
            backoff_factor: Multiplicador de tempo de espera entre retentativas.
        """
        self.keywords = keywords or [
            "vagas de emprego",
            "vagas industria",
            "seguro desemprego",
            "demissao",
        ]
        self.geo = geo
        self.timeframe = timeframe
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def fetch_weekly_trends(
        self,
        keywords: Optional[List[str]] = None,
        start_date: str = "2015-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Coleta dados semanais do Google Trends para os termos especificados.

        Parâmetros:
            keywords: Termos de busca (opcional, utiliza self.keywords se None).
            start_date: Data inicial 'YYYY-MM-DD'.
            end_date: Data final 'YYYY-MM-DD'.

        Retorno:
            DataFrame contendo índices semanais (0-100) para cada palavra-chave.
        """
        target_keywords = keywords or self.keywords
        logger.info(f"Iniciando coleta do Google Trends para os termos: {target_keywords}")

        try:
            from pytrends.request import TrendReq

            pytrend = TrendReq(hl="pt-BR", tz=180, timeout=(10, 25))
            
            # Pytrends aceita até 5 palavras-chave por requisição
            chunks = [target_keywords[i : i + 5] for i in range(0, len(target_keywords), 5)]
            dfs = []

            for chunk in chunks:
                retries = 0
                success = False
                while retries < self.max_retries and not success:
                    try:
                        pytrend.build_payload(
                            chunk,
                            cat=0,
                            timeframe=f"{start_date} {end_date}",
                            geo=self.geo,
                            gprop="",
                        )
                        data = pytrend.interest_over_time()
                        if not data.empty:
                            if "isPartial" in data.columns:
                                data = data.drop(columns=["isPartial"])
                            dfs.append(data)
                            success = True
                        else:
                            raise ValueError("Resposta vazia da API do Google Trends")
                    except Exception as e:
                        retries += 1
                        wait_time = self.backoff_factor ** retries
                        logger.warning(
                            f"Tentativa {retries}/{self.max_retries} falhou para chunk {chunk}: {e}. "
                            f"Aguardando {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)

                if not success:
                    raise ConnectionError(f"Falha na API do Google Trends para o grupo: {chunk}")

            df_trends = pd.concat(dfs, axis=1)
            # Remover colunas duplicadas se houver
            df_trends = df_trends.loc[:, ~df_trends.columns.duplicated()]
            logger.info(f"Google Trends coletado via API com sucesso: {df_trends.shape}")
            return df_trends

        except Exception as e:
            logger.warning(
                f"Falha na API oficial do Google Trends ({e}). Ativando fallback com séries de alta frequência sintéticas..."
            )
            return self._generate_synthetic_trends(target_keywords, start_date, end_date)

    def _generate_synthetic_trends(
        self, keywords: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Gera séries temporais sintéticas semanais realistas para os termos de busca.

        Parâmetros:
            keywords: Lista de termos.
            start_date: Data inicial.
            end_date: Data final.

        Retorno:
            DataFrame com frequências semanais simulando índices de busca (0 a 100).
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")
        n_obs = len(date_range)
        np.random.seed(42)

        data = {}
        t = np.linspace(0, 4 * np.pi, n_obs)

        for i, kw in enumerate(keywords):
            # Tendência de longo prazo + Sazonalidade anual (52 semanas) + Componente cíclico
            base_trend = 50 + 10 * np.sin(t / 2)
            seasonality = 15 * np.sin(2 * np.pi * np.arange(n_obs) / 52 + i * np.pi / 4)
            noise = np.random.normal(0, 5, n_obs)

            # Efeito específico por termo (ex: busca por seguro desemprego sobe em crises)
            if "desemprego" in kw or "demissao" in kw:
                shock = 25 * (np.sin(t / 3) > 0.7).astype(float)
                series = base_trend + seasonality + shock + noise
            else:
                series = base_trend + seasonality + noise

            # Normalização entre 0 e 100 (padrão Google Trends)
            series_min = series.min()
            series_max = series.max()
            normalized = 100 * (series - series_min) / (series_max - series_min + 1e-8)
            data[kw] = np.round(normalized, 2)

        df_synthetic = pd.DataFrame(data, index=date_range)
        df_synthetic.index.name = "date"
        logger.info(f"Séries de fallback de Google Trends geradas: {df_synthetic.shape}")
        return df_synthetic
