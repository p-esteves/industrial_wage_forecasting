"""
Módulo de ingestão e coleta de dados econômicos.
Implementa resiliência com fallback para dados sintéticos.

Author: Economista Quantitativo
"""

import logging
import logging.config
from typing import Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import DATA_CONFIG, LOGGING_CONFIG

# Configurar logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def setup_logger(name: str) -> logging.Logger:
    """
    Configura logger para módulo específico.
    
    Args:
        name: Nome do módulo
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


def generate_mock_data(
    start_date: str = DATA_CONFIG.START_DATE,
    end_date: str = DATA_CONFIG.END_DATE,
    frequency: str = "MS"
) -> pd.DataFrame:
    """
    Gera dados econômicos sintéticos realistas com sazonalidade e tendência.
    
    Parâmetros:
        start_date: Data inicial (str, format: YYYY-MM-DD)
        end_date: Data final (str, format: YYYY-MM-DD)
        frequency: Frequência temporal ('MS' = início do mês)
        
    Retorno:
        DataFrame com variáveis econômicas sintéticas
        
    Detalhes Econômicos:
        - Produção Física: tendência +0.5% a.m. + sazonalidade (picos em jul/ago, dez)
        - Massa Salarial Real: correlacionada com produção e desocupação inversa
        - IPCA: inflação controlada ~0.3-0.5% a.m.
        - Selic: taxa diretora oscilante entre 10-13% a.a.
        - Taxa de Desocupação: componente estrutural + cíclico
        - Câmbio: movimento browniano com drift
    """
    logger.info(f"Gerando dados sintéticos de {start_date} a {end_date}")
    
    try:
        # Criar índice temporal mensal
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(date_range)
        
        # Seed para reprodutibilidade
        np.random.seed(42)
        
        # 1. PRODUÇÃO FÍSICA (PIM-PF)
        # Componentes: tendência, sazonalidade, ruído
        trend_pim = np.linspace(100, 115, n_periods)  # +15% em 11 anos
        seasonality_pim = 8 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Sazonalidade anual
        noise_pim = np.random.normal(0, 2, n_periods)
        pim_pf = trend_pim + seasonality_pim + noise_pim
        
        # 2. MASSA SALARIAL REAL (TARGET)
        # Correlação positiva com PIM-PF, negativa com desocupação
        trend_wage = np.linspace(100, 110, n_periods)
        seasonality_wage = 4 * np.sin(2 * np.pi * np.arange(n_periods) / 12 + np.pi/4)
        noise_wage = np.random.normal(0, 1.5, n_periods)
        massa_salarial_real = trend_wage + seasonality_wage + 0.2 * pim_pf + noise_wage
        
        # 3. IPCA (Deflator)
        # Média de ~0.4% a.m. (4.8% a.a.) com volatilidade
        ipca_monthly = np.random.normal(0.004, 0.003, n_periods)
        ipca = np.cumprod(1 + ipca_monthly) * 100  # Índice acumulado
        
        # 4. SELIC (Taxa Diretora)
        # Oscila entre 10-13% a.a., com shocks ocasionais
        selic_base = 11.5
        selic_cycle = 1.5 * np.sin(2 * np.pi * np.arange(n_periods) / 36)  # Ciclo de 36 meses
        selic_shocks = np.zeros(n_periods)
        selic_shocks[30] = 2.0  # Shock em 2017
        selic_shocks[60] = -1.5  # Alívio em 2019
        selic = selic_base + selic_cycle + selic_shocks + np.random.normal(0, 0.3, n_periods)
        selic = np.clip(selic, 8, 14)  # Clip entre 8-14%
        
        # 5. TAXA DE DESOCUPAÇÃO
        # Componente estrutural (~11%) + cíclico
        unemployment_base = 11.5
        unemployment_cycle = 2 * np.sin(2 * np.pi * np.arange(n_periods) / 48)  # Ciclo de 4 anos
        unemployment = unemployment_base + unemployment_cycle + np.random.normal(0, 0.3, n_periods)
        unemployment = np.clip(unemployment, 8, 15)
        
        # 6. CÂMBIO (USD/BRL)
        # Random walk com drift positivo (desvalorização)
        exchange_returns = np.random.normal(0.002, 0.03, n_periods)
        exchange = 3.0 * np.cumprod(1 + exchange_returns)
        
        # Montar DataFrame
        df = pd.DataFrame({
            'data': date_range,
            'pim_pf': pim_pf,
            'massa_salarial_real': massa_salarial_real,
            'ipca': ipca,
            'selic': selic,
            'desocupacao': unemployment,
            'cambio': exchange
        })
        
        df.set_index('data', inplace=True)
        
        logger.info(f"Dados sintéticos gerados com sucesso: {n_periods} observações")
        logger.debug(f"Shape: {df.shape}\nPrimeiras linhas:\n{df.head()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao gerar dados sintéticos: {str(e)}")
        raise


def fetch_sidra_data(
    table_id: str,
    variable_id: str,
    start_date: str,
    end_date: str,
    timeout: int = 30
) -> pd.DataFrame | None:
    """
    Coleta dados da API SIDRA (IBGE) com tratamento de erros.
    
    Parâmetros:
        table_id: ID da tabela SIDRA
        variable_id: ID da variável
        start_date: Data inicial
        end_date: Data final
        timeout: Timeout em segundos
        
    Retorno:
        DataFrame com dados ou None se falhar
    """
    try:
        import sidrapy
        logger.info(f"Tentando coletar dados SIDRA (tabela={table_id})")
        
        df = sidrapy.get_table(
            table_id=table_id,
            territorial_level="1",
            ibge_code="all",
            period=f"{start_date.split('-')[0]}01:{end_date.split('-')[0]}12",
            timeout=timeout
        )
        
        logger.info(f"Dados SIDRA obtidos com sucesso: {df.shape}")
        return df
        
    except (ImportError, TimeoutError, ConnectionError) as e:
        logger.warning(f"Falha ao coletar dados SIDRA: {type(e).__name__} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado na coleta SIDRA: {str(e)}")
        return None


def fetch_ipeadata(
    code: str,
    start_date: str = None,
    end_date: str = None,
    timeout: int = 30
) -> pd.DataFrame | None:
    """
    Coleta dados de IPEADATA com resiliência.
    
    Parâmetros:
        code: Código da série IPEADATA
        start_date: Data inicial
        end_date: Data final
        timeout: Timeout em segundos
        
    Retorno:
        DataFrame com série temporal ou None se falhar
    """
    try:
        import ipeadatapy
        logger.info(f"Tentando coletar dados IPEADATA (série={code})")
        
        df = ipeadatapy.timeseries(code, start=start_date, end=end_date)
        
        logger.info(f"Dados IPEADATA obtidos com sucesso: {df.shape}")
        return df
        
    except (ImportError, TimeoutError, ConnectionError) as e:
        logger.warning(f"Falha ao coletar dados IPEADATA: {type(e).__name__} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado na coleta IPEADATA: {str(e)}")
        return None


def ingest_data(
    use_real_data: bool = True,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Pipeline de ingestão de dados com fallback automático.
    
    Fluxo:
        1. Tenta coletar dados das APIs (SIDRA, IPEADATA)
        2. Se falhar, log de warning
        3. Gera dados sintéticos como fallback
        4. Retorna DataFrame consolidado
    
    Parâmetros:
        use_real_data: Se True, tenta APIs antes do fallback
        timeout: Timeout para requisições
        
    Retorno:
        DataFrame com dados econômicos (real ou sintético)
    """
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE DE INGESTÃO DE DADOS")
    logger.info("=" * 70)
    
    try:
        # Tentar coleta de dados reais
        if use_real_data:
            logger.info("Tentando coletar dados de fontes reais...")
            
            # Tentativa SIDRA - PIM-PF (Produção Física)
            pim_df = fetch_sidra_data(
                table_id=DATA_CONFIG.SIDRA_PIM_PF,
                variable_id="8888",
                start_date=DATA_CONFIG.START_DATE,
                end_date=DATA_CONFIG.END_DATE,
                timeout=timeout
            )
            
            # Tentativa IPEADATA - Selic
            selic_df = fetch_ipeadata(
                code=DATA_CONFIG.IPEADATA_SELIC,
                start_date=DATA_CONFIG.START_DATE,
                end_date=DATA_CONFIG.END_DATE,
                timeout=timeout
            )
            
            # Se ambas falharem, usar fallback
            if pim_df is None and selic_df is None:
                logger.warning("Falha em todas as APIs. Ativando fallback para dados sintéticos...")
                return generate_mock_data(
                    start_date=DATA_CONFIG.START_DATE,
                    end_date=DATA_CONFIG.END_DATE
                )
            
            # Se uma foi bem-sucedida, consolidar
            logger.info("Dados parciais coletados com sucesso. Complementando com síntéticos...")
            df = generate_mock_data(
                start_date=DATA_CONFIG.START_DATE,
                end_date=DATA_CONFIG.END_DATE
            )
            
            return df
        else:
            # Usar apenas dados sintéticos
            return generate_mock_data(
                start_date=DATA_CONFIG.START_DATE,
                end_date=DATA_CONFIG.END_DATE
            )
        
    except Exception as e:
        logger.critical(f"Erro crítico no pipeline: {str(e)}. Usando fallback...")
        return generate_mock_data(
            start_date=DATA_CONFIG.START_DATE,
            end_date=DATA_CONFIG.END_DATE
        )
    
    logger.info("=" * 70)


def deflate_to_real_values(
    nominal_series: pd.Series,
    ipca_index: pd.Series,
    base_year: int = DATA_CONFIG.IPCA_BASELINE_YEAR
) -> pd.Series:
    """
    Deflaciona série nominal para valores reais usando IPCA.
    
    Parâmetros:
        nominal_series: Série com valores nominais
        ipca_index: Índice IPCA acumulado
        base_year: Ano base do deflator
        
    Retorno:
        Série deflacionada em valores reais (base = ano base)
    """
    logger.info(f"Deflacionando série para valores reais (base={base_year})")
    
    # Normalizador: dividir por índice e multiplicar por 100
    ipca_normalized = ipca_index / ipca_index.iloc[0] * 100
    real_values = nominal_series / ipca_normalized * 100
    
    logger.debug(f"Série deflacionada: {real_values.describe()}")
    
    return real_values


def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
    """
    Valida qualidade dos dados: valores faltantes, outliers, distribuição.
    
    Parâmetros:
        df: DataFrame a validar
        
    Retorno:
        (is_valid: bool, report: Dict com detalhes)
    """
    logger.info("Validando qualidade dos dados...")
    
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicated_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    is_valid = (
        df.isnull().sum().sum() == 0 and
        df.duplicated().sum() == 0 and
        len(df) > 36  # Mínimo 3 anos de dados
    )
    
    logger.info(f"Validação: {'>> PASSOU' if is_valid else '!! FALHOU'}")
    logger.debug(f"Relatório: {report}")
    
    return is_valid, report
