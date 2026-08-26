"""
Módulo de Visualização Interativa do Nowcasting vs. Forecasting (Plotly).

Gera gráficos com a série histórica, as estimativas de Nowcast no mês corrente (com intervalo de incerteza)
e o horizonte de projeção futura de Forecast.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def plot_nowcast_vs_forecast(
    y_historical: pd.Series,
    nowcast_value: float,
    nowcast_lower: float,
    nowcast_upper: float,
    forecast_series: pd.Series,
    nowcast_date: Optional[pd.Timestamp] = None,
    filename: str = "outputs/06_nowcasting_vs_forecasting.html",
) -> go.Figure:
    """
    Cria gráfico Plotly integrando Histórico, Nowcast (com banda de confiança) e Forecast.

    Parâmetros:
        y_historical: Série histórica mensal observada (até t-1 ou t-2).
        nowcast_value: Estimativa pontual do Nowcast para o mês corrente t.
        nowcast_lower: Limite inferior do intervalo de confiança do Nowcast.
        nowcast_upper: Limite superior do intervalo de confiança do Nowcast.
        forecast_series: Série de previsões de médio prazo (t+1 a t+12).
        nowcast_date: Data correspondente ao mês t (se None, assume 1 mês após y_historical).
        filename: Caminho do arquivo HTML de saída.

    Retorno:
        Objeto go.Figure do Plotly.
    """
    logger.info("Gerando visualização de Nowcasting vs Forecasting...")

    if nowcast_date is None:
        nowcast_date = y_historical.index[-1] + pd.DateOffset(months=1)

    # 1. Trançar Série Histórica
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_historical.index,
            y=y_historical.values,
            mode="lines+markers",
            name="Histórico Oficial (IBGE)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
        )
    )

    # 2. Conectar último ponto histórico ao Nowcast
    connect_x = [y_historical.index[-1], nowcast_date]
    connect_y = [y_historical.values[-1], nowcast_value]

    fig.add_trace(
        go.Scatter(
            x=connect_x,
            y=connect_y,
            mode="lines",
            showlegend=False,
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        )
    )

    # 3. Traçar o Ponto de Nowcast com Banda de Incerteza
    fig.add_trace(
        go.Scatter(
            x=[nowcast_date],
            y=[nowcast_value],
            mode="markers+text",
            name=f"Nowcast Mês Corrente (R$ {nowcast_value:.2f})",
            marker=dict(color="#ff7f0e", size=12, symbol="diamond"),
            text=[f"Nowcast: R$ {nowcast_value:.2f}"],
            textposition="top center",
        )
    )

    # Intervalo de Incerteza do Nowcast
    fig.add_trace(
        go.Scatter(
            x=[nowcast_date, nowcast_date],
            y=[nowcast_lower, nowcast_upper],
            mode="lines",
            name="Intervalo de Confiança (95%)",
            line=dict(color="#ff7f0e", width=4),
        )
    )

    # 4. Conectar Nowcast ao Forecast Futuro
    if not forecast_series.empty:
        fc_connect_x = [nowcast_date] + list(forecast_series.index)
        fc_connect_y = [nowcast_value] + list(forecast_series.values)

        fig.add_trace(
            go.Scatter(
                x=fc_connect_x,
                y=fc_connect_y,
                mode="lines+markers",
                name="Forecast de Médio Prazo (12 Meses)",
                line=dict(color="#2ca02c", width=2.5, dash="dot"),
                marker=dict(size=5),
            )
        )

    # Configuração de Layout Profissional
    fig.update_layout(
        title={
            "text": "<b>Massa Salarial Real da Indústria: Integrado Nowcasting & Forecasting</b><br><sup>Estimativa de Alta Frequência (Mês Corrente) vs. Projeção de Médio Prazo</sup>",
            "y": 0.93,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Período",
        yaxis_title="Massa Salarial Real (R$ Base 2020)",
        template="plotly_white",
        hovermode="x unified",
        height=650,
        width=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)"),
    )

    # Destacar a área de Nowcast com uma linha vertical suave
    fig.add_vline(
        x=nowcast_date.timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Mês Corrente (Nowcast)",
        annotation_position="top left",
    )

    # Salvar em HTML
    path_file = Path(filename)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path_file))
    logger.info(f"Visualização de Nowcasting salva em: {path_file}")

    return fig
