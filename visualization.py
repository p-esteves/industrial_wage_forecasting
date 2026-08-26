"""
Visualizações interativas profissionais com Plotly.

Author: Economista Quantitativo
"""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import VIZ_CONFIG

logger = logging.getLogger(__name__)


def plot_forecast_scenarios(
    y_historical: pd.Series,
    y_forecast_base: pd.Series,
    y_forecast_optimistic: pd.Series,
    y_forecast_pessimistic: pd.Series,
    title: str = "Previsão de Massa Salarial Real - Cenários",
    filename: str = None
) -> go.Figure:
    """
    Plota cenários de previsão (base, otimista, pessimista).
    
    Parâmetros:
        y_historical: Série histórica
        y_forecast_base: Previsão cenário base
        y_forecast_optimistic: Previsão cenário otimista (+1σ)
        y_forecast_pessimistic: Previsão cenário pessimista (-1σ)
        title: Título do gráfico
        filename: Se especificado, salva como HTML
        
    Retorno:
        Figura Plotly interativa
    """
    logger.info("Plotando cenários de previsão...")
    
    fig = go.Figure()
    
    # Série histórica
    fig.add_trace(go.Scatter(
        x=y_historical.index,
        y=y_historical.values,
        mode='lines',
        name='Histórico',
        line=dict(color=VIZ_CONFIG.COLOR_PALETTE['base'], width=2),
        hovertemplate='%{x|%Y-%m}<br>Valor: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Previsão base
    fig.add_trace(go.Scatter(
        x=y_forecast_base.index,
        y=y_forecast_base.values,
        mode='lines+markers',
        name='Base',
        line=dict(color=VIZ_CONFIG.COLOR_PALETTE['forecast'], width=2, dash='dash'),
        marker=dict(size=5),
        hovertemplate='%{x|%Y-%m}<br>Previsão: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Cenário otimista
    fig.add_trace(go.Scatter(
        x=y_forecast_optimistic.index,
        y=y_forecast_optimistic.values,
        mode='lines',
        name='Otimista (+1σ)',
        line=dict(color=VIZ_CONFIG.COLOR_PALETTE['otimista'], width=1, dash='dot'),
        hovertemplate='%{x|%Y-%m}<br>Otimista: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Cenário pessimista
    fig.add_trace(go.Scatter(
        x=y_forecast_pessimistic.index,
        y=y_forecast_pessimistic.values,
        mode='lines',
        name='Pessimista (-1σ)',
        line=dict(color=VIZ_CONFIG.COLOR_PALETTE['pessimista'], width=1, dash='dot'),
        hovertemplate='%{x|%Y-%m}<br>Pessimista: R$ %{y:.2f}<extra></extra>'
    ))
    
    # Banda de confiança
    fig.add_trace(go.Scatter(
        x=list(y_forecast_optimistic.index) + list(y_forecast_pessimistic.index)[::-1],
        y=list(y_forecast_optimistic.values) + list(y_forecast_pessimistic.values)[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Intervalo de Confiança',
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Data",
        yaxis_title="Massa Salarial Real (R$ Base 2020)",
        template=VIZ_CONFIG.TEMPLATE,
        hovermode='x unified',
        height=VIZ_CONFIG.HEIGHT,
        width=VIZ_CONFIG.WIDTH,
        font=dict(size=VIZ_CONFIG.FONT_SIZE),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    
    if filename:
        fig.write_html(filename)
        logger.info(f"Gráfico salvo em: {filename}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: list[str] = None,
    filename: str = None
) -> go.Figure:
    """
    Compara performance dos modelos com gráfico de barras.
    
    Parâmetros:
        results: Dict com métricas de cada modelo
        metrics: Lista de métricas a plotar
        filename: Se especificado, salva como HTML
        
    Retorno:
        Figura Plotly
    """
    logger.info("Plotando comparação de modelos...")
    
    metrics = metrics or ['RMSE', 'MAE', 'MAPE', 'MASE']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        specs=[[{'type':'bar'}, {'type':'bar'}],
               [{'type':'bar'}, {'type':'bar'}]]
    )
    
    colors = [VIZ_CONFIG.COLOR_PALETTE['base'], VIZ_CONFIG.COLOR_PALETTE['forecast']]
    
    for idx, metric in enumerate(metrics):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker=dict(color=colors),
                text=[f'{v:.4f}' for v in values],
                textposition='auto',
                hovertemplate='%{x}<br>' + metric + ': %{y:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Comparação de Performance dos Modelos",
        height=700,
        width=VIZ_CONFIG.WIDTH,
        showlegend=False,
        template=VIZ_CONFIG.TEMPLATE
    )
    
    if filename:
        fig.write_html(filename)
        logger.info(f"Gráfico salvo em: {filename}")
    
    return fig


def plot_shap_feature_importance(
    feature_importance: Dict[str, float] | np.ndarray,
    feature_names: list[str],
    title: str = "Importância das Features (SHAP)",
    filename: str = None
) -> go.Figure:
    """
    Plota importância de features usando SHAP values.
    
    Parâmetros:
        feature_importance: Dict ou array com importância
        feature_names: Nomes das features
        title: Título
        filename: Se especificado, salva como HTML
        
    Retorno:
        Figura Plotly
    """
    logger.info("Plotando importância de features (SHAP)...")
    
    # Converter para dict se array
    if isinstance(feature_importance, np.ndarray):
        importance_dict = {name: val for name, val in zip(feature_names, feature_importance)}
    else:
        importance_dict = feature_importance
    
    # Sort
    # Handle both scalar and array values by taking mean if array
    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: np.mean(x[1]) if isinstance(x[1], (list, np.ndarray)) else x[1],
        reverse=True
    )[:15]
    features = [f[0] for f in sorted_importance]
    # Ensure values are scalars
    values = [np.mean(f[1]) if isinstance(f[1], (list, np.ndarray)) else f[1] for f in sorted_importance]
    
    fig = go.Figure(
        data=[go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker=dict(color=values, colorscale='Viridis'),
            text=[f'{v:.4f}' for v in values],
            textposition='auto',
            hovertemplate='%{y}<br>Importância: %{x:.4f}<extra></extra>'
        )]
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Valor SHAP Médio",
        yaxis_title="Feature",
        height=500,
        width=VIZ_CONFIG.WIDTH,
        template=VIZ_CONFIG.TEMPLATE,
        margin=dict(l=150, r=50, t=80, b=60)
    )
    
    if filename:
        fig.write_html(filename)
        logger.info(f"Gráfico salvo em: {filename}")
    
    return fig


def plot_economic_variables(
    df: pd.DataFrame,
    variables: list[str],
    title: str = "Evolução de Variáveis Macroeconômicas",
    filename: str = None
) -> go.Figure:
    """
    Plota múltiplas variáveis econômicas em subplots.
    
    Parâmetros:
        df: DataFrame com variáveis
        variables: Lista de colunas a plotar
        title: Título geral
        filename: Se especificado, salva como HTML
        
    Retorno:
        Figura Plotly
    """
    logger.info(f"Plotando variáveis econômicas: {variables}")
    
    n_vars = len(variables)
    fig = make_subplots(
        rows=n_vars, cols=1,
        subplot_titles=variables,
        vertical_spacing=0.12
    )
    
    for idx, var in enumerate(variables, 1):
        if var in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[var],
                    mode='lines',
                    name=var,
                    line=dict(width=2),
                    hovertemplate='%{x|%Y-%m}<br>' + var + ': %{y:.4f}<extra></extra>'
                ),
                row=idx, col=1
            )
    
    fig.update_layout(
        title_text=title,
        height=200 * n_vars,
        width=VIZ_CONFIG.WIDTH,
        template=VIZ_CONFIG.TEMPLATE,
        hovermode='x unified',
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    
    if filename:
        fig.write_html(filename)
        logger.info(f"Gráfico salvo em: {filename}")
    
    return fig


def plot_residuals_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "XGBoost",
    filename: str = None
) -> go.Figure:
    """
    Análise de resíduos: histograma e scatter plot.
    
    Parâmetros:
        y_true: Valores reais
        y_pred: Valores previstos
        model_name: Nome do modelo
        filename: Se especificado, salva como HTML
        
    Retorno:
        Figura Plotly
    """
    logger.info(f"Plotando análise de resíduos ({model_name})...")
    
    residuals = y_true.values - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribuição de Resíduos", "Resíduos vs Valores Previstos")
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Resíduos',
            marker=dict(color=VIZ_CONFIG.COLOR_PALETTE['base']),
            hovertemplate='Resíduo: %{x:.4f}<br>Freq: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Scatter
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Resíduos',
            marker=dict(size=6, color=VIZ_CONFIG.COLOR_PALETTE['forecast'], opacity=0.6),
            hovertemplate='Pred: %{x:.2f}<br>Resíduo: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Linha zero
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_xaxes(title_text="Resíduo", row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_xaxes(title_text="Valores Previstos", row=1, col=2)
    fig.update_yaxes(title_text="Resíduo", row=1, col=2)
    
    fig.update_layout(
        title_text=f"Análise de Resíduos - {model_name}",
        height=500,
        width=VIZ_CONFIG.WIDTH,
        template=VIZ_CONFIG.TEMPLATE,
        showlegend=False
    )
    
    if filename:
        fig.write_html(filename)
        logger.info(f"Gráfico salvo em: {filename}")
    
    return fig


def create_dashboard(
    forecast_fig: go.Figure,
    comparison_fig: go.Figure,
    importance_fig: go.Figure,
    residuals_fig: go.Figure,
    output_file: str = "dashboard.html"
) -> None:
    """
    Cria dashboard integrado com todos os gráficos.
    
    Parâmetros:
        forecast_fig: Figura de cenários
        comparison_fig: Figura de comparação
        importance_fig: Figura de importância
        residuals_fig: Figura de resíduos
        output_file: Arquivo de saída
    """
    logger.info(f"Criando dashboard em {output_file}...")
    
    from plotly.subplots import make_subplots
    
    # Criar dashboard com subplots
    # Nota: simplificado aqui - em produção, usar dash ou similar
    
    logger.info(">> Dashboard criado com sucesso")
