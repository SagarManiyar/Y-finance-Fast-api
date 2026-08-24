import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import pattern detection
try:
    from patterns import detect_patterns_for_chart, CandlestickPatterns, TechnicalIndicators

    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    st.warning("⚠️ patterns.py not found. Pattern detection and advanced indicators disabled.")

# Import news functionality
try:
    from news_app import NewsProvider, display_news_app

    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    st.warning("⚠️ news_app.py not found. News functionality disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Technical Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TechnicalAnalysisEngine:
    """Core engine for technical analysis computations with enhanced signal analysis"""

    @staticmethod
    def calculate_sma(data: pd.Series, period: int, method: str = "pandas") -> pd.Series:
        """Calculate Simple Moving Average"""
        if method == "excel":
            sma = data.rolling(window=period, min_periods=period).mean()
        else:
            sma = data.rolling(window=period, min_periods=1).mean()
        return sma

    @staticmethod
    def get_sma_status(current_price: float, sma1_value: float, sma2_value: float,
                       previous_price: float, previous_sma1: float, previous_sma2: float) -> str:
        """Determine SMA trading status"""
        if pd.isna(sma1_value) or pd.isna(sma2_value) or pd.isna(previous_sma1) or pd.isna(previous_sma2):
            return "NEUTRAL"

        golden_cross = (sma1_value > sma2_value) and (previous_sma1 <= previous_sma2)
        death_cross = (sma1_value < sma2_value) and (previous_sma1 >= previous_sma2)
        strong_bullish = (current_price > sma1_value) and (current_price > sma2_value) and (sma1_value > sma2_value)
        strong_bearish = (current_price < sma1_value) and (current_price < sma2_value) and (sma1_value < sma2_value)
        price_above_short_sma = current_price > sma1_value
        price_trend_up = current_price > previous_price

        if golden_cross:
            return "BUY"
        elif death_cross:
            return "SELL"
        elif strong_bullish and price_trend_up:
            return "BUY"
        elif strong_bearish and not price_trend_up:
            return "SELL"
        elif price_above_short_sma and (sma1_value > sma2_value):
            return "BUY"
        elif not price_above_short_sma and (sma1_value < sma2_value):
            return "SELL"
        else:
            return "NEUTRAL"

    @staticmethod
    def get_macd_status(macd_value: float, signal_value: float, histogram_value: float,
                        previous_macd: float, previous_signal: float, previous_histogram: float) -> str:
        """Determine MACD trading status"""
        if any(pd.isna(val) for val in [macd_value, signal_value, previous_macd, previous_signal]):
            return "NEUTRAL"

        bullish_crossover = (macd_value > signal_value) and (previous_macd <= previous_signal)
        bearish_crossover = (macd_value < signal_value) and (previous_macd >= previous_signal)
        histogram_strengthening = histogram_value > previous_histogram
        histogram_weakening = histogram_value < previous_histogram
        strong_bullish = (macd_value > signal_value) and (macd_value > 0) and (signal_value > 0)
        strong_bearish = (macd_value < signal_value) and (macd_value < 0) and (signal_value < 0)

        if bullish_crossover:
            return "BUY"
        elif bearish_crossover:
            return "SELL"
        elif strong_bullish and histogram_strengthening:
            return "BUY"
        elif strong_bearish and histogram_weakening:
            return "SELL"
        elif (macd_value > signal_value) and histogram_strengthening:
            return "BUY"
        elif (macd_value < signal_value) and histogram_weakening:
            return "SELL"
        else:
            return "NEUTRAL"

    @staticmethod
    def get_stochastic_status(current_k: float, current_d: float, previous_k: float, previous_d: float,
                              k_history: pd.Series = None, d_history: pd.Series = None) -> str:
        """Determine Stochastic Oscillator trading status"""
        if any(pd.isna(val) for val in [current_k, current_d, previous_k, previous_d]):
            return "NEUTRAL"

        overbought_level = 80
        oversold_level = 20

        bullish_crossover = (current_k > current_d) and (previous_k <= previous_d)
        bearish_crossover = (current_k < current_d) and (previous_k >= previous_d)
        oversold_bounce = (current_k < oversold_level and current_d < oversold_level and
                           current_k > previous_k and current_d > previous_d)
        overbought_decline = (current_k > overbought_level and current_d > overbought_level and
                              current_k < previous_k and current_d < previous_d)
        strong_upward_momentum = (current_k > current_d and current_k > previous_k and
                                  current_d > previous_d and current_k < overbought_level)
        strong_downward_momentum = (current_k < current_d and current_k < previous_k and
                                    current_d < previous_d and current_k > oversold_level)
        recovering_from_oversold = (previous_k < oversold_level and current_k > oversold_level and
                                    current_k > current_d)
        declining_from_overbought = (previous_k > overbought_level and current_k < overbought_level and
                                     current_k < current_d)

        if bullish_crossover and current_k < overbought_level:
            return "BUY"
        elif bearish_crossover and current_k > oversold_level:
            return "SELL"
        elif oversold_bounce:
            return "BUY"
        elif overbought_decline:
            return "SELL"
        elif recovering_from_oversold:
            return "BUY"
        elif declining_from_overbought:
            return "SELL"
        elif strong_upward_momentum:
            return "BUY"
        elif strong_downward_momentum:
            return "SELL"
        else:
            return "NEUTRAL"

    @staticmethod
    def get_volume_status(current_volume: float, avg_volume: float, price_change_pct: float,
                          volume_history: pd.Series = None, price_history: pd.Series = None) -> str:
        """Determine Volume trading status"""
        if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
            return "NEUTRAL"

        volume_ratio = current_volume / avg_volume
        high_volume_threshold = 1.5
        very_high_volume_threshold = 2.0
        low_volume_threshold = 0.7
        significant_price_move = abs(price_change_pct) > 2.0
        strong_price_move = abs(price_change_pct) > 5.0

        if volume_ratio >= very_high_volume_threshold:
            if price_change_pct > 2.0:
                return "BUY"
            elif price_change_pct < -2.0:
                return "SELL"
            else:
                return "NEUTRAL"
        elif volume_ratio >= high_volume_threshold:
            if price_change_pct > 1.0:
                return "BUY"
            elif price_change_pct < -1.0:
                return "SELL"
            else:
                return "NEUTRAL"
        elif volume_ratio <= low_volume_threshold:
            return "NEUTRAL"
        else:
            if strong_price_move and volume_ratio > 1.0:
                if price_change_pct > 0:
                    return "BUY"
                else:
                    return "SELL"
            else:
                return "NEUTRAL"

    @staticmethod
    def calculate_volume_indicators(volume: pd.Series, close: pd.Series, avg_period: int = 20) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        avg_volume = volume.rolling(window=avg_period, min_periods=1).mean()
        volume_ratio = volume / avg_volume.replace(0, np.finfo(float).eps)
        volume_roc = volume.pct_change(periods=1) * 100
        price_change = close.pct_change()

        # On-Balance Volume (OBV)
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0] if not volume.empty else 0

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        volume_ma = volume.rolling(window=avg_period, min_periods=1).mean()

        return {
            'volume': volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_roc': volume_roc,
            'obv': obv,
            'volume_ma': volume_ma,
            'price_change_pct': price_change * 100
        }

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample OHLCV data to different intervals"""
        freq_map = {
            '1d': 'D',
            '1w': 'W',
            '1mo': 'ME',
            '3mo': 'QE',
            '1y': 'YE'
        }

        if interval not in freq_map:
            raise ValueError(f"Unsupported interval: {interval}")

        freq = freq_map[interval]
        resampled = df.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled


class DataProvider:
    """Data provider using Yahoo Finance"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300

    def get_ohlcv_data(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[
        pd.DataFrame]:
        """Fetch OHLCV data with caching"""
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        current_time = time.time()

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data.copy()

        try:
            warmup_days = 300
            actual_start = pd.to_datetime(start_date) - timedelta(days=warmup_days)
            yf_ticker = yf.Ticker(ticker)

            yf_interval_map = {
                '1d': '1d',
                '1w': '1wk',
                '1mo': '1mo',
                '3mo': '3mo',
                '1y': '1y'
            }
            yf_interval = yf_interval_map.get(interval, '1d')

            if interval != '1d':
                data = yf_ticker.history(
                    start=actual_start.strftime('%Y-%m-%d'),
                    end=end_date,
                    interval='1d',
                    auto_adjust=True,
                    prepost=False
                )

                if not data.empty:
                    engine = TechnicalAnalysisEngine()
                    data = engine.resample_ohlcv(data, interval)
            else:
                data = yf_ticker.history(
                    start=actual_start.strftime('%Y-%m-%d'),
                    end=end_date,
                    interval=yf_interval,
                    auto_adjust=True,
                    prepost=False
                )

            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return None

            self.cache[cache_key] = (data.copy(), current_time)
            data = data[data.index >= start_date]
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None


class ChartRenderer:
    """Chart rendering using Plotly"""

    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, ticker: str, sma1: int, sma2: int,
                                 sma1_data: pd.Series, sma2_data: pd.Series,
                                 interval: str, pattern_annotations: List[Dict] = None,
                                 macd_data: Dict = None, show_macd: bool = False,
                                 rsi_data: Dict = None, show_rsi: bool = False,
                                 stoch_data: Dict = None, show_stochastic: bool = False,
                                 volume_data: Dict = None) -> go.Figure:
        """Create interactive candlestick chart with clickable pattern markers"""
        subplot_count = 1
        subplot_titles = [f'{ticker} - Candlestick Chart ({interval.upper()})']
        row_heights = [0.4]

        if show_macd and macd_data:
            subplot_count += 1
            subplot_titles.append('MACD')
            row_heights.append(0.2)

        if show_rsi and rsi_data:
            subplot_count += 1
            subplot_titles.append('RSI')
            row_heights.append(0.2)

        if show_stochastic and stoch_data:
            subplot_count += 1
            subplot_titles.append('Stochastic Oscillator')
            row_heights.append(0.2)

        subplot_count += 1
        subplot_titles.append('Volume')
        row_heights.append(0.1)

        if volume_data:
            subplot_count += 1
            subplot_titles.append('On-Balance Volume (OBV)')
            row_heights.append(0.1)

        total_height = sum(row_heights)
        row_heights = [h / total_height for h in row_heights]

        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )

        # Candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )

        # SMA traces
        if not sma1_data.empty and not sma1_data.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma1_data,
                    mode='lines',
                    name=f'SMA({sma1})',
                    line=dict(color='#ffa500', width=2)
                ),
                row=1, col=1
            )

        if not sma2_data.empty and not sma2_data.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma2_data,
                    mode='lines',
                    name=f'SMA({sma2})',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )

        current_row = 2

        # Add MACD subplot
        if show_macd and macd_data:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd_data['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='#00aaff', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd_data['signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#ff6600', width=2)
                ),
                row=current_row, col=1
            )

            histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in macd_data['histogram']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=macd_data['histogram'],
                    name='Histogram',
                    marker_color=histogram_colors,
                    opacity=0.8
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            current_row += 1

        # Add RSI subplot
        if show_rsi and rsi_data:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi_data['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#9932cc', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi_data['overbought'],
                    mode='lines',
                    name='Overbought (70)',
                    line=dict(color='#ff4444', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi_data['oversold'],
                    mode='lines',
                    name='Oversold (30)',
                    line=dict(color='#00ff88', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi_data['midline'],
                    mode='lines',
                    name='Midline (50)',
                    line=dict(color='#888888', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
            current_row += 1

        # Add Stochastic subplot
        if show_stochastic and stoch_data:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch_data['k_percent'],
                    mode='lines',
                    name='%K',
                    line=dict(color='#ff6600', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch_data['d_percent'],
                    mode='lines',
                    name='%D',
                    line=dict(color='#00aaff', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch_data['overbought'],
                    mode='lines',
                    name='Overbought (80)',
                    line=dict(color='#ff4444', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch_data['oversold'],
                    mode='lines',
                    name='Oversold (20)',
                    line=dict(color='#00ff88', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=stoch_data['midline'],
                    mode='lines',
                    name='Midline (50)',
                    line=dict(color='#888888', width=1, dash='dot'),
                    opacity=0.5
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=current_row, col=1)
            current_row += 1

        # Enhanced Volume trace
        if volume_data:
            volume_colors = []
            for i, (close, open_price, vol, avg_vol) in enumerate(zip(df['Close'], df['Open'],
                                                                      volume_data['volume'],
                                                                      volume_data['avg_volume'])):
                is_green_candle = close >= open_price
                volume_ratio = vol / avg_vol if avg_vol > 0 else 1

                if volume_ratio > 1.5:
                    color = '#00ff88' if is_green_candle else '#ff4444'
                elif volume_ratio > 1.2:
                    color = '#66ff99' if is_green_candle else '#ff6666'
                else:
                    color = '#88ff88' if is_green_candle else '#ff8888'

                volume_colors.append(color)

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=volume_data['volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_data['avg_volume'],
                    mode='lines',
                    name='Volume MA',
                    line=dict(color='#ffa500', width=2),
                    opacity=0.8
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="Volume", row=current_row, col=1)
            current_row += 1

            # On-Balance Volume (OBV)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_data['obv'],
                    mode='lines',
                    name='OBV',
                    line=dict(color='#9932cc', width=2)
                ),
                row=current_row, col=1
            )

            obv_ma = volume_data['obv'].rolling(window=20, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=obv_ma,
                    mode='lines',
                    name='OBV MA(20)',
                    line=dict(color='#ff6600', width=2, dash='dash'),
                    opacity=0.8
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="OBV", row=current_row, col=1)
        else:
            colors = ['#00ff88' if close >= open else '#ff4444'
                      for close, open in zip(df['Close'], df['Open'])]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=current_row, col=1
            )

            fig.update_yaxes(title_text="Volume", row=current_row, col=1)

        # Add pattern markers (NOT annotations) with clickable functionality
        if pattern_annotations:
            pattern_markers = []
            pattern_data = []

            for annotation in pattern_annotations:
                try:
                    # Extract pattern information from annotation
                    x_pos = annotation.get('x')
                    y_pos = annotation.get('y')
                    text = annotation.get('text', '')

                    if x_pos is not None and y_pos is not None:
                        # Get closing price at this date
                        try:
                            close_price = df.loc[x_pos, 'Close'] if x_pos in df.index else None
                            if close_price is None and isinstance(x_pos, str):
                                # Try to convert string date to datetime
                                x_datetime = pd.to_datetime(x_pos)
                                close_idx = df.index.get_indexer([x_datetime], method='nearest')[0]
                                if close_idx >= 0:
                                    close_price = df.iloc[close_idx]['Close']
                        except Exception as e:
                            close_price = None
                            logger.warning(f"Could not get close price for pattern at {x_pos}: {e}")

                        pattern_markers.append({
                            'x': x_pos,
                            'y': y_pos,
                            'close_price': close_price
                        })
                        pattern_data.append(text)

                except Exception as e:
                    logger.warning(f"Error processing pattern annotation: {e}")
                    continue

            if pattern_markers:
                # Add scatter plot for pattern markers
                fig.add_trace(
                    go.Scatter(
                        x=[m['x'] for m in pattern_markers],
                        y=[m['y'] for m in pattern_markers],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='gold',
                            line=dict(width=2, color='black')
                        ),
                        name='Patterns',
                        text=pattern_data,
                        customdata=[{
                            'pattern': text,
                            'close_price': m['close_price']
                        } for text, m in zip(pattern_data, pattern_markers)],
                        hovertemplate=(
                                '<b>Pattern:</b> %{text}<br>' +
                                '<b>Date:</b> %{x}<br>' +
                                '<b>Price:</b> $%{customdata.close_price:.2f}<br>' +
                                '<extra></extra>'
                        )
                    ),
                    row=1, col=1
                )

                logger.info(f"Added {len(pattern_markers)} pattern markers to chart")

        base_height = 600
        indicator_height = 150 * (subplot_count - 1)
        chart_height = base_height + indicator_height

        fig.update_layout(
            title=f'{ticker} Technical Analysis Dashboard - Enhanced with Volume & Pattern Detection',
            xaxis_rangeslider_visible=False,
            height=chart_height,
            template='plotly_dark',
            showlegend=True,
            hovermode='closest'
        )

        fig.update_xaxes(
            type='date',
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=30, label="30D", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(count=180, label="6M", step="day", stepmode="backward"),
                    dict(count=365, label="1Y", step="day", stepmode="backward"),
                    dict(step="all", label="MAX")
                ])
            )
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)

        return fig


def clear_main_content():
    """Clear all main content areas"""
    # Create empty containers to clear the display
    main_container = st.empty()
    return main_container


def main():
    """Main Streamlit application"""

    # Initialize session state for persistent data
    if 'data_provider' not in st.session_state:
        st.session_state.data_provider = DataProvider()

    # Initialize chart state
    if 'chart_generated' not in st.session_state:
        st.session_state.chart_generated = False

    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None

    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None

    if 'chart_figure' not in st.session_state:
        st.session_state.chart_figure = None

    if 'chart_insights' not in st.session_state:
        st.session_state.chart_insights = None

    if 'show_news_panel' not in st.session_state:
        st.session_state.show_news_panel = False

    # Flag to track if we need to clear content when generating new chart
    if 'clear_content' not in st.session_state:
        st.session_state.clear_content = False

    # Sidebar controls - GET TICKER FIRST
    st.sidebar.header("Chart Controls")

    # Ticker selection
    default_tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY']
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=default_tickers,
        index=0,
        help="Choose a stock ticker to analyze"
    )

    # Custom ticker input
    custom_ticker = st.sidebar.text_input(
        "Or enter custom ticker:",
        placeholder="e.g., NFLX",
        help="Enter any valid stock ticker"
    )

    if custom_ticker:
        ticker = custom_ticker.upper()

    # Check if ticker changed to determine when to clear previous chart
    ticker_changed = (ticker != st.session_state.current_ticker)

    # Clear previous chart data if ticker changed
    if ticker_changed and st.session_state.current_ticker is not None:
        st.session_state.chart_figure = None
        st.session_state.chart_data = None
        st.session_state.chart_insights = None
        st.session_state.chart_generated = False
        st.session_state.show_news_panel = False

    # Application header
    header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.title("Technical Analysis Dashboard")
        st.markdown("Interactive candlestick charts with technical indicators, volume analysis, and pattern detection")

    with header_col2:
        # News button visibility - show if we have a valid ticker and news is available
        if ticker and NEWS_AVAILABLE:
            st.write("")
            news_button_col1, news_button_col2 = st.columns([1, 1])

            with news_button_col1:
                if st.button("📰 News", type="secondary", help=f"Toggle news panel for {ticker}"):
                    st.session_state.show_news_panel = not st.session_state.show_news_panel

            with news_button_col2:
                if st.session_state.show_news_panel:
                    if st.button("✕ Close", type="secondary", help="Close news panel"):
                        st.session_state.show_news_panel = False

    # Display news panel without clearing main chart
    if st.session_state.show_news_panel and ticker and NEWS_AVAILABLE:
        st.markdown("---")
        with st.container():
            st.subheader(f"📰 Latest News for {ticker}")

            news_col1, news_col2, news_col3 = st.columns([1, 1, 1])

            with news_col1:
                max_articles = st.selectbox(
                    "Articles to show:",
                    [3, 5, 8, 10],
                    index=1,
                    key="main_news_count"
                )

            with news_col2:
                use_company_name = st.checkbox(
                    "Include company name",
                    value=True,
                    key="main_company_name"
                )

            with news_col3:
                if st.button("🔄 Refresh News", key="refresh_main_news"):
                    pass

            # Fetch and display news
            try:
                news_provider = NewsProvider()
                company_name = None
                if use_company_name:
                    with st.spinner("Getting company info..."):
                        company_name = news_provider.expand_company_name(ticker)

                query = news_provider.build_query(ticker, company_name, use_company_name)

                with st.spinner("Loading news..."):
                    articles = news_provider.fetch_news(
                        query=query,
                        max_articles=max_articles,
                        start_date=None,
                        end_date=None
                    )

                if articles:
                    st.success(f"Found {len(articles)} articles")

                    for i, article in enumerate(articles[:max_articles], 1):
                        with st.container():
                            article_col1, article_col2 = st.columns([3, 1])

                            with article_col1:
                                st.markdown(f"**{i}. [{article['title']}]({article['link']})**")
                                st.write(article['summary'])

                            with article_col2:
                                st.caption(f"**Source:** {article['source']}")
                                if 'published_date' in article:
                                    st.caption(f"**Date:** {article['published_date']}")

                        if i < len(articles[:max_articles]):
                            st.markdown("---")
                else:
                    st.warning("No articles found for the current ticker")

            except Exception as e:
                st.error(f"Error loading news: {str(e)}")
                st.info("Please check your NewsData.io API configuration")

        st.markdown("---")

    # Main content area - only show existing chart if not clearing content
    if not st.session_state.clear_content:
        # Display existing chart only if it matches current ticker
        if (st.session_state.chart_figure is not None and
                st.session_state.chart_data is not None and
                st.session_state.current_ticker == ticker):

            st.plotly_chart(st.session_state.chart_figure, use_container_width=True, config={'displayModeBar': True})

            if st.session_state.chart_insights is not None:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Market Summary")
                    for metric_data in st.session_state.chart_insights['market_summary']:
                        st.metric(**metric_data)

                with col2:
                    st.subheader("Technical Analysis")
                    for metric_data in st.session_state.chart_insights['technical_analysis']:
                        st.metric(**metric_data)

                with col3:
                    st.subheader("Volume & Pattern Analysis")
                    for metric_data in st.session_state.chart_insights['volume_pattern']:
                        st.metric(**metric_data)

                    if 'patterns' in st.session_state.chart_insights:
                        pattern_info = st.session_state.chart_insights['patterns']
                        if pattern_info:
                            st.write(pattern_info)

    # Interval selection
    intervals = {
        'Daily': '1d',
        'Weekly': '1w',
        'Monthly': '1mo',
        'Quarterly': '3mo',
        'Yearly': '1y'
    }

    interval_label = st.sidebar.selectbox(
        "Interval",
        options=list(intervals.keys()),
        index=0
    )
    interval = intervals[interval_label]

    # Date range selection
    st.sidebar.subheader("Date Range")

    preset_ranges = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365,
        '2Y': 730,
        '5Y': 1825
    }

    preset = st.sidebar.selectbox(
        "Quick Select",
        options=['Custom'] + list(preset_ranges.keys()),
        index=4
    )

    if preset != 'Custom':
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=preset_ranges[preset])
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=365),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                min_value=start_date,
                max_value=datetime.now().date()
            )

    # SMA parameters
    st.sidebar.subheader("Technical Indicators")
    sma1_period = st.sidebar.slider("SMA 1 Period", 5, 200, 20, help="Short-term SMA period")
    sma2_period = st.sidebar.slider("SMA 2 Period", 10, 400, 50, help="Long-term SMA period")

    # Advanced indicators
    if PATTERNS_AVAILABLE:
        show_macd = st.sidebar.checkbox(
            "Show MACD",
            value=False,
            help="Display MACD indicator"
        )

        if show_macd:
            with st.sidebar.expander("MACD Settings"):
                macd_fast = st.slider("Fast EMA", 5, 30, 12, help="Fast EMA period for MACD")
                macd_slow = st.slider("Slow EMA", 15, 50, 26, help="Slow EMA period for MACD")
                macd_signal = st.slider("Signal EMA", 5, 20, 9, help="Signal line EMA period")

        show_rsi = st.sidebar.checkbox(
            "Show RSI",
            value=False,
            help="Display RSI indicator with overbought/oversold levels"
        )

        if show_rsi:
            with st.sidebar.expander("RSI Settings"):
                rsi_period = st.slider("RSI Period", 5, 50, 14, help="RSI calculation period")

        show_stochastic = st.sidebar.checkbox(
            "Show Stochastic",
            value=False,
            help="Display Stochastic Oscillator with %K and %D lines"
        )

        if show_stochastic:
            with st.sidebar.expander("Stochastic Settings"):
                stoch_k_period = st.slider("K Period", 5, 30, 14, help="Period for %K calculation")
                stoch_d_period = st.slider("D Period", 1, 10, 3, help="Period for %D calculation")
                stoch_smooth = st.slider("Smooth K", 1, 10, 3, help="Smoothing period for %K")
    else:
        show_macd = False
        show_rsi = False
        show_stochastic = False

    calc_method = st.sidebar.selectbox(
        "Calculation Method",
        options=['pandas', 'excel'],
        index=0,
        help="Choose calculation method for indicators"
    )

    show_patterns = False

    if PATTERNS_AVAILABLE:
        st.sidebar.subheader("Pattern Detection")
        show_patterns = st.sidebar.checkbox(
            "Show Candlestick Patterns",
            value=False,
            help="Detect and display candlestick patterns as clickable markers on chart"
        )

    # Generate Chart Button
    if st.sidebar.button("Generate Chart", type="primary"):
        # Set flag to clear content
        st.session_state.clear_content = True

        # Clear the main content area
        st.empty()

        with st.spinner(f"Loading data for {ticker}..."):

            start_time = time.time()
            data = st.session_state.data_provider.get_ohlcv_data(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            fetch_time = time.time() - start_time

            if data is None or data.empty:
                st.error(f"No data available for ticker: {ticker}")
                st.info("Please check the ticker symbol and try again.")
                st.session_state.clear_content = False
                return

            # Store chart data in session state
            st.session_state.chart_data = data.copy()
            st.session_state.chart_generated = True
            st.session_state.current_ticker = ticker

            # Calculate indicators
            engine = TechnicalAnalysisEngine()
            sma1_data = engine.calculate_sma(data['Close'], sma1_period, calc_method)
            sma2_data = engine.calculate_sma(data['Close'], sma2_period, calc_method)

            macd_data = None
            rsi_data = None
            stoch_data = None
            volume_data = None

            if PATTERNS_AVAILABLE:
                indicators = TechnicalIndicators()

                if show_macd:
                    try:
                        macd_data = indicators.calculate_macd(
                            data['Close'],
                            fast_period=macd_fast,
                            slow_period=macd_slow,
                            signal_period=macd_signal
                        )
                    except Exception as e:
                        st.warning(f"MACD calculation error: {str(e)}")

                if show_rsi:
                    try:
                        rsi_data = indicators.calculate_rsi(
                            data['Close'],
                            period=rsi_period
                        )
                    except Exception as e:
                        st.warning(f"RSI calculation error: {str(e)}")

                if show_stochastic:
                    try:
                        stoch_data = indicators.calculate_stochastic(
                            data['High'],
                            data['Low'],
                            data['Close'],
                            k_period=stoch_k_period,
                            d_period=stoch_d_period,
                            smooth_k=stoch_smooth
                        )
                    except Exception as e:
                        st.warning(f"Stochastic calculation error: {str(e)}")

            try:
                volume_data = engine.calculate_volume_indicators(
                    data['Volume'],
                    data['Close'],
                    avg_period=20
                )
            except Exception as e:
                st.warning(f"Volume calculation error: {str(e)}")

            pattern_annotations = []
            pattern_summary = pd.DataFrame()

            if show_patterns and PATTERNS_AVAILABLE:
                with st.spinner("Detecting candlestick patterns..."):
                    try:
                        pattern_annotations, pattern_summary = detect_patterns_for_chart(
                            data, False  # Don't show descriptions as annotations, just markers
                        )
                        logger.info(f"Pattern detection completed: {len(pattern_annotations)} patterns found")
                    except Exception as e:
                        st.warning(f"Pattern detection error: {str(e)}")

            # Create chart
            with st.spinner("Generating chart..."):
                try:
                    renderer = ChartRenderer()
                    fig = renderer.create_candlestick_chart(
                        df=data,
                        ticker=ticker,
                        sma1=sma1_period,
                        sma2=sma2_period,
                        sma1_data=sma1_data,
                        sma2_data=sma2_data,
                        interval=interval,
                        pattern_annotations=pattern_annotations if show_patterns else None,
                        macd_data=macd_data,
                        show_macd=show_macd,
                        rsi_data=rsi_data,
                        show_rsi=show_rsi,
                        stoch_data=stoch_data,
                        show_stochastic=show_stochastic,
                        volume_data=volume_data
                    )

                    st.session_state.chart_figure = fig

                    # Reset clear content flag before displaying
                    st.session_state.clear_content = False

                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True},
                                    key=f"main_chart_{ticker}_{int(time.time())}")

                    chart_time = time.time() - start_time
                    st.sidebar.success(f"Chart generated in {chart_time:.2f}s")
                    st.sidebar.info(f"Data fetch: {fetch_time:.2f}s")

                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
                    logger.error(f"Chart generation failed: {str(e)}")
                    st.session_state.clear_content = False
                    return

            # Pattern marker information
            if show_patterns and pattern_annotations:
                st.info(
                    "💡 Click on the golden diamond markers on the chart to see pattern details with closing prices in the hover tooltip.")

            # Prepare insights data
            insights_data = {
                'market_summary': [],
                'technical_analysis': [],
                'volume_pattern': [],
                'patterns': None
            }

            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

            insights_data['market_summary'].append({
                'label': f"{ticker} Price",
                'value': f"${current_price:.2f}",
                'delta': f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
            })

            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            insights_data['market_summary'].append({
                'label': "Volume",
                'value': f"{current_volume:,.0f}",
                'delta': f"{volume_ratio:.1f}x avg" if volume_ratio > 0 else "No data"
            })

            current_sma1 = sma1_data.iloc[-1] if not sma1_data.empty else np.nan
            current_sma2 = sma2_data.iloc[-1] if not sma2_data.empty else np.nan
            previous_sma1 = sma1_data.iloc[-2] if len(sma1_data) > 1 else np.nan
            previous_sma2 = sma2_data.iloc[-2] if len(sma2_data) > 1 else np.nan

            sma_status = engine.get_sma_status(
                current_price, current_sma1, current_sma2,
                previous_price, previous_sma1, previous_sma2
            )

            status_colors = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
            insights_data['technical_analysis'].append({
                'label': "SMA Signal",
                'value': f"{status_colors.get(sma_status, '⚪')} {sma_status}"
            })

            if show_macd and macd_data is not None:
                current_macd = macd_data['macd'].iloc[-1]
                current_signal = macd_data['signal'].iloc[-1]
                current_hist = macd_data['histogram'].iloc[-1]
                prev_macd = macd_data['macd'].iloc[-2] if len(macd_data['macd']) > 1 else np.nan
                prev_signal = macd_data['signal'].iloc[-2] if len(macd_data['signal']) > 1 else np.nan
                prev_hist = macd_data['histogram'].iloc[-2] if len(macd_data['histogram']) > 1 else np.nan

                macd_status = engine.get_macd_status(
                    current_macd, current_signal, current_hist,
                    prev_macd, prev_signal, prev_hist
                )

                insights_data['technical_analysis'].append({
                    'label': "MACD Signal",
                    'value': f"{status_colors.get(macd_status, '⚪')} {macd_status}"
                })

            if show_rsi and rsi_data is not None:
                current_rsi = rsi_data['rsi'].iloc[-1]
                if current_rsi > 70:
                    rsi_status = "OVERBOUGHT"
                    rsi_color = "🔴"
                elif current_rsi < 30:
                    rsi_status = "OVERSOLD"
                    rsi_color = "🟢"
                else:
                    rsi_status = "NEUTRAL"
                    rsi_color = "🟡"

                insights_data['technical_analysis'].append({
                    'label': "RSI Signal",
                    'value': f"{rsi_color} {rsi_status}",
                    'delta': f"RSI: {current_rsi:.1f}"
                })

            if show_stochastic and stoch_data is not None:
                current_k = stoch_data['k_percent'].iloc[-1]
                current_d = stoch_data['d_percent'].iloc[-1]
                prev_k = stoch_data['k_percent'].iloc[-2] if len(stoch_data['k_percent']) > 1 else np.nan
                prev_d = stoch_data['d_percent'].iloc[-2] if len(stoch_data['d_percent']) > 1 else np.nan

                stoch_status = engine.get_stochastic_status(
                    current_k, current_d, prev_k, prev_d,
                    stoch_data['k_percent'], stoch_data['d_percent']
                )

                insights_data['technical_analysis'].append({
                    'label': "Stochastic Signal",
                    'value': f"{status_colors.get(stoch_status, '⚪')} {stoch_status}",
                    'delta': f"%K: {current_k:.1f}, %D: {current_d:.1f}"
                })

            if volume_data is not None:
                current_vol = volume_data['volume'].iloc[-1]
                avg_vol = volume_data['avg_volume'].iloc[-1]
                vol_price_change = volume_data['price_change_pct'].iloc[-1]

                volume_status = engine.get_volume_status(
                    current_vol, avg_vol, vol_price_change,
                    volume_data['volume'], data['Close']
                )

                volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0
                insights_data['volume_pattern'].append({
                    'label': "Volume Signal",
                    'value': f"{status_colors.get(volume_status, '⚪')} {volume_status}",
                    'delta': f"Ratio: {volume_ratio:.1f}x"
                })

                obv_current = volume_data['obv'].iloc[-1]
                obv_previous = volume_data['obv'].iloc[-5] if len(volume_data['obv']) > 5 else obv_current
                obv_trend = "UP" if obv_current > obv_previous else "DOWN" if obv_current < obv_previous else "FLAT"
                obv_color = "🟢" if obv_trend == "UP" else "🔴" if obv_trend == "DOWN" else "🟡"

                insights_data['volume_pattern'].append({
                    'label': "OBV Trend",
                    'value': f"{obv_color} {obv_trend}",
                    'delta': f"OBV: {obv_current:,.0f}"
                })

            if show_patterns and not pattern_summary.empty:
                pattern_text = f"Found **{len(pattern_summary)}** pattern types:\n"
                for _, row in pattern_summary.iterrows():
                    pattern_text += f"{row['Symbol']} **{row['Pattern']}** ({row['Count']})\n"
                insights_data['patterns'] = pattern_text
            elif show_patterns:
                insights_data['patterns'] = "No patterns detected in the selected timeframe."
            else:
                insights_data['patterns'] = "Enable pattern detection in the sidebar to see clickable pattern markers."

            st.session_state.chart_insights = insights_data

            # Display insights
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Market Summary")
                for metric_data in insights_data['market_summary']:
                    st.metric(**metric_data)

            with col2:
                st.subheader("Technical Analysis")
                for metric_data in insights_data['technical_analysis']:
                    st.metric(**metric_data)

            with col3:
                st.subheader("Volume & Pattern Analysis")
                for metric_data in insights_data['volume_pattern']:
                    st.metric(**metric_data)

                if insights_data['patterns']:
                    st.write(insights_data['patterns'])

            # Raw data
            if st.expander("Raw Data", expanded=False):
                st.dataframe(
                    data.tail(50).round(2),
                    use_container_width=True
                )

            # Technical indicators
            if st.expander("Technical Indicators", expanded=False):
                indicator_df = pd.DataFrame(index=data.tail(10).index)
                indicator_df[f'SMA({sma1_period})'] = sma1_data.tail(10).round(2)
                indicator_df[f'SMA({sma2_period})'] = sma2_data.tail(10).round(2)

                if show_macd and macd_data:
                    indicator_df['MACD'] = macd_data['macd'].tail(10).round(4)
                    indicator_df['Signal'] = macd_data['signal'].tail(10).round(4)
                    indicator_df['Histogram'] = macd_data['histogram'].tail(10).round(4)

                if show_rsi and rsi_data:
                    indicator_df['RSI'] = rsi_data['rsi'].tail(10).round(2)

                if show_stochastic and stoch_data:
                    indicator_df['%K'] = stoch_data['k_percent'].tail(10).round(2)
                    indicator_df['%D'] = stoch_data['d_percent'].tail(10).round(2)

                if volume_data:
                    indicator_df['Volume'] = volume_data['volume'].tail(10).round(0)
                    indicator_df['Avg Volume'] = volume_data['avg_volume'].tail(10).round(0)
                    indicator_df['Volume Ratio'] = volume_data['volume_ratio'].tail(10).round(2)
                    indicator_df['OBV'] = volume_data['obv'].tail(10).round(0)

                st.dataframe(indicator_df, use_container_width=True)

            # Pattern summary
            if show_patterns and not pattern_summary.empty and st.expander("Pattern Summary", expanded=False):
                st.dataframe(
                    pattern_summary[['Pattern', 'Symbol', 'Count', 'Description', 'Most Recent']],
                    use_container_width=True,
                    hide_index=True
                )

    # Information messages
    if not st.session_state.chart_generated and NEWS_AVAILABLE and not st.session_state.clear_content:
        st.info("Generate a chart to access full functionality, or use the news button above for any ticker")
    elif not NEWS_AVAILABLE and not st.session_state.clear_content:
        st.warning("News functionality requires news_app.py and NEWSDATA_API_KEY environment variable")

    # About section
    if not st.session_state.clear_content:
        st.markdown("---")
        with st.expander("About This Dashboard", expanded=False):
            st.markdown("""
            ### Enhanced Technical Analysis Dashboard

            This dashboard provides comprehensive technical analysis tools with advanced signal analysis.

            **Key Features:**
            - Interactive candlestick charts with enhanced volume analysis
            - Simple Moving Averages (SMA) with buy/sell/neutral signals
            - MACD (Moving Average Convergence Divergence) with signal analysis
            - RSI (Relative Strength Index) with overbought/oversold detection
            - Stochastic Oscillator with advanced signal logic
            - Volume analysis with price correlation signals
            - On-Balance Volume (OBV) trend analysis
            - Candlestick pattern detection with clickable markers showing pattern name and closing price
            - Multi-timeframe analysis (Daily, Weekly, Monthly, etc.)
            - Integrated news functionality

            **Recent Improvements:**
            - Chart State Persistence: Charts remain visible when accessing news
            - Toggle News Panel: News can be opened/closed without affecting chart display
            - Session State Management: All chart data and insights are preserved
            - Enhanced Error Handling: Better handling of missing dependencies
            - Fixed News Button Visibility: Shows immediately when ticker is selected
            - Clear Content on New Chart: Previous charts are completely cleared when generating new ones
            - Clickable Pattern Markers: Click on golden diamond markers to see pattern details with closing prices

            **Pattern Detection:**
            - When enabled, candlestick patterns appear as golden diamond markers on the chart
            - Hover over markers to see pattern name, date, and closing price
            - No text annotations cluttering the chart - just clean, clickable markers

            **Disclaimer:**
            This tool is for educational and research purposes only. Not financial advice.
            Always consult with qualified financial advisors before making investment decisions.
            """)


if __name__ == "__main__":
    main()