"""
Candlestick Pattern Detection Module and Technical Indicators
Identifies various candlestick patterns and calculates technical indicators including RSI and Stochastic Oscillator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicator calculations"""

    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[
        str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            data: Price series (typically close prices)
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)

        Returns:
            Dictionary containing MACD line, Signal line, and Histogram
        """
        # Validate input data
        if data.empty or len(data) < max(fast_period, slow_period, signal_period):
            logger.warning("Insufficient data for MACD calculation")
            return {
                'macd': pd.Series(dtype=float, index=data.index),
                'signal': pd.Series(dtype=float, index=data.index),
                'histogram': pd.Series(dtype=float, index=data.index)
            }

        # Calculate EMAs
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # MACD Histogram
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate RSI (Relative Strength Index) with signal lines

        Args:
            data: Price series (typically close prices)
            period: RSI period (default 14)

        Returns:
            Dictionary containing RSI, overbought line (70), oversold line (30)
        """
        # Validate input data
        if data.empty or len(data) < period + 1:
            logger.warning("Insufficient data for RSI calculation")
            return {
                'rsi': pd.Series(dtype=float, index=data.index),
                'overbought': pd.Series(70, index=data.index, name='Overbought'),
                'oversold': pd.Series(30, index=data.index, name='Oversold'),
                'midline': pd.Series(50, index=data.index, name='Midline')
            }

        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using EMA
        avg_gains = gains.ewm(com=period - 1, adjust=False).mean()
        avg_losses = losses.ewm(com=period - 1, adjust=False).mean()

        # Avoid division by zero
        avg_losses = avg_losses.replace(0, np.finfo(float).eps)

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Create signal lines
        overbought_line = pd.Series(70, index=data.index, name='Overbought')
        oversold_line = pd.Series(30, index=data.index, name='Oversold')
        midline = pd.Series(50, index=data.index, name='Midline')

        return {
            'rsi': rsi,
            'overbought': overbought_line,
            'oversold': oversold_line,
            'midline': midline
        }

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                             k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator with %K and %D lines

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation (default 14)
            d_period: Period for %D calculation (default 3)
            smooth_k: Smoothing period for %K (default 3)

        Returns:
            Dictionary containing %K, %D, overbought line (80), oversold line (20)
        """
        # Validate input data
        min_length = max(k_period, smooth_k, d_period)
        if high.empty or low.empty or close.empty or len(close) < min_length:
            logger.warning("Insufficient data for Stochastic calculation")
            return {
                'k_percent': pd.Series(dtype=float, index=close.index),
                'd_percent': pd.Series(dtype=float, index=close.index),
                'overbought': pd.Series(80, index=close.index, name='Overbought'),
                'oversold': pd.Series(20, index=close.index, name='Oversold'),
                'midline': pd.Series(50, index=close.index, name='Midline')
            }

        # Calculate lowest low and highest high over the specified period
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        # Avoid division by zero
        range_hilo = highest_high - lowest_low
        range_hilo = range_hilo.replace(0, np.finfo(float).eps)

        # Calculate raw %K
        k_raw = 100 * ((close - lowest_low) / range_hilo)

        # Smooth %K
        k_percent = k_raw.rolling(window=smooth_k, min_periods=1).mean()

        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

        # Create signal lines
        overbought_line = pd.Series(80, index=close.index, name='Overbought')
        oversold_line = pd.Series(20, index=close.index, name='Oversold')
        midline = pd.Series(50, index=close.index, name='Midline')

        return {
            'k_percent': k_percent,
            'd_percent': d_percent,
            'overbought': overbought_line,
            'oversold': oversold_line,
            'midline': midline
        }

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average

        Args:
            data: Price series
            period: EMA period

        Returns:
            EMA series
        """
        if data.empty or len(data) < period:
            logger.warning("Insufficient data for EMA calculation")
            return pd.Series(dtype=float, index=data.index)

        return data.ewm(span=period, adjust=False).mean()


class CandlestickPatterns:
    """Candlestick pattern detection and analysis"""

    # Pattern symbols for chart display
    PATTERN_SYMBOLS = {
        'hammer': '🔨',
        'inverted_hammer': '🔨⬆️',
        'bullish_engulfing': '🟢⬆️',
        'bearish_engulfing': '🔴⬇️',
        'morning_star': '⭐🌅',
        'evening_star': '⭐🌙',
        'piercing_line': '⚡⬆️',
        'dark_cloud_cover': '☁️⬇️',
        'shooting_star': '🌟⬇️',
        'hanging_man': '🪢⬇️',
        'three_white_soldiers': '💥⬆️',
        'three_black_crows': '🦅⬇️',
        'doji': '✖️'
    }

    # Pattern descriptions
    PATTERN_DESCRIPTIONS = {
        'hammer': 'Hammer - Bullish reversal pattern',
        'inverted_hammer': 'Inverted Hammer - Potential bullish reversal',
        'bullish_engulfing': 'Bullish Engulfing - Strong bullish reversal',
        'bearish_engulfing': 'Bearish Engulfing - Strong bearish reversal',
        'morning_star': 'Morning Star - Bullish reversal pattern',
        'evening_star': 'Evening Star - Bearish reversal pattern',
        'piercing_line': 'Piercing Line - Bullish reversal',
        'dark_cloud_cover': 'Dark Cloud Cover - Bearish reversal',
        'shooting_star': 'Shooting Star - Bearish reversal',
        'hanging_man': 'Hanging Man - Bearish reversal',
        'three_white_soldiers': 'Three White Soldiers - Strong bullish trend',
        'three_black_crows': 'Three Black Crows - Strong bearish trend',
        'doji': 'Doji - Indecision/reversal signal'
    }

    # Detailed pattern explanations for hover
    PATTERN_EXPLANATIONS = {
        'hammer': 'A bullish reversal pattern with a small body at the top and a long lower shadow, indicating buying pressure after a decline.',
        'inverted_hammer': 'A potential bullish reversal with a small body at the bottom and a long upper shadow, suggesting buyers may be gaining control.',
        'bullish_engulfing': 'A strong bullish reversal where a large green candle completely engulfs the previous red candle, indicating strong buying pressure.',
        'bearish_engulfing': 'A strong bearish reversal where a large red candle completely engulfs the previous green candle, indicating strong selling pressure.',
        'morning_star': 'A three-candle bullish reversal pattern consisting of a bearish candle, a small-bodied star, and a bullish candle.',
        'evening_star': 'A three-candle bearish reversal pattern consisting of a bullish candle, a small-bodied star, and a bearish candle.',
        'piercing_line': 'A bullish reversal pattern where a green candle opens below the previous red candle\'s low and closes above its midpoint.',
        'dark_cloud_cover': 'A bearish reversal pattern where a red candle opens above the previous green candle\'s high and closes below its midpoint.',
        'shooting_star': 'A bearish reversal pattern with a small body and long upper shadow, indicating selling pressure after an uptrend.',
        'hanging_man': 'A bearish reversal pattern similar to a hammer but appearing after an uptrend, suggesting potential weakness.',
        'three_white_soldiers': 'A strong bullish pattern of three consecutive green candles with progressively higher closes.',
        'three_black_crows': 'A strong bearish pattern of three consecutive red candles with progressively lower closes.',
        'doji': 'A reversal or indecision pattern where open and close prices are nearly equal, indicating market uncertainty.'
    }

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame

        Args:
            df: DataFrame with Open, High, Low, Close columns
        """
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        self.df = df.copy()
        self.patterns = {}

        # Calculate helper metrics
        self._calculate_helpers()

    def _calculate_helpers(self):
        """Calculate helper metrics for pattern detection"""
        self.df['body'] = abs(self.df['Close'] - self.df['Open'])
        self.df['upper_shadow'] = self.df['High'] - np.maximum(self.df['Open'], self.df['Close'])
        self.df['lower_shadow'] = np.minimum(self.df['Open'], self.df['Close']) - self.df['Low']
        self.df['range'] = self.df['High'] - self.df['Low']
        self.df['is_bullish'] = self.df['Close'] > self.df['Open']
        self.df['is_bearish'] = self.df['Close'] < self.df['Open']

        # Average body size for relative comparisons (20-period rolling)
        # Add minimum values to prevent division by zero
        self.df['avg_body'] = self.df['body'].rolling(window=20, min_periods=1).mean()
        self.df['avg_range'] = self.df['range'].rolling(window=20, min_periods=1).mean()

        # Ensure no zero values that could cause division issues
        self.df['avg_body'] = self.df['avg_body'].replace(0, self.df['body'].mean() or 0.01)
        self.df['avg_range'] = self.df['avg_range'].replace(0, self.df['range'].mean() or 0.01)

        # Handle any remaining NaN or infinite values
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(0)

    def detect_hammer(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect Hammer pattern
        - Small body at upper end of range
        - Lower shadow at least 2x body size
        - Minimal upper shadow
        """
        # Avoid division by zero
        body_safe = self.df['body'].replace(0, np.finfo(float).eps)

        conditions = (
                (self.df['lower_shadow'] >= threshold * body_safe) &
                (self.df['upper_shadow'] <= 0.1 * body_safe) &
                (self.df['body'] <= 0.3 * self.df['range']) &
                (self.df['body'] > 0) &  # Avoid doji
                (self.df['range'] > 0)  # Valid range
        )
        return conditions.fillna(False)

    def detect_inverted_hammer(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect Inverted Hammer pattern
        - Small body at lower end of range
        - Upper shadow at least 2x body size
        - Minimal lower shadow
        """
        # Avoid division by zero
        body_safe = self.df['body'].replace(0, np.finfo(float).eps)

        conditions = (
                (self.df['upper_shadow'] >= threshold * body_safe) &
                (self.df['lower_shadow'] <= 0.1 * body_safe) &
                (self.df['body'] <= 0.3 * self.df['range']) &
                (self.df['body'] > 0) &  # Avoid doji
                (self.df['range'] > 0)  # Valid range
        )
        return conditions.fillna(False)

    def detect_bullish_engulfing(self) -> pd.Series:
        """
        Detect Bullish Engulfing pattern
        - Two candles: first bearish, second bullish
        - Second candle's body completely engulfs first candle's body
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(1, len(self.df)):
            try:
                if (self.df['is_bearish'].iloc[i - 1] and  # Previous candle is bearish
                        self.df['is_bullish'].iloc[i] and  # Current candle is bullish
                        self.df['Open'].iloc[i] < self.df['Close'].iloc[i - 1] and  # Open below prev close
                        self.df['Close'].iloc[i] > self.df['Open'].iloc[i - 1]):  # Close above prev open
                    conditions.iloc[i] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_bearish_engulfing(self) -> pd.Series:
        """
        Detect Bearish Engulfing pattern
        - Two candles: first bullish, second bearish
        - Second candle's body completely engulfs first candle's body
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(1, len(self.df)):
            try:
                if (self.df['is_bullish'].iloc[i - 1] and  # Previous candle is bullish
                        self.df['is_bearish'].iloc[i] and  # Current candle is bearish
                        self.df['Open'].iloc[i] > self.df['Close'].iloc[i - 1] and  # Open above prev close
                        self.df['Close'].iloc[i] < self.df['Open'].iloc[i - 1]):  # Close below prev open
                    conditions.iloc[i] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_morning_star(self) -> pd.Series:
        """
        Detect Morning Star pattern
        - Three candles: bearish, small body (doji/spinning top), bullish
        - Gap down and gap up
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(2, len(self.df)):
            try:
                candle1 = i - 2  # First candle
                candle2 = i - 1  # Middle candle (star)
                candle3 = i  # Third candle

                avg_body_safe = max(self.df['avg_body'].iloc[candle2], 0.01)

                if (self.df['is_bearish'].iloc[candle1] and  # First candle bearish
                        self.df['body'].iloc[candle2] < 0.3 * avg_body_safe and  # Small middle body
                        self.df['is_bullish'].iloc[candle3] and  # Third candle bullish
                        self.df['Close'].iloc[candle2] < self.df['Close'].iloc[candle1] and  # Gap down
                        self.df['Close'].iloc[candle3] > (
                                self.df['Open'].iloc[candle1] + self.df['Close'].iloc[candle1]) / 2):  # Recovery
                    conditions.iloc[candle3] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_evening_star(self) -> pd.Series:
        """
        Detect Evening Star pattern
        - Three candles: bullish, small body (doji/spinning top), bearish
        - Gap up and gap down
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(2, len(self.df)):
            try:
                candle1 = i - 2  # First candle
                candle2 = i - 1  # Middle candle (star)
                candle3 = i  # Third candle

                avg_body_safe = max(self.df['avg_body'].iloc[candle2], 0.01)

                if (self.df['is_bullish'].iloc[candle1] and  # First candle bullish
                        self.df['body'].iloc[candle2] < 0.3 * avg_body_safe and  # Small middle body
                        self.df['is_bearish'].iloc[candle3] and  # Third candle bearish
                        self.df['Close'].iloc[candle2] > self.df['Close'].iloc[candle1] and  # Gap up
                        self.df['Close'].iloc[candle3] < (
                                self.df['Open'].iloc[candle1] + self.df['Close'].iloc[candle1]) / 2):  # Decline
                    conditions.iloc[candle3] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_piercing_line(self) -> pd.Series:
        """
        Detect Piercing Line pattern
        - Two candles: bearish, then bullish
        - Second candle opens below first's low, closes above first's midpoint
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(1, len(self.df)):
            try:
                if (self.df['is_bearish'].iloc[i - 1] and  # First candle bearish
                        self.df['is_bullish'].iloc[i] and  # Second candle bullish
                        self.df['Open'].iloc[i] < self.df['Low'].iloc[i - 1] and  # Opens below prev low
                        self.df['Close'].iloc[i] > (
                                self.df['Open'].iloc[i - 1] + self.df['Close'].iloc[
                            i - 1]) / 2):  # Closes above midpoint
                    conditions.iloc[i] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_dark_cloud_cover(self) -> pd.Series:
        """
        Detect Dark Cloud Cover pattern
        - Two candles: bullish, then bearish
        - Second candle opens above first's high, closes below first's midpoint
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(1, len(self.df)):
            try:
                if (self.df['is_bullish'].iloc[i - 1] and  # First candle bullish
                        self.df['is_bearish'].iloc[i] and  # Second candle bearish
                        self.df['Open'].iloc[i] > self.df['High'].iloc[i - 1] and  # Opens above prev high
                        self.df['Close'].iloc[i] < (
                                self.df['Open'].iloc[i - 1] + self.df['Close'].iloc[
                            i - 1]) / 2):  # Closes below midpoint
                    conditions.iloc[i] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_shooting_star(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect Shooting Star pattern
        - Small body at lower end of range
        - Long upper shadow (at least 2x body)
        - Minimal lower shadow
        - Appears after uptrend
        """
        # Avoid division by zero
        body_safe = self.df['body'].replace(0, np.finfo(float).eps)

        # Simple uptrend check using 5-period price comparison
        uptrend_check = self.df['Close'] > self.df['Close'].shift(5)
        uptrend_check = uptrend_check.fillna(False)

        conditions = (
                (self.df['upper_shadow'] >= threshold * body_safe) &
                (self.df['lower_shadow'] <= 0.1 * body_safe) &
                (self.df['body'] <= 0.3 * self.df['range']) &
                (self.df['body'] > 0) &  # Avoid doji
                (self.df['range'] > 0) &  # Valid range
                uptrend_check  # After uptrend (simple check)
        )
        return conditions.fillna(False)

    def detect_hanging_man(self, threshold: float = 2.0) -> pd.Series:
        """
        Detect Hanging Man pattern
        - Small body at upper end of range
        - Long lower shadow (at least 2x body)
        - Minimal upper shadow
        - Appears after uptrend
        """
        # Avoid division by zero
        body_safe = self.df['body'].replace(0, np.finfo(float).eps)

        # Simple uptrend check using 5-period price comparison
        uptrend_check = self.df['Close'] > self.df['Close'].shift(5)
        uptrend_check = uptrend_check.fillna(False)

        conditions = (
                (self.df['lower_shadow'] >= threshold * body_safe) &
                (self.df['upper_shadow'] <= 0.1 * body_safe) &
                (self.df['body'] <= 0.3 * self.df['range']) &
                (self.df['body'] > 0) &  # Avoid doji
                (self.df['range'] > 0) &  # Valid range
                uptrend_check  # After uptrend (simple check)
        )
        return conditions.fillna(False)

    def detect_three_white_soldiers(self) -> pd.Series:
        """
        Detect Three White Soldiers pattern
        - Three consecutive bullish candles
        - Each opens within previous body
        - Each closes higher than previous
        - Bodies of similar size
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(2, len(self.df)):
            try:
                candle1 = i - 2
                candle2 = i - 1
                candle3 = i

                avg_body_safe = [
                    max(self.df['avg_body'].iloc[candle1], 0.01),
                    max(self.df['avg_body'].iloc[candle2], 0.01),
                    max(self.df['avg_body'].iloc[candle3], 0.01)
                ]

                if (self.df['is_bullish'].iloc[candle1] and
                        self.df['is_bullish'].iloc[candle2] and
                        self.df['is_bullish'].iloc[candle3] and
                        # Each opens within previous body
                        self.df['Open'].iloc[candle2] > self.df['Open'].iloc[candle1] and
                        self.df['Open'].iloc[candle2] < self.df['Close'].iloc[candle1] and
                        self.df['Open'].iloc[candle3] > self.df['Open'].iloc[candle2] and
                        self.df['Open'].iloc[candle3] < self.df['Close'].iloc[candle2] and
                        # Each closes higher
                        self.df['Close'].iloc[candle2] > self.df['Close'].iloc[candle1] and
                        self.df['Close'].iloc[candle3] > self.df['Close'].iloc[candle2] and
                        # Bodies of reasonable size
                        self.df['body'].iloc[candle1] > 0.5 * avg_body_safe[0] and
                        self.df['body'].iloc[candle2] > 0.5 * avg_body_safe[1] and
                        self.df['body'].iloc[candle3] > 0.5 * avg_body_safe[2]):
                    conditions.iloc[candle3] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_three_black_crows(self) -> pd.Series:
        """
        Detect Three Black Crows pattern
        - Three consecutive bearish candles
        - Each opens within previous body
        - Each closes lower than previous
        - Bodies of similar size
        """
        conditions = pd.Series(False, index=self.df.index)

        for i in range(2, len(self.df)):
            try:
                candle1 = i - 2
                candle2 = i - 1
                candle3 = i

                avg_body_safe = [
                    max(self.df['avg_body'].iloc[candle1], 0.01),
                    max(self.df['avg_body'].iloc[candle2], 0.01),
                    max(self.df['avg_body'].iloc[candle3], 0.01)
                ]

                if (self.df['is_bearish'].iloc[candle1] and
                        self.df['is_bearish'].iloc[candle2] and
                        self.df['is_bearish'].iloc[candle3] and
                        # Each opens within previous body
                        self.df['Open'].iloc[candle2] < self.df['Open'].iloc[candle1] and
                        self.df['Open'].iloc[candle2] > self.df['Close'].iloc[candle1] and
                        self.df['Open'].iloc[candle3] < self.df['Open'].iloc[candle2] and
                        self.df['Open'].iloc[candle3] > self.df['Close'].iloc[candle2] and
                        # Each closes lower
                        self.df['Close'].iloc[candle2] < self.df['Close'].iloc[candle1] and
                        self.df['Close'].iloc[candle3] < self.df['Close'].iloc[candle2] and
                        # Bodies of reasonable size
                        self.df['body'].iloc[candle1] > 0.5 * avg_body_safe[0] and
                        self.df['body'].iloc[candle2] > 0.5 * avg_body_safe[1] and
                        self.df['body'].iloc[candle3] > 0.5 * avg_body_safe[2]):
                    conditions.iloc[candle3] = True
            except (IndexError, KeyError):
                continue

        return conditions

    def detect_doji(self, threshold: float = 0.1) -> pd.Series:
        """
        Detect Doji pattern
        - Open and close prices are nearly equal
        - Body is very small relative to range
        """
        # Avoid division by zero
        range_safe = self.df['range'].replace(0, np.finfo(float).eps)
        avg_range_safe = self.df['avg_range'].replace(0, np.finfo(float).eps)

        conditions = (
                (self.df['body'] <= threshold * range_safe) &
                (self.df['range'] > 0.5 * avg_range_safe)  # Meaningful range
        )
        return conditions.fillna(False)

    def detect_all_patterns(self) -> Dict[str, pd.Series]:
        """
        Detect all candlestick patterns

        Returns:
            Dictionary with pattern names as keys and boolean Series as values
        """
        patterns = {
            'hammer': self.detect_hammer(),
            'inverted_hammer': self.detect_inverted_hammer(),
            'bullish_engulfing': self.detect_bullish_engulfing(),
            'bearish_engulfing': self.detect_bearish_engulfing(),
            'morning_star': self.detect_morning_star(),
            'evening_star': self.detect_evening_star(),
            'piercing_line': self.detect_piercing_line(),
            'dark_cloud_cover': self.detect_dark_cloud_cover(),
            'shooting_star': self.detect_shooting_star(),
            'hanging_man': self.detect_hanging_man(),
            'three_white_soldiers': self.detect_three_white_soldiers(),
            'three_black_crows': self.detect_three_black_crows(),
            'doji': self.detect_doji()
        }

        self.patterns = patterns
        return patterns

    def get_pattern_summary(self) -> pd.DataFrame:
        """
        Get summary of detected patterns

        Returns:
            DataFrame with pattern counts and recent occurrences
        """
        if not self.patterns:
            self.detect_all_patterns()

        summary_data = []

        for pattern_name, pattern_series in self.patterns.items():
            count = pattern_series.sum()
            if count > 0:
                # Get most recent occurrence
                recent_dates = self.df.index[pattern_series].tolist()
                most_recent = recent_dates[-1] if recent_dates else None

                summary_data.append({
                    'Pattern': pattern_name.replace('_', ' ').title(),
                    'Symbol': self.PATTERN_SYMBOLS.get(pattern_name, '❓'),
                    'Count': int(count),
                    'Description': self.PATTERN_DESCRIPTIONS.get(pattern_name, 'Unknown pattern'),
                    'Most Recent': most_recent.strftime('%Y-%m-%d') if most_recent else 'N/A'
                })

        return pd.DataFrame(summary_data)

    def get_pattern_annotations_with_hover(self, show_descriptions: bool = False) -> List[Dict]:
        """
        Get pattern annotations for chart display with proper text formatting

        Args:
            show_descriptions: Whether to show pattern descriptions

        Returns:
            List of annotation dictionaries for Plotly (only valid properties)
        """
        if not self.patterns:
            self.detect_all_patterns()

        annotations = []

        for pattern_name, pattern_series in self.patterns.items():
            pattern_indices = self.df.index[pattern_series]

            for date in pattern_indices:
                symbol = self.PATTERN_SYMBOLS.get(pattern_name, '❓')
                description = self.PATTERN_DESCRIPTIONS.get(pattern_name, 'Unknown')
                explanation = self.PATTERN_EXPLANATIONS.get(pattern_name, 'No detailed explanation available')

                # Get price data for this date
                try:
                    price_data = self.df.loc[date]
                    high_price = price_data['High']
                    close_price = price_data['Close']
                    open_price = price_data['Open']
                    low_price = price_data['Low']
                except (KeyError, IndexError):
                    logger.warning(f"Could not find price data for date {date}")
                    continue

                # Determine bullish/bearish nature with more specific logic
                if 'doji' in pattern_name:
                    pattern_type = "Neutral"
                elif any(keyword in pattern_name for keyword in ['bullish', 'morning', 'hammer', 'piercing', 'white']):
                    pattern_type = "Bullish"
                elif any(keyword in pattern_name for keyword in
                         ['bearish', 'evening', 'shooting', 'hanging', 'dark', 'black']):
                    pattern_type = "Bearish"
                else:
                    # Fallback to candle color
                    pattern_type = "Bullish" if close_price >= open_price else "Bearish"

                # Format the pattern name for display
                pattern_display_name = pattern_name.replace('_', ' ').title()

                # Create comprehensive display text with proper formatting
                if show_descriptions:
                    text = (
                        f"<b>{symbol} {pattern_display_name}</b><br>"
                        f"<i>{pattern_type} Pattern</i><br>"
                        f"Price: ${close_price:.2f}<br>"
                        f"Range: ${low_price:.2f} - ${high_price:.2f}<br>"
                        f"Date: {date.strftime('%Y-%m-%d')}"
                    )
                else:
                    text = (
                        f"<b>{symbol}</b><br>"
                        f"{pattern_display_name}<br>"
                        f"<i>{pattern_type}</i><br>"
                        f"${close_price:.2f}"
                    )

                # Determine annotation color based on pattern type
                if pattern_type == "Bullish":
                    arrow_color = '#00ff88'  # Green
                    border_color = '#00ff88'
                elif pattern_type == "Bearish":
                    arrow_color = '#ff4444'  # Red
                    border_color = '#ff4444'
                else:
                    arrow_color = '#ffa500'  # Orange for neutral
                    border_color = '#ffa500'

                # Only use valid Plotly annotation properties with enhanced formatting
                annotation = {
                    'x': date,
                    'y': high_price * 1.03,  # Slightly above the high
                    'text': text,
                    'showarrow': True,
                    'arrowhead': 2,
                    'arrowsize': 1.5,
                    'arrowwidth': 2,
                    'arrowcolor': arrow_color,
                    'font': dict(size=11, color='white', family='Arial'),
                    'bgcolor': 'rgba(0,0,0,0.8)',
                    'bordercolor': border_color,
                    'borderwidth': 2,
                    'borderpad': 4,
                    'opacity': 0.9
                }
                annotations.append(annotation)

        return annotations

    def get_pattern_annotations(self, show_descriptions: bool = False) -> List[Dict]:
        """
        Legacy method for backward compatibility
        """
        return self.get_pattern_annotations_with_hover(show_descriptions)

    def debug_patterns(self) -> Dict[str, int]:
        """
        Debug method to check pattern detection counts

        Returns:
            Dictionary with pattern names and their counts
        """
        if not self.patterns:
            self.detect_all_patterns()

        pattern_counts = {}
        for pattern_name, pattern_series in self.patterns.items():
            count = pattern_series.sum()
            if count > 0:
                pattern_counts[pattern_name] = int(count)
                logger.info(f"Found {count} instances of {pattern_name}")

        return pattern_counts


def detect_patterns_for_chart(df: pd.DataFrame, show_descriptions: bool = False) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Convenience function to detect patterns and return chart annotations

    Args:
        df: OHLCV DataFrame
        show_descriptions: Whether to show pattern descriptions on chart

    Returns:
        Tuple of (annotations for chart, pattern summary DataFrame)
    """
    try:
        detector = CandlestickPatterns(df)

        # Debug: Log pattern detection results
        pattern_counts = detector.debug_patterns()
        if pattern_counts:
            logger.info(f"Pattern detection summary: {pattern_counts}")
        else:
            logger.info("No patterns detected in the data")

        annotations = detector.get_pattern_annotations_with_hover(show_descriptions)
        summary = detector.get_pattern_summary()

        logger.info(f"Created {len(annotations)} annotations for chart display")

        return annotations, summary

    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return [], pd.DataFrame()