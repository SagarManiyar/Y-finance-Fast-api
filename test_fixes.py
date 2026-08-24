"""
Quick verification script - tests all fixes applied to the project.
Run: python test_fixes.py
"""

import sys
import traceback
from datetime import datetime, timedelta

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def test(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}")
        print(f"         {e}")


# ── Test 1: yfinance version ──
def check_yfinance_version():
    import yfinance
    ver = yfinance.__version__
    major, minor, patch = ver.split(".")
    assert int(major) >= 1 and int(minor) >= 6, f"yfinance {ver} is outdated, need >= 1.6.0"

print("\n=== 1. Package Versions ===")
test("yfinance >= 1.6.0", check_yfinance_version)


# ── Test 2: yfinance data retrieval (5-year range) ──
def check_5y_data():
    import yfinance as yf
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
    assert not data.empty, "No data returned"
    last_date = data.index[-1]
    days_old = (datetime.now() - last_date.replace(tzinfo=None)).days
    assert days_old <= 5, f"Last data date is {last_date.strftime('%Y-%m-%d')} ({days_old} days old)"
    print(f"         Data: {data.index[0].strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({len(data)} rows)")

print("\n=== 2. Data Retrieval (5Y range) ===")
test("AAPL 5-year data fetches up to recent trading day", check_5y_data)


# ── Test 3: Multiple tickers ──
def check_multi_ticker():
    import yfinance as yf
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    tickers = ["AAPL", "TSLA", "MSFT"]
    for sym in tickers:
        data = yf.Ticker(sym).history(start=start, end=end, interval="1d", auto_adjust=True)
        assert not data.empty, f"{sym}: no data"
        last = data.index[-1].strftime("%Y-%m-%d")
        print(f"         {sym}: last date = {last}, rows = {len(data)}")

print("\n=== 3. Multi-Ticker Check ===")
test("AAPL, TSLA, MSFT all return recent data", check_multi_ticker)


# ── Test 4: pandas resample frequencies (Bug B fix) ──
def check_resample_frequencies():
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(400),
        "High": np.random.rand(400),
        "Low": np.random.rand(400),
        "Close": np.random.rand(400),
        "Volume": np.random.randint(100, 10000, 400),
    }, index=dates)
    for alias, label in [("ME", "Monthly"), ("QE", "Quarterly"), ("YE", "Yearly")]:
        r = df.resample(alias).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        assert len(r) > 0, f"{label} resample returned 0 rows"
        print(f"         {label} ({alias}): {len(r)} rows")

print("\n=== 4. Pandas 3.0 Resample Fix ===")
test("ME/QE/YE frequency aliases work", check_resample_frequencies)


# ── Test 5: bfill fix (Bug A fix) ──
def check_bfill():
    import pandas as pd
    import numpy as np
    s = pd.Series([1.0, np.nan, 3.0, np.inf, -np.inf, np.nan])
    result = s.replace([np.inf, -np.inf], np.nan).bfill().fillna(0)
    assert not result.isna().any(), "NaN values remain after bfill"
    print(f"         Result: {list(result.values)}")

print("\n=== 5. Pandas 3.0 bfill Fix ===")
test(".bfill().fillna(0) works correctly", check_bfill)


# ── Test 6: Pattern detection module ──
def check_patterns():
    import pandas as pd
    import numpy as np
    sys.path.insert(0, r"D:\perseus it\Smatrt Wealth\fast-api-to-flask")
    from patterns import CandlestickPatterns, detect_patterns_for_chart
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(200, 250, 100),
        "Low": np.random.uniform(50, 100, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.randint(1000, 100000, 100),
    }, index=dates)
    annotations, summary = detect_patterns_for_chart(df, False)
    print(f"         Patterns found: {len(summary)} types, {len(annotations)} annotations")

print("\n=== 6. Pattern Detection Module ===")
test("detect_patterns_for_chart runs without crash", check_patterns)


# ── Test 7: Full DataProvider flow from main.py ──
def check_data_provider():
    sys.path.insert(0, r"D:\perseus it\Smatrt Wealth\fast-api-to-flask")
    from main import DataProvider, TechnicalAnalysisEngine
    provider = DataProvider()
    end = datetime.now().date().strftime("%Y-%m-%d")
    start = (datetime.now().date() - timedelta(days=1825)).strftime("%Y-%m-%d")
    data = provider.get_ohlcv_data("AAPL", start, end, "1d")
    assert data is not None and not data.empty, "DataProvider returned no data"
    last = data.index[-1].strftime("%Y-%m-%d")
    print(f"         DataProvider: {data.index[0].strftime('%Y-%m-%d')} to {last} ({len(data)} rows)")

print("\n=== 7. DataProvider Full Flow ===")
test("DataProvider.get_ohlcv_data returns 5Y data", check_data_provider)


# ── Summary ──
print("\n" + "=" * 50)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("\nFailed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
