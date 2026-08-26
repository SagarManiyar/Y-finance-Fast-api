# Smart Wealth - Issue Analysis & Step-by-Step Fix Guide

## Date: 2026-08-24

---

## Problem Statement

1. When fetching 5 years of data, the last date returned is July 31, 2026 - no data after that
2. Version compatibility issues between packages

---

## Environment

| Component   | Installed Version | Latest Available | Status                    |
|-------------|-------------------|------------------|---------------------------|
| Python      | 3.14.2            | -                | Very new, bleeding edge   |
| yfinance    | 1.3.0             | **1.6.0**        | **3 major versions behind** |
| pandas      | 3.0.3             | -                | Has breaking changes vs 2.x |
| numpy       | 2.4.4             | -                | OK                        |
| plotly       | 5.24.1            | -                | OK                        |
| streamlit   | 1.57.0            | -                | OK                        |

---

## Root Cause Analysis

### Issue 1: Data Stopping at July 31

**Direct yfinance test result (2026-08-24):** All tickers (AAPL, TSLA, MSFT) return data correctly up to August 21, 2026 (last trading day) when called directly from the command line with the same 5-year date range.

**Most likely causes:**

1. **yfinance 1.3.0 intermittent API issues** - Yahoo Finance frequently changes their API. yfinance 1.3.0 is 3 major versions behind (1.6.0 is latest). Older versions have known issues with data retrieval, cookie handling, and API endpoint changes. Versions 1.4.0 through 1.6.0 contain critical bug fixes for data fetching.

2. **Streamlit session caching** - The `DataProvider` class in `main.py` caches data in memory (300-second TTL). If you kept the same Streamlit browser session open since late July, the session state could hold stale data. Refreshing the browser page (full page reload, not just clicking "Generate Chart") clears this.

3. **csv_candlestick.py design flaw** - This file creates a `stocks_data.csv` file once and never updates it. If this CSV was created when July 31 was the last trading day, it will forever show data only up to July 31.

### Issue 2: Version Errors (CONFIRMED BUGS)

**Bug A: `patterns.py` line 284 - CRASHES in pandas 3.0**

```python
# CURRENT (BROKEN):
self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(0)
```

Error: `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'`

The `method` parameter was removed from `fillna()` in pandas 3.0. Must use `.bfill()` instead.

**Bug B: `main.py` lines 230-236 - CRASHES in pandas 3.0**

```python
# CURRENT (BROKEN for monthly, quarterly, yearly):
freq_map = {
    '1d': 'D',
    '1w': 'W',
    '1mo': 'M',    # ValueError: 'M' is no longer supported
    '3mo': 'Q',    # ValueError: 'Q' is no longer supported
    '1y': 'Y'      # ValueError: 'Y' is no longer supported
}
```

Error: `ValueError: 'M' is no longer supported for offsets. Please use 'ME' instead.`

In pandas 3.0, the frequency aliases changed:
- `'M'` -> `'ME'` (Month End)
- `'Q'` -> `'QE'` (Quarter End)
- `'Y'` -> `'YE'` (Year End)

This means **Monthly, Quarterly, and Yearly intervals are completely broken** and will crash the app.

---

## Step-by-Step Fix Guide

### Step 1: Upgrade yfinance (Fixes data cutoff issue)

Run this command in your project's virtual environment:

```bash
pip install --upgrade yfinance
```

This upgrades from 1.3.0 to 1.6.0 which includes:
- Fixed Yahoo Finance API compatibility
- Improved cookie handling
- Better data retrieval reliability
- Bug fixes for data truncation issues

### Step 2: Fix `patterns.py` line 284 (pandas 3.0 compatibility)

**File:** `patterns.py`
**Line:** 284

Change FROM:
```python
self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(0)
```

Change TO:
```python
self.df = self.df.replace([np.inf, -np.inf], np.nan).bfill().fillna(0)
```

### Step 3: Fix `main.py` lines 230-236 (pandas 3.0 compatibility)

**File:** `main.py`
**Lines:** 230-236

Change FROM:
```python
freq_map = {
    '1d': 'D',
    '1w': 'W',
    '1mo': 'M',
    '3mo': 'Q',
    '1y': 'Y'
}
```

Change TO:
```python
freq_map = {
    '1d': 'D',
    '1w': 'W',
    '1mo': 'ME',
    '3mo': 'QE',
    '1y': 'YE'
}
```

### Step 4: Fix `csv_candlestick.py` stale CSV issue (Design fix)

**File:** `csv_candlestick.py`
**Lines:** 22-35

The current code creates the CSV file once and never updates it:

```python
# CURRENT (creates once, never refreshes):
if not os.path.exists(csv_file):
    # ... fetch and save data
```

Two options to fix:

**Option A: Always refresh data** (remove the file check):
```python
all_data = []
for ticker in companies:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=today, interval="1d")
    if not df.empty:
        df = df.reset_index()
        df["Company"] = ticker
        all_data.append(df)

if all_data:
    final_df = pd.concat(all_data)
    final_df.to_csv(csv_file, index=False)
```

**Option B: Refresh if CSV is older than 1 day:**
```python
import time

csv_is_stale = False
if os.path.exists(csv_file):
    file_age_hours = (time.time() - os.path.getmtime(csv_file)) / 3600
    csv_is_stale = file_age_hours > 24

if not os.path.exists(csv_file) or csv_is_stale:
    all_data = []
    for ticker in companies:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=today, interval="1d")
        if not df.empty:
            df = df.reset_index()
            df["Company"] = ticker
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data)
        final_df.to_csv(csv_file, index=False)
```

### Step 5: Clear Streamlit session state (Immediate fix for stale data)

After making code changes, do a hard refresh of the Streamlit app:
1. Stop the running Streamlit server (Ctrl+C in terminal)
2. Delete any `stocks_data.csv` file if it exists
3. Restart: `streamlit run main.py`

---

## Verification After Fixes

After applying all fixes, verify:

1. **Data range test:** Select 5Y preset, generate chart for AAPL - the last data point should be the most recent trading day
2. **Monthly interval test:** Switch to Monthly interval and generate chart - should not crash
3. **Pattern detection test:** Enable "Show Candlestick Patterns" checkbox and generate chart - should not crash
4. **csv_candlestick.py test:** Run `streamlit run csv_candlestick.py` - should show recent data

---

## Summary of All Changes Required

| File | Line(s) | Change | Reason |
|------|---------|--------|--------|
| (pip upgrade) | - | `pip install --upgrade yfinance` | Data cutoff fix + API compatibility |
| `patterns.py` | 284 | `.fillna(method='bfill')` -> `.bfill()` | pandas 3.0 removed `method` param |
| `main.py` | 233-235 | `'M'`->`'ME'`, `'Q'`->`'QE'`, `'Y'`->`'YE'` | pandas 3.0 removed old aliases |
| `csv_candlestick.py` | 23 | Remove/update `if not os.path.exists` check | Stale CSV never refreshes |

---

## Issue 3: Server Error "A type extension with name pandas.period already defined" (2026-08-25/26)

### Symptom

Hitting the deployed `/history-5y` endpoint on `https://y-finance.smart-copilots.ai/` (e.g. via Postman) returned:

```json
{"detail":"Server error: A type extension with name pandas.period already defined"}
```

### Root Cause: Two supervisor programs running the same app on different ports, ALB pointed at the stale one

The server (`i-07dfd7d8914b4d74f`) runs **two separate supervisor-managed processes** for the same `app.py`:

| Supervisor program   | Launch command                                              | Port | Notes |
|-----------------------|--------------------------------------------------------------|------|-------|
| `fastapi`             | `python /var/www/y-finance-fast-api/app.py`                  | 5000 | Hardcoded port in `app.py`'s `__main__` block. **This is what the ALB target group actually forwards to.** |
| `y-finance-fast-api`   | `uvicorn app:app --host 0.0.0.0 --port 8000`                  | 8000 | Not reachable from the public internet at all. |

The ALB (`y-finance-smart-copilots-ai-alb`) → target group `Y-Finance-Fast-Api-tg` → forwards `HTTP:80`/`HTTPS:443` traffic to **port 5000** on this instance.

During earlier debugging, code changes (pyarrow monkey-patch, `safe_to_parquet` helper, etc.) were deployed to `app.py` on disk, and the `y-finance-fast-api` (port 8000) supervisor program was restarted to pick them up. However, the `fastapi` program (port 5000) — the one actually receiving public traffic — was **never restarted**, so it kept running the old, unpatched code in memory indefinitely. This is why:
- Direct testing against `http://127.0.0.1:8000/...` always succeeded (correct process).
- Every request through the public URL / Postman always failed (stale process on port 5000).
- The target group showed `Unhealthy` with `404` health-check failures, since the stale process was in a broken state.

### Fix

Restart the correct supervisor program — **`fastapi`**, not `y-finance-fast-api`:

```bash
sudo /opt/supervisor-venv/bin/supervisorctl -c /etc/supervisor/supervisord.conf restart fastapi
```

(`supervisorctl` is not on PATH by default on this box; supervisord runs from `/opt/supervisor-venv/`.)

Verify:

```bash
curl -s -X POST https://y-finance.smart-copilots.ai/history-5y -H "Content-Type: application/json" -d '{"tickers": "AAPL"}'
```

Should return `"status":"success"` with parquet data.

### Important operational note for future deploys

**Any code change to `app.py` must be picked up by restarting the `fastapi` program (port 5000)** — that is the process actually serving `y-finance.smart-copilots.ai`. Restarting `y-finance-fast-api` (port 8000) alone has no effect on production traffic and will cause the exact same "works locally, fails via public URL" confusion again.

Longer-term cleanup to consider (not yet done — requires deliberate decision, since it touches production routing):
- Confirm whether the `y-finance-fast-api` (port 8000, uvicorn) supervisor program is still needed at all, or if it's a leftover from an earlier deployment attempt.
- If not needed, remove it to avoid two processes running the same app in different states.
- Alternatively, standardize on one launch method (uvicorn) and point the ALB target group at that single port.
