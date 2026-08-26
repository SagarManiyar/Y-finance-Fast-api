# ============================================================
# app.py  —  FastAPI version (migrated from Flask)
#
# WHAT CHANGED vs Flask version:
#   - Flask  → FastAPI + Uvicorn
#   - flask_cors CORS()  → FastAPI CORSMiddleware
#   - @app.route(path, methods=["POST"])  → @app.post(path)
#   - request.get_json()  → Request body parsed via Pydantic or raw Request
#   - jsonify({...})  → return dict  (FastAPI auto-serialises to JSON)
#   - return jsonify(...), 404  → raise HTTPException(status_code=404, ...)
#   - send_file(buffer, ...)  → StreamingResponse(buffer, ...)
#   - if __name__ == "__main__": app.run()  → uvicorn.run()
#
# NOTHING ELSE CHANGED:
#   - All business logic, helper functions, indicator calculations,
#     pattern detection, threading, boto3 calls — 100% identical.
#   - All URL paths are identical (Laravel calls require zero changes).
#   - Response JSON shapes are identical.
# ============================================================

import io
import threading

# Patch pyarrow BEFORE pandas is imported.
# pandas 3.0 + pyarrow 25 has a bug where Arrow extension types
# (pandas.period, pandas.interval, etc.) get registered twice,
# crashing with "A type extension with name ... already defined".
# This patch makes duplicate registrations a silent no-op.
import pyarrow as pa
_original_register_ext = pa.register_extension_type
def _safe_register_ext(ext_type):
    try:
        _original_register_ext(ext_type)
    except (pa.ArrowKeyError, KeyError, Exception):
        pass
pa.register_extension_type = _safe_register_ext

import uvicorn
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import boto3
import json
import base64

_parquet_lock = threading.Lock()


def safe_to_parquet(df):
    """Thread-safe parquet conversion."""
    buf = io.BytesIO()
    with _parquet_lock:
        df.to_parquet(buf, engine='pyarrow', index=False)
    buf.seek(0)
    return buf


# ===========================
# App Setup  (replaces Flask(__name__) + CORS(app))
# ===========================
app = FastAPI()

# Allow all origins — same behaviour as Flask-CORS default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lambda_client = boto3.client('lambda', region_name='us-east-1')


# ===========================
# Technical Indicator Helpers  — UNCHANGED
# ===========================

def compute_sma(series, window):
    return series.rolling(window=window).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def detect_patterns(hist):
    patterns = {}
    opens  = hist["Open"].values
    closes = hist["Close"].values
    highs  = hist["High"].values
    lows   = hist["Low"].values
    dates  = [idx.strftime("%Y-%m-%d") for idx in hist.index]

    for i in range(1, len(hist)):
        found      = []
        body       = closes[i] - opens[i]
        prev_body  = closes[i - 1] - opens[i - 1]
        body_size  = abs(body)
        total_range = highs[i] - lows[i]

        if total_range > 0 and body_size / total_range < 0.1:
            found.append("Doji")

        if total_range > 0:
            lower_wick = min(opens[i], closes[i]) - lows[i]
            upper_wick = highs[i] - max(opens[i], closes[i])
            if lower_wick > 2 * body_size and upper_wick < body_size:
                found.append("Hammer")

        if (prev_body < 0 and body > 0
                and closes[i] > opens[i - 1]
                and opens[i] < closes[i - 1]):
            found.append("Bullish Engulfing")

        if (prev_body > 0 and body < 0
                and closes[i] < opens[i - 1]
                and opens[i] > closes[i - 1]):
            found.append("Bearish Engulfing")

        if i >= 2:
            prev2_body = closes[i - 2] - opens[i - 2]
            if (prev2_body < 0
                    and abs(prev_body) < abs(prev2_body) * 0.3
                    and body > 0
                    and closes[i] > (opens[i - 2] + closes[i - 2]) / 2):
                found.append("Morning Star")

        if i >= 2:
            prev2_body = closes[i - 2] - opens[i - 2]
            if (prev2_body > 0
                    and abs(prev_body) < abs(prev2_body) * 0.3
                    and body < 0
                    and closes[i] < (opens[i - 2] + closes[i - 2]) / 2):
                found.append("Evening Star")

        if found:
            patterns[dates[i]] = found

    return patterns


def series_to_list(series, dates):
    return [
        {"date": d, "value": round(v, 4)}
        for d, v in zip(dates, series.tolist())
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]


# ===========================
# Normalize tickers — UNCHANGED
# ===========================
def normalize_tickers_input(data):
    if not data:
        return []
    if "ticker" in data and not data.get("tickers"):
        raw = data.get("ticker")
    else:
        raw = data.get("tickers")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        return [str(p).strip().upper() for p in raw if str(p).strip()]
    else:
        return [str(raw).strip().upper()]


# ===========================
# /company endpoint
# CHANGED: @app.route → @app.post | request.get_json() → await request.json()
#          jsonify → return dict | jsonify(...),400 → raise HTTPException
# ===========================
@app.post("/company")
async def stock(request: Request):
    # FastAPI: read raw JSON body from request
    data   = await request.json()
    ticker = data.get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker name is required")

    start_date = data.get("start_date")
    end_date   = data.get("end_date")
    stock_obj  = yf.Ticker(ticker)

    try:
        if start_date and end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            hist   = stock_obj.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"))
        elif start_date:
            hist = stock_obj.history(start=start_date)
        else:
            hist = stock_obj.history(period="1mo")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    company_info = getattr(stock_obj, "info", {}) or {}
    fi = {
        k: (float(v) if isinstance(v, (int, float)) else str(v))
        for k, v in getattr(stock_obj, "fast_info", {}).items()
    }
    fi["recommendationsSummary"] = company_info.get("recommendationKey", "N/A")
    fi["longDescription"]        = company_info.get("longBusinessSummary", "N/A")
    fi["companyName"]            = company_info.get("longName", "N/A")
    fi["country"]                = company_info.get("country", "N/A")
    fi["sector"]                 = company_info.get("sector", "N/A")
    fi["industry"]               = company_info.get("industry", "N/A")

    current_price = company_info.get("currentPrice")
    eps_ttm       = company_info.get("trailingEps")
    fi["EPS_TTM"]  = eps_ttm if eps_ttm else "N/A"
    fi["PE_Ratio"] = (
        round(current_price / eps_ttm, 2)
        if (eps_ttm and eps_ttm != 0 and current_price)
        else "N/A"
    )

    dates        = [idx.strftime("%Y-%m-%d") for idx in hist.index]
    history_data = []
    for index, row in hist.iterrows():
        history_data.append({
            "Date":         index.strftime("%Y-%m-%d"),
            "Open":         float(row["Open"]),
            "High":         float(row["High"]),
            "Low":          float(row["Low"]),
            "Close":        float(row["Close"]),
            "Volume":       int(row["Volume"]) if "Volume" in row else 0,
            "Dividends":    float(row.get("Dividends", 0)),
            "Stock Splits": float(row.get("Stock Splits", 0))
        })

    response = {"fast_info": fi, "history": history_data}
    close    = hist["Close"]
    high     = hist["High"]
    low      = hist["Low"]
    volume   = hist["Volume"]

    sma1_period = data.get("sma1")
    sma2_period = data.get("sma2")
    if sma1_period:
        response["sma1"] = {"period": int(sma1_period), "data": series_to_list(compute_sma(close, int(sma1_period)), dates)}
    if sma2_period:
        response["sma2"] = {"period": int(sma2_period), "data": series_to_list(compute_sma(close, int(sma2_period)), dates)}

    macd_config = data.get("macd", {})
    if macd_config and macd_config.get("enabled"):
        fast_p, slow_p, signal_p = int(macd_config.get("fast", 12)), int(macd_config.get("slow", 26)), int(macd_config.get("signal", 9))
        macd_line, signal_line, histogram = compute_macd(close, fast_p, slow_p, signal_p)
        response["macd"] = {"settings": {"fast": fast_p, "slow": slow_p, "signal": signal_p}, "macd": series_to_list(macd_line, dates), "signal": series_to_list(signal_line, dates), "histogram": series_to_list(histogram, dates)}

    rsi_config = data.get("rsi", {})
    if rsi_config and rsi_config.get("enabled"):
        rsi_period = int(rsi_config.get("period", 14))
        response["rsi"] = {"period": rsi_period, "data": series_to_list(compute_rsi(close, rsi_period), dates)}

    obv_config = data.get("obv", {})
    if obv_config and obv_config.get("enabled"):
        response["obv"] = {"data": series_to_list(compute_obv(close, volume), dates)}

    stoch_config = data.get("stoch", {})
    if stoch_config and stoch_config.get("enabled"):
        k_period, d_period = int(stoch_config.get("k_period", 14)), int(stoch_config.get("d_period", 3))
        k, d = compute_stochastic(high, low, close, k_period, d_period)
        response["stoch"] = {"settings": {"k_period": k_period, "d_period": d_period}, "k": series_to_list(k, dates), "d": series_to_list(d, dates)}

    if data.get("patterns"):
        response["patterns"] = detect_patterns(hist)

    return response


# ===========================
# /companies endpoint
# ===========================
def fetch_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="11d")
        if hist.empty:
            return {"error": f"No data found for '{ticker}'"}
        hist = hist.tail(10)
        info = stock.info or {}

        current_price              = info.get("currentPrice")
        eps_ttm                    = info.get("trailingEps")
        yesterday_close            = float(hist.iloc[-1]["Close"])
        day_before_yesterday_close = float(hist.iloc[-2]["Close"]) if len(hist) >= 2 else None

        pe_ratio = round(current_price / eps_ttm, 2) if (eps_ttm and eps_ttm != 0 and current_price) else "N/A"

        company_info = {
            "companyName": info.get("longName") or info.get("shortName"),
            "country":     info.get("country"),
            "currency":    info.get("currency"),
            "currentPrice": current_price,
            "yesterdayClose": yesterday_close,
            "dayBeforeYesterdayClose": day_before_yesterday_close,
            "dayHigh":   info.get("dayHigh"),
            "dayLow":    info.get("dayLow"),
            "marketCap": info.get("marketCap"),
            "EPS_TTM":   eps_ttm if eps_ttm else "N/A",
            "PE_Ratio":  pe_ratio,
        }

        line_data = [
            {"Date": index.strftime("%Y-%m-%d"), "Close": float(row["Close"]),
             "High": float(row["High"]), "Low": float(row["Low"]),
             "Open": float(row["Open"]), "Volume": int(row["Volume"]),
             "Stock Splits": float(row.get("Stock Splits", 0)),
             "Dividends": float(row.get("Dividends", 0))}
            for index, row in hist.iterrows()
        ]

        return {"info": company_info, "lineData": line_data}

    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {str(e)}"}


@app.post("/companies")
async def companies(request: Request):
    data    = await request.json()
    tickers = normalize_tickers_input(data)
    if not tickers:
        raise HTTPException(status_code=400, detail="Tickers are required")

    results     = {}
    max_workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_data, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except Exception as e:
                results[t] = {"error": str(e)}

    return {"requested": tickers, "results": results}


# ===========================
# Yesterday data helpers — UNCHANGED
# ===========================
def fetch_yesterday_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="5d")
        if hist.empty:
            return {"error": f"No data found for '{ticker}'"}

        today_str     = datetime.now().strftime("%Y-%m-%d")
        last_row_date = hist.index[-1].strftime("%Y-%m-%d")

        if last_row_date == today_str and len(hist) > 1:
            row         = hist.iloc[-2]
            target_date = hist.index[-2]
        else:
            row         = hist.iloc[-1]
            target_date = hist.index[-1]

        return {
            "Date":         target_date.strftime("%Y-%m-%d"),
            "Open":         float(row["Open"]),
            "High":         float(row["High"]),
            "Low":          float(row["Low"]),
            "Close":        float(row["Close"]),
            "Volume":       int(row["Volume"]),
            "Dividends":    float(row.get("Dividends", 0)),
            "Stock Splits": float(row.get("Stock Splits", 0)),
            "Ticker":       ticker,
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {str(e)}"}


def fetch_yesterday_overview_data(ticker):
    try:
        stock        = yf.Ticker(ticker)
        company_info = getattr(stock, "info", {}) or {}
        fast_info    = {}
        try:
            fast_info = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                         for k, v in getattr(stock, "fast_info", {}).items()}
        except Exception:
            fast_info = {}

        hist = stock.history(period="5d")
        if hist.empty:
            return {"error": f"No data found for '{ticker}'"}

        today_str     = datetime.now().strftime("%Y-%m-%d")
        last_row_date = hist.index[-1].strftime("%Y-%m-%d")

        if last_row_date == today_str and len(hist) > 1:
            yesterday_row  = hist.iloc[-2]
            yesterday_date = hist.index[-2]
        else:
            yesterday_row  = hist.iloc[-1]
            yesterday_date = hist.index[-1]

        current_price = company_info.get("currentPrice")
        eps_ttm       = company_info.get("trailingEps")

        overview_data = {
            "EPS_TTM": eps_ttm if eps_ttm else "N/A",
            "PE_Ratio": (round(current_price / eps_ttm, 2) if (eps_ttm and eps_ttm != 0 and current_price) else "N/A"),
            "companyName": company_info.get("longName", company_info.get("shortName", "N/A")),
            "country":     company_info.get("country", "N/A"),
            "currency":    company_info.get("currency", "N/A"),
            "dayHigh":     float(yesterday_row["High"]) if "High" in yesterday_row else fast_info.get("dayHigh", "N/A"),
            "dayLow":      float(yesterday_row["Low"])  if "Low"  in yesterday_row else fast_info.get("dayLow",  "N/A"),
            "exchange":    fast_info.get("exchange", "N/A"),
            "fiftyDayAverage": fast_info.get("fiftyDayAverage", "N/A"),
            "industry":    company_info.get("industry", "N/A"),
            "lastPrice":   float(yesterday_row["Close"])  if "Close"  in yesterday_row else fast_info.get("lastPrice",  "N/A"),
            "lastVolume":  float(yesterday_row["Volume"]) if "Volume" in yesterday_row else 0.0,
            "longDescription": company_info.get("longBusinessSummary", "N/A"),
            "marketCap":   company_info.get("marketCap", "None"),
            "open":        float(yesterday_row["Open"]) if "Open" in yesterday_row else fast_info.get("open", "N/A"),
            "previousClose": fast_info.get("regularMarketPreviousClose", fast_info.get("previousClose", "N/A")),
            "quoteType":   fast_info.get("quoteType", "N/A"),
            "recommendationsSummary": company_info.get("recommendationKey", "N/A"),
            "regularMarketPreviousClose": fast_info.get("regularMarketPreviousClose", "N/A"),
            "sector":      company_info.get("sector", "N/A"),
            "shares":      company_info.get("sharesOutstanding", "None"),
            "tenDayAverageVolume":     fast_info.get("tenDayAverageVolume", "N/A"),
            "threeMonthAverageVolume": fast_info.get("threeMonthAverageVolume", "N/A"),
            "timezone":    fast_info.get("timezone", "N/A"),
            "twoHundredDayAverage": fast_info.get("twoHundredDayAverage", "N/A"),
            "yearChange":  fast_info.get("yearChange", "N/A"),
            "yearHigh":    fast_info.get("yearHigh", "N/A"),
            "yearLow":     fast_info.get("yearLow", "N/A"),
            "asOfDate":    yesterday_date.strftime("%Y-%m-%d"),
        }

        for key, value in overview_data.items():
            if value not in ["N/A", "None"] and not isinstance(value, (int, float, str)):
                try:
                    overview_data[key] = float(value)
                except (ValueError, TypeError):
                    pass

        return overview_data

    except Exception as e:
        return {"error": f"Failed to fetch overview data for '{ticker}': {str(e)}"}


# ===========================
# /overview-data-for-each-ticker endpoint
# ===========================
@app.post("/overview-data-for-each-ticker")
async def yesterday_individual_overview(request: Request):
    try:
        data    = await request.json()
        tickers = normalize_tickers_input(data)
        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        ticker_files = []
        max_workers  = min(8, len(tickers))
        current_date = datetime.now().strftime("%Y-%m-%d")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_yesterday_overview_data, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                res    = future.result()

                if "error" not in res:
                    res["ticker"] = ticker
                    df = pd.DataFrame([res])

                    column_order = [
                        "ticker", "companyName", "asOfDate", "lastPrice", "open", "dayHigh", "dayLow",
                        "previousClose", "regularMarketPreviousClose", "volume", "lastVolume",
                        "marketCap", "shares", "EPS_TTM", "PE_Ratio", "currency", "exchange",
                        "sector", "industry", "country", "quoteType", "timezone",
                        "fiftyDayAverage", "twoHundredDayAverage", "yearHigh", "yearLow", "yearChange",
                        "tenDayAverageVolume", "threeMonthAverageVolume",
                        "recommendationsSummary", "longDescription"
                    ]
                    df = df[[col for col in column_order if col in df.columns]]

                    buffer = safe_to_parquet(df)
                    file_content   = buffer.read()
                    base64_content = base64.b64encode(file_content).decode('utf-8')

                    ticker_files.append({
                        "ticker":     ticker,
                        "date":       current_date,
                        "data_type":  "overview",
                        "period":     "1d",
                        "content":    base64_content,
                        "size_bytes": len(file_content),
                        "as_of_date": res.get("asOfDate", current_date),
                    })

        if not ticker_files:
            raise HTTPException(status_code=404, detail="No valid data found")

        return {
            "status":    "success",
            "count":     len(ticker_files),
            "data_type": "overview",
            "files":     ticker_files,
            "message":   f"Successfully retrieved overview data for {len(ticker_files)} tickers",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===========================
# /yesterday endpoint — Parquet bulk file
# CHANGED: send_file → StreamingResponse
# ===========================
@app.post("/yesterday")
async def yesterday(request: Request):
    try:
        data    = await request.json()
        tickers = normalize_tickers_input(data)

        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        results_list = []
        max_workers  = min(8, len(tickers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_yesterday_data, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                t   = future_to_ticker[future]
                res = future.result()
                if "error" not in res:
                    res["Ticker"] = t
                    results_list.append(res)

        if not results_list:
            raise HTTPException(status_code=404, detail="No valid data found to create file")

        df     = pd.DataFrame(results_list)
        buffer = safe_to_parquet(df)

        filename = f"yesterday_stocks_{datetime.now().strftime('%Y%m%d')}.parquet"

        # StreamingResponse replaces Flask's send_file
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===========================
# /yesterday-individual endpoint
# ===========================
@app.post("/yesterday-individual")
async def yesterday_individual(request: Request):
    try:
        data    = await request.json()
        tickers = normalize_tickers_input(data)

        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        ticker_files = []
        max_workers  = min(8, len(tickers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_yesterday_data, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                res    = future.result()

                if "error" not in res:
                    res["Ticker"] = ticker
                    df = pd.DataFrame([res])

                    buffer = safe_to_parquet(df)
                    file_content   = buffer.read()
                    base64_content = base64.b64encode(file_content).decode('utf-8')

                    ticker_files.append({
                        "ticker":     ticker,
                        "date":       datetime.now().strftime("%Y-%m-%d"),
                        "data_type":  "technical",
                        "period":     "5y",
                        "content":    base64_content,
                        "size_bytes": len(file_content)
                    })

        if not ticker_files:
            raise HTTPException(status_code=404, detail="No valid data found")

        return {
            "status":    "success",
            "count":     len(ticker_files),
            "files":     ticker_files,
            "data_type": "technical"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ============================================================
# fetch_10day_technical_data helper — UNCHANGED
# ============================================================
def fetch_10day_technical_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="90d")

        if hist.empty:
            return {"error": f"No data found for '{ticker}'"}

        today_str = datetime.now().strftime("%Y-%m-%d")
        if hist.index[-1].strftime("%Y-%m-%d") == today_str:
            hist = hist.iloc[:-1]

        if len(hist) < 2:
            return {"error": f"Insufficient history for '{ticker}'"}

        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]

        hist["PreviousClose"] = close.shift(1)
        hist["SMA20"] = compute_sma(close, 20)
        hist["SMA50"] = compute_sma(close, 50)

        sma20       = hist["SMA20"]
        above       = close > sma20
        below       = close < sma20
        three_above = above & above.shift(1).fillna(False) & above.shift(2).fillna(False)
        three_below = below & below.shift(1).fillna(False) & below.shift(2).fillna(False)

        sma_conditions = [three_above, three_below, above, below]
        sma_labels     = ["Strong Buy", "Strong Sell", "Buy", "Sell"]
        sma_values     = [2, -2, 1, -1]

        hist["SMA_Signal"]       = np.select(sma_conditions, sma_labels,  default="Neutral")
        hist["SMA_Signal_Value"] = np.select(sma_conditions, sma_values,  default=0).astype(float)
        hist["SMA_Rec_Score"]    = hist["SMA_Signal_Value"]

        sma_nan = sma20.isna()
        hist.loc[sma_nan, "SMA_Signal"]       = "Neutral"
        hist.loc[sma_nan, "SMA_Signal_Value"] = 0.0
        hist.loc[sma_nan, "SMA_Rec_Score"]    = 0.0

        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()

        hist["RSI_Gain"]           = avg_gain
        hist["RSI_Loss"]           = avg_loss
        hist["RSI_Gain_Loss_Value"] = avg_gain - avg_loss

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        hist["RSI"] = rsi

        rsi_conditions = [
            (rsi >= 50) & (rsi < 70),
            (rsi >= 30) & (rsi < 50),
            (rsi >= 70),
            (rsi < 30),
        ]
        hist["RSI_Signal"] = np.select(rsi_conditions, ["Strong Buy", "Buy", "Strong Sell", "Sell"], default="Neutral")
        hist.loc[rsi.isna(), "RSI_Signal"] = "Neutral"

        hist["Volume_SMA_10"] = volume.rolling(10).mean()

        macd_line, macd_signal, macd_hist = compute_macd(close, 12, 26, 9)
        hist["MACD_Line"]      = macd_line
        hist["MACD_Signal"]    = macd_signal
        hist["MACD_Histogram"] = macd_hist

        stoch_k, stoch_d = compute_stochastic(high, low, close, 14, 3)
        hist["Stoch_K"] = stoch_k
        hist["Stoch_D"] = stoch_d

        hist["OBV"] = compute_obv(close, volume)

        macd_conditions = [
            (macd_line > macd_signal) & (macd_line > 0),
            (macd_line > macd_signal) & (macd_line <= 0),
            (macd_line < macd_signal) & (macd_line < 0),
            (macd_line < macd_signal) & (macd_line >= 0),
        ]
        hist["macd_buy_sell"] = np.select(macd_conditions, ["Strong Buy", "Buy", "Strong Sell", "Sell"], default="Neutral")
        hist.loc[macd_line.isna() | macd_signal.isna(), "macd_buy_sell"] = "Neutral"

        hist["sma_buy_sell"] = hist["SMA_Signal"]
        hist["rsi_buy_sell"] = hist["RSI_Signal"]
        hist["Ticker"]       = ticker

        col_order = [
            "Date", "Open", "High", "Low", "Close", "Volume",
            "Dividends", "Stock Splits",
            "PreviousClose",
            "SMA20", "SMA50",
            "SMA_Signal", "SMA_Signal_Value", "SMA_Rec_Score",
            "RSI_Gain", "RSI_Loss", "RSI_Gain_Loss_Value",
            "RSI", "RSI_Signal",
            "Volume_SMA_10",
            "MACD_Line", "MACD_Signal", "MACD_Histogram",
            "Stoch_K", "Stoch_D",
            "OBV",
            "macd_buy_sell", "sma_buy_sell", "rsi_buy_sell",
            "Ticker",
        ]
        available = [c for c in col_order if c in hist.columns]
        df        = hist[available].copy()
        df        = df.tail(10).reset_index(drop=True)

        return df

    except Exception as e:
        return {"error": f"Failed to fetch 10-day technical data for '{ticker}': {str(e)}"}


# ============================================================
# /last-10-days-overview endpoint
# ============================================================
@app.post("/last-10-days-overview")
async def last_10_days_overview(request: Request):
    try:
        data    = await request.json()
        tickers = normalize_tickers_input(data)

        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        ticker_files = []
        errors       = []
        max_workers  = min(6, len(tickers))
        current_date = datetime.now().strftime("%Y-%m-%d")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(fetch_10day_technical_data, t): t
                for t in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    res = future.result()
                except Exception as exc:
                    errors.append({"ticker": ticker, "error": str(exc)})
                    continue

                if isinstance(res, dict) and "error" in res:
                    errors.append({"ticker": ticker, "error": res["error"]})
                    continue

                if not isinstance(res, pd.DataFrame) or res.empty:
                    errors.append({"ticker": ticker, "error": "No rows returned"})
                    continue

                buf = safe_to_parquet(res)
                file_content   = buf.read()
                base64_content = base64.b64encode(file_content).decode("utf-8")

                ticker_files.append({
                    "ticker":     ticker,
                    "date":       current_date,
                    "data_type":  "technical",
                    "content":    base64_content,
                    "size_bytes": len(file_content),
                    "rows":       len(res),
                })

        if not ticker_files:
            raise HTTPException(status_code=404, detail="No valid data found for any ticker")

        return {
            "status":         "success",
            "count":          len(ticker_files),
            "data_type":      "technical",
            "files":          ticker_files,
            "tickers_failed": len(errors),
            **({"errors": errors} if errors else {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===========================
# /history-5y endpoint
# ===========================
# def fetch_5y_data(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         df    = stock.history(period="5y", interval="1d")

#         if df.empty:
#             return None

#         df = df.reset_index()
#         df['Date']          = df['Date'].dt.strftime('%Y-%m-%d')
#         df["PreviousClose"] = df["Close"].shift(1)

#         return df

#     except Exception as e:
#         print(f"Error fetching {ticker}: {e}")
#         return None
def _exclude_today(df, date_col='Date'):
    today_str = datetime.now().strftime('%Y-%m-%d')
    return df[df[date_col].astype(str).str[:10] < today_str].reset_index(drop=True)


def fetch_5y_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period="5y", interval="1d")

        if df.empty:
            return None

        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # FIX: drop today's row — market may still be open / data incomplete
        df = _exclude_today(df, 'Date')

        if df.empty:
            return None

        df["PreviousClose"] = df["Close"].shift(1)

        return df

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

@app.post("/history-5y")
async def history_5y(request: Request):
    try:
        data    = await request.json()
        tickers = normalize_tickers_input(data)

        if not tickers:
            raise HTTPException(status_code=400, detail="Tickers are required")

        ticker_files     = []
        max_workers      = min(4, len(tickers))
        current_date_str = datetime.now().strftime("%Y-%m-%d")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_5y_data, t): t for t in tickers}
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                res    = future.result()

                if isinstance(res, pd.DataFrame):
                    buffer = safe_to_parquet(res)
                    file_content   = buffer.read()
                    base64_content = base64.b64encode(file_content).decode('utf-8')

                    ticker_files.append({
                        "ticker":     ticker,
                        "date":       current_date_str,
                        "data_type":  "technical",
                        "period":     "5y",
                        "content":    base64_content,
                        "size_bytes": len(file_content)
                    })

        if not ticker_files:
            raise HTTPException(status_code=404, detail="No valid data found")

        return {
            "status":    "success",
            "count":     len(ticker_files),
            "data_type": "technical",
            "files":     ticker_files,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# ===========================
# App Entry Point
# CHANGED: app.run(debug=True) → uvicorn.run(...)
# port kept at 5000 to match existing Flask setup
# ===========================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=False   # set True only during local development
    )
