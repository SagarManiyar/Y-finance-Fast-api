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
# Fundamental Data Helpers
# ===========================

# Quant Screener v1 — the 167 fields the extract is restricted to.
# Order below IS the parquet column order. Grouped by screener section.
# Verified present on AAPL (167/167) against yfinance 1.6.0 on 2026-09-03.
QUANT_FIELDS = [
    # --- SECTION 1: identity, currency & universe control (19) ---
    "info_symbol", "info_shortName", "info_longName", "info_exchange",
    "info_fullExchangeName", "info_quoteType", "info_currency",
    "info_financialCurrency", "info_sector", "info_sectorKey", "info_industry",
    "info_industryKey", "info_country", "info_region", "info_marketCap",
    "info_lastFiscalYearEnd", "info_mostRecentQuarter",
    "info_exchangeDataDelayedBy", "info_maxAge",

    # --- SECTION 2: value factor (13) ---
    "info_trailingPE", "info_forwardPE", "info_priceToBook",
    "info_priceToSalesTrailing12Months", "info_enterpriseToEbitda",
    "info_enterpriseToRevenue", "info_pegRatio", "info_trailingPegRatio",
    "info_dividendYield", "info_enterpriseValue", "info_bookValue",
    "info_payoutRatio", "info_trailingAnnualDividendYield",

    # --- SECTION 3: growth factor (19) ---
    "info_revenueGrowth", "info_earningsGrowth", "info_earningsQuarterlyGrowth",
    "info_returnOnEquity", "info_returnOnAssets", "info_grossMargins",
    "info_operatingMargins", "info_ebitdaMargins", "info_profitMargins",
    "info_trailingEps", "info_forwardEps", "info_epsTrailingTwelveMonths",
    "info_epsForward", "info_epsCurrentYear", "info_revenuePerShare",
    "info_totalRevenue", "info_netIncomeToCommon", "info_freeCashflow",
    "info_operatingCashflow",

    # --- SECTION 3b: growth series inputs (25) ---
    "income_Total Revenue", "income_Net Income", "income_Gross Profit",
    "income_Diluted EPS", "income_Basic EPS", "income_Diluted Average Shares",
    "income_Operating Income", "income_EBIT", "income_EBITDA",
    "income_Normalized EBITDA", "income_Research And Development",
    "income_Tax Rate For Calcs", "income_Tax Provision", "income_Pretax Income",
    "quarterly_income_Total Revenue", "quarterly_income_Net Income",
    "quarterly_income_Gross Profit", "quarterly_income_Diluted EPS",
    "quarterly_income_EBIT", "quarterly_income_EBITDA",
    "quarterly_income_Operating Income", "cashflow_Free Cash Flow",
    "quarterly_cashflow_Free Cash Flow", "balance_Stockholders Equity",
    "quarterly_balance_Stockholders Equity",

    # --- SECTION 4: momentum factor, info_* portion only (22) ---
    # The rest of Momentum needs .history() OHLCV and is NOT in this extract.
    "info_currentPrice", "info_regularMarketPrice", "info_fiftyDayAverage",
    "info_twoHundredDayAverage", "info_fiftyDayAverageChangePercent",
    "info_twoHundredDayAverageChangePercent", "info_fiftyTwoWeekHigh",
    "info_fiftyTwoWeekLow", "info_fiftyTwoWeekHighChangePercent",
    "info_fiftyTwoWeekLowChangePercent", "info_fiftyTwoWeekChangePercent",
    "info_52WeekChange", "info_SandP52WeekChange", "info_volume",
    "info_regularMarketVolume", "info_averageVolume", "info_averageVolume10days",
    "info_averageDailyVolume10Day", "info_averageDailyVolume3Month",
    "info_beta", "info_allTimeHigh", "info_allTimeLow",

    # --- SECTION 5: quality factor component inputs (24) ---
    "balance_Total Assets", "balance_Total Liabilities Net Minority Interest",
    "balance_Working Capital", "balance_Retained Earnings", "balance_Total Debt",
    "balance_Net Debt", "balance_Invested Capital",
    "balance_Cash And Cash Equivalents",
    "balance_Cash Cash Equivalents And Short Term Investments",
    "balance_Current Assets", "balance_Current Liabilities",
    "balance_Long Term Debt", "balance_Inventory",
    "balance_Ordinary Shares Number", "balance_Share Issued",
    "balance_Common Stock Equity", "cashflow_Operating Cash Flow",
    "cashflow_Capital Expenditure", "cashflow_Stock Based Compensation",
    "quarterly_balance_Total Assets", "quarterly_balance_Current Assets",
    "quarterly_balance_Current Liabilities",
    "quarterly_balance_Ordinary Shares Number",
    "quarterly_cashflow_Operating Cash Flow",

    # --- SECTION 6: pre-screen removal criteria (37) ---
    "info_debtToEquity", "info_ebitda", "info_totalDebt", "info_currentRatio",
    "info_quickRatio", "info_totalCash", "info_totalCashPerShare",
    "income_Interest Expense", "income_Interest Expense Non Operating",
    "income_Net Interest Income", "cashflow_Interest Paid Supplemental Data",
    "cashflow_Issuance Of Capital Stock", "cashflow_Common Stock Issuance",
    "cashflow_Repurchase Of Capital Stock", "cashflow_Net Common Stock Issuance",
    "quarterly_cashflow_Net Common Stock Issuance", "info_sharesOutstanding",
    "info_impliedSharesOutstanding", "info_floatShares",
    "info_recommendationMean", "info_recommendationKey",
    "info_numberOfAnalystOpinions", "info_averageAnalystRating",
    "info_targetMeanPrice", "info_targetMedianPrice", "info_targetHighPrice",
    "info_targetLowPrice", "info_auditRisk", "info_overallRisk",
    "info_sharesShort", "info_sharesShortPriorMonth", "info_shortRatio",
    "info_shortPercentOfFloat", "info_sharesPercentSharesOut",
    "info_dateShortInterest", "info_heldPercentInstitutions",
    "info_heldPercentInsiders",

    # --- SECTION 7: earnings timing (8) ---
    "info_earningsTimestamp", "info_earningsTimestampStart",
    "info_earningsTimestampEnd", "info_isEarningsDateEstimate",
    "info_exDividendDate", "info_dividendDate", "info_lastSplitDate",
    "info_lastSplitFactor",
]

# Fields that gate the pre-screen removal criteria. A ticker whose extract has
# NO usable value for any of these is returned as status "failed", because a
# null here would silently pass a stock that the removal criteria should reject.
#
# Deliberately NOT the full 37-field Section 6 list: the 6 short-interest fields
# and cashflow_Interest Paid Supplemental Data were absent on every non-US
# listing tested (SHEL.L / 005930.KS / NPN.JO), so gating on those would fail
# 100% of non-US tickers. Every field below was present on all 4 tickers tested.
PRESCREEN_CRITICAL_FIELDS = [
    "info_debtToEquity",        # Heavy Debt — D/E
    "info_ebitda",              # Net Debt/EBITDA denominator
    "info_totalDebt",           # leverage
    "info_currentRatio",        # solvency stress: current ratio < 1
    "info_sharesOutstanding",   # dilution screen
    "income_Interest Expense",  # interest coverage
]

# Prefix -> yfinance accessor. LONGEST PREFIX FIRST so that, for example,
# "quarterly_income_*" is never mis-bucketed into the annual "income_*" source.
_FUNDAMENTAL_SOURCES = [
    ("quarterly_income_", "quarterly_income_stmt"),
    ("quarterly_balance_", "quarterly_balance_sheet"),
    ("quarterly_cashflow_", "quarterly_cashflow"),
    ("income_", "income_stmt"),
    ("balance_", "balance_sheet"),
    ("cashflow_", "cashflow"),
]

_QUANT_FIELDS_SET = set(QUANT_FIELDS)


def get_error_suggestion(ticker, error_msg):
    """Generate error suggestions based on error message."""
    error_lower = str(error_msg).lower()

    if "invalid" in error_lower or "not found" in error_lower:
        return f"Invalid ticker symbol '{ticker}'. Check if the symbol is correct. Use official stock exchange symbols (e.g., AAPL for Apple)."
    elif "delisted" in error_lower or "no data" in error_lower:
        return f"No data available for '{ticker}'. The company may be delisted, or the ticker symbol may have changed."
    elif "network" in error_lower or "connection" in error_lower:
        return "Network error connecting to YFinance. Please try again later."
    else:
        return f"Unable to fetch fundamental data for '{ticker}'. Verify ticker symbol and try again."


def fetch_fundamental_data(ticker):
    """
    Fetch the 167 Quant Screener fundamental fields for a ticker.

    Returns tuple: (data_df, error_msg, missing_fields)

    Output shape — identical to the earlier full-field extract, just narrowed:
      - Exactly 167 columns, in QUANT_FIELDS order. Fields Yahoo does not
        return are still present, filled with null, so every ticker's parquet
        has the same schema and 100 files concatenate without reconciliation.
      - One row per fiscal period, ordinally aligned: row 0 = most recent
        period, row 1 = next most recent, etc. Row count = the longest
        statement returned for that ticker (4-7 in testing).
      - info_* are point-in-time snapshots, so they carry a value on row 0
        and null on rows 1+.

    NOTE on ordinal alignment: annual and quarterly statements have different
    period-end dates and different lengths, so row 1 of an "income_*" column
    and row 1 of a "quarterly_income_*" column are DIFFERENT dates. This
    extract carries no period-end date columns, so the dates are not
    recoverable from the parquet alone.
    """
    try:
        stock = yf.Ticker(ticker)

        snapshot = {}   # info_* -> single scalar value
        series = {}     # statement fields -> array, one value per fiscal period

        # --- 1. Stock info (snapshot) ---
        try:
            info = stock.info or {}
            for key, value in info.items():
                field = "info_" + key
                if field in _QUANT_FIELDS_SET:
                    snapshot[field] = value
        except Exception as e:
            print(f"Warning: could not fetch info for {ticker}: {e}")

        # --- 2-7. Annual + quarterly income / balance / cashflow (series) ---
        for prefix, accessor in _FUNDAMENTAL_SOURCES:
            try:
                stmt = getattr(stock, accessor)
            except Exception as e:
                print(f"Warning: could not fetch {accessor} for {ticker}: {e}")
                continue

            if not isinstance(stmt, pd.DataFrame) or stmt.empty:
                continue

            # Yahoo returns period columns most-recent-first today, but nothing
            # in the API guarantees that. Sort explicitly descending: row 0 MUST
            # be the most recent period, or every CAGR / YoY delta computed
            # downstream silently inverts sign with no visible error.
            try:
                stmt = stmt.reindex(columns=sorted(stmt.columns, reverse=True))
            except TypeError:
                pass  # unorderable column labels — keep Yahoo's own order

            # Index positionally rather than by label: Yahoo occasionally
            # repeats a line-item label, and .loc on a duplicate label returns
            # a DataFrame instead of a Series. First occurrence wins.
            for pos, label in enumerate(stmt.index):
                field = prefix + str(label)
                if field in _QUANT_FIELDS_SET and field not in series:
                    series[field] = stmt.iloc[pos].to_numpy()

        if not snapshot and not series:
            return None, get_error_suggestion(ticker, "No fundamental data found"), list(QUANT_FIELDS)

        # Row count = longest statement actually returned for this ticker.
        max_len = max((len(arr) for arr in series.values()), default=1) or 1

        # Build every one of the 167 columns, in spec order, padding to max_len.
        columns = {}
        for field in QUANT_FIELDS:
            if field in series:
                values = list(series[field])[:max_len]
                values += [None] * (max_len - len(values))
            elif field in snapshot:
                values = [snapshot[field]] + [None] * (max_len - 1)
            else:
                values = [None] * max_len
            columns[field] = values

        df = pd.DataFrame(columns, columns=QUANT_FIELDS)

        if df.empty:
            return None, get_error_suggestion(ticker, "Empty fundamental data"), list(QUANT_FIELDS)

        # A field counts as missing when it has no usable value at all — either
        # Yahoo never returned it, or returned it as null. Both are unusable
        # downstream, and the pre-screen gate needs to treat them the same.
        missing_fields = [f for f in QUANT_FIELDS if df[f].isna().all()]

        return df, None, missing_fields

    except Exception as e:
        error_msg = str(e)
        suggestion = get_error_suggestion(ticker, error_msg)
        return None, suggestion, list(QUANT_FIELDS)


def build_fundamental_file_entry(ticker, date_str, timestamp):
    """
    Fetch one ticker and shape it into a single "files" entry.

    Returns a dict whose "status" is:
      - "success" — parquet built, all pre-screen gate fields usable
      - "failed"  — no data at all, OR a pre-screen gate field has no usable
                    value (see PRESCREEN_CRITICAL_FIELDS for why this gates)
    """
    df, error, missing_fields = fetch_fundamental_data(ticker)

    if df is None or df.empty:
        return {
            "ticker": ticker,
            "date": date_str,
            "status": "failed",
            "error": error or "No fundamental data available",
            "suggestion": get_error_suggestion(ticker, error or "No data"),
        }

    missing_set = set(missing_fields)
    missing_critical = [f for f in PRESCREEN_CRITICAL_FIELDS if f in missing_set]

    if missing_critical:
        return {
            "ticker": ticker,
            "date": date_str,
            "status": "failed",
            "error": "Missing pre-screen field(s): " + ", ".join(missing_critical),
            "suggestion": (
                f"'{ticker}' returned data but has no usable value for "
                f"{len(missing_critical)} pre-screen removal-criteria field(s). "
                "Scoring this ticker could pass a stock the removal criteria "
                "should reject. Common causes: a non-equity quote type (ETF, "
                "fund, index), a recent IPO with no full statement history, or "
                "a financial-sector name where Yahoo omits these line items."
            ),
            "missing_critical_fields": missing_critical,
            "missing_fields": missing_fields,
            "missing_count": len(missing_fields),
        }

    buffer = safe_to_parquet(df)
    file_content = buffer.read()
    base64_content = base64.b64encode(file_content).decode("utf-8")

    return {
        "ticker": ticker,
        "date": date_str,
        "fetch_date": timestamp,
        "data_type": "fundamental",
        "status": "success",
        "content": base64_content,
        "size_bytes": len(file_content),
        "metadata": {
            "rows": len(df),
            "columns": len(df.columns),
            "fields": df.columns.tolist(),
            "missing_fields": missing_fields,
            "missing_count": len(missing_fields),
            "data_sources": [
                "stock_info", "income_statement", "quarterly_income_statement",
                "balance_sheet", "quarterly_balance_sheet",
                "cash_flow", "quarterly_cash_flow",
            ],
        },
    }


@app.post("/fundamentals/{ticker}")
async def get_fundamentals_single(ticker: str):
    """
    Fetch the 167 Quant Screener fundamental fields for a single ticker.
    Returns: JSON metadata + parquet content (base64 encoded)
    """
    try:
        ticker = ticker.upper().strip()

        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        current_timestamp = datetime.now().isoformat() + "Z"

        entry = build_fundamental_file_entry(ticker, current_date_str, current_timestamp)

        return JSONResponse(
            status_code=200,
            content={
                "status": entry["status"],
                "count": 1,
                "data_type": "fundamental",
                "field_count": len(QUANT_FIELDS),
                "files": [entry],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/fundamentals")
async def get_fundamentals_batch(request: Request):
    """
    Fetch the 167 Quant Screener fundamental fields for multiple tickers.
    Body: {"tickers": ["AAPL", "TSLA", ...]}
    Max 100 tickers per request.
    Returns: JSON with a per-ticker status, one parquet per ticker.
    """
    try:
        data = await request.json()
        tickers_input = data.get("tickers", [])

        if not tickers_input:
            raise HTTPException(status_code=400, detail="Tickers array is required")

        # Normalize tickers
        if isinstance(tickers_input, str):
            raw = tickers_input.split(",")
        elif isinstance(tickers_input, list):
            raw = tickers_input
        else:
            raise HTTPException(status_code=400, detail="Tickers must be string or array")

        # Dedupe while preserving the caller's order, so the response "files"
        # array lines up with the order the tickers were sent in.
        tickers = []
        seen = set()
        for item in raw:
            symbol = str(item).strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                tickers.append(symbol)

        if not tickers:
            raise HTTPException(status_code=400, detail="No valid tickers provided")

        if len(tickers) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 tickers per request")

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        current_timestamp = datetime.now().isoformat() + "Z"

        results = {}
        max_workers = min(10, len(tickers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    build_fundamental_file_entry, t, current_date_str, current_timestamp
                ): t
                for t in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    print(f"Error building fundamentals for {ticker}: {e}")
                    results[ticker] = {
                        "ticker": ticker,
                        "date": current_date_str,
                        "status": "failed",
                        "error": str(e),
                        "suggestion": get_error_suggestion(ticker, str(e)),
                    }

        ticker_files = [results[t] for t in tickers]

        success_count = sum(1 for f in ticker_files if f.get("status") == "success")
        failed_count = len(ticker_files) - success_count

        return {
            "status": "completed",
            "count": len(ticker_files),
            "data_type": "fundamental",
            "field_count": len(QUANT_FIELDS),
            "summary": {
                "total": len(ticker_files),
                "success": success_count,
                "failed": failed_count,
                "failed_tickers": [
                    f["ticker"] for f in ticker_files if f.get("status") == "failed"
                ],
            },
            "files": ticker_files,
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
