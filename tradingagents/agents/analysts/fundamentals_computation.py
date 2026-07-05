"""
Deterministic fundamental ratio computation module for fundamentals analyst.

Adapted from skills/fundamental/compute_ratios.py, but integrated into the
pipeline to be called synchronously with pipeline data-access functions.
This module computes all fundamental ratios, signal/confidence derivation,
and builds the JSON envelope payload — the LLM contributes only value/growth
evaluations and a one-line summary.

Key design principle: pure computation, no LLM, deterministic ratio tables.
"""

import json
import math
from typing import Any, Optional

import yfinance as yf
import pandas as pd

from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.dataflows.y_finance import yf_retry


# ── helpers ──────────────────────────────────────────────────────────────────

def r2(x):
    """Round to 4 decimal places."""
    return round(float(x), 4) if x is not None else None


def ratio(a, b):
    """Compute ratio a/b, returning None on invalid input."""
    if a is None or b is None or b == 0:
        return None
    return r2(a / b)


def pick(d, *keys):
    """Return the first non-None value found under any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def normalize_statements(raw):
    """
    Accept financial data in two common shapes:
      A) {field_name: {date_str: value, ...}, ...}   ← columnar (yfinance DataFrame columns)
      B) [{Date: ..., field: value, ...}, ...]        ← row-oriented
    Returns {date_str_10char: {field: value}}.
    """
    if isinstance(raw, dict):
        result = {}
        for field, date_vals in raw.items():
            if not isinstance(date_vals, dict):
                continue
            for date, val in date_vals.items():
                key = str(date)[:10]
                result.setdefault(key, {})[field] = val
        return result

    if isinstance(raw, list):
        result = {}
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            date = str(rec.get("Date") or rec.get("date") or "")[:10]
            if not date:
                continue
            for k, v in rec.items():
                if k not in ("Date", "date"):
                    result.setdefault(date, {})[k] = v
        return result

    return {}


# ── memory wiring (issue #9) ─────────────────────────────────────────────────

CONFIDENCE_SCORE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}


def confidence_to_score(confidence):
    """Map HIGH/MEDIUM/LOW confidence to numeric score for memory_store_decision."""
    return CONFIDENCE_SCORE.get(str(confidence).upper(), 0.3)


def build_key_drivers(details):
    """Build key_drivers payload for memory_store_decision from envelope details."""
    value = details.get("value") or {}
    growth = details.get("growth") or {}
    return {
        "value": {
            "signal": value.get("signal"),
            "confidence": value.get("confidence"),
            "key_ratios": value.get("key_ratios", []),
        },
        "growth": {
            "signal": growth.get("signal"),
            "confidence": growth.get("confidence"),
            "key_ratios": growth.get("key_ratios", []),
        },
        "insider_sentiment": details.get("insider_sentiment"),
    }


# ── Fetch raw yfinance data ────────────────────────────────────────────────────

def fetch_ticker_info(ticker: str) -> dict:
    """Fetch ticker info (comprehensive company data)."""
    try:
        canonical = normalize_symbol(ticker)
        ticker_obj = yf.Ticker(canonical)
        info = yf_retry(lambda: ticker_obj.info)
        return info if info else {}
    except Exception as e:
        return {}


def fetch_income_statement(ticker: str, freq: str = "annual") -> dict:
    """Fetch income statement data."""
    try:
        canonical = normalize_symbol(ticker)
        ticker_obj = yf.Ticker(canonical)
        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_financials)
        else:
            data = yf_retry(lambda: ticker_obj.financials)

        if isinstance(data, pd.DataFrame):
            return data.to_dict()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {}


def fetch_balance_sheet(ticker: str, freq: str = "annual") -> dict:
    """Fetch balance sheet data."""
    try:
        canonical = normalize_symbol(ticker)
        ticker_obj = yf.Ticker(canonical)
        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet)
        else:
            data = yf_retry(lambda: ticker_obj.balance_sheet)

        if isinstance(data, pd.DataFrame):
            return data.to_dict()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {}


def fetch_cashflow(ticker: str, freq: str = "annual") -> dict:
    """Fetch cash flow statement data."""
    try:
        canonical = normalize_symbol(ticker)
        ticker_obj = yf.Ticker(canonical)
        if freq.lower() == "quarterly":
            data = yf_retry(lambda: ticker_obj.quarterly_cashflow)
        else:
            data = yf_retry(lambda: ticker_obj.cashflow)

        if isinstance(data, pd.DataFrame):
            return data.to_dict()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {}


def fetch_holders(ticker: str) -> dict:
    """Fetch insider transactions data."""
    try:
        canonical = normalize_symbol(ticker)
        ticker_obj = yf.Ticker(canonical)

        # Try to get insider transactions
        insider_txns = yf_retry(lambda: ticker_obj.insider_transactions)
        if isinstance(insider_txns, pd.DataFrame) and not insider_txns.empty:
            return {"insider_transactions": insider_txns.to_dict('records')}
        return {}
    except Exception:
        return {}


# ── Main computation ──────────────────────────────────────────────────────────

def compute(ticker: str) -> dict[str, Any]:
    """
    Compute the fundamental ratio table (context, annual, insider_sentiment)
    from live yfinance data.

    Args:
        ticker: stock ticker symbol

    Returns:
        dict with keys: context, annual, insider_sentiment, forecast
        Deterministic: same ticker → same output (within same market day).
    """
    # Fetch raw data
    info_raw = fetch_ticker_info(ticker)
    income_raw = fetch_income_statement(ticker, "annual")
    balance_raw = fetch_balance_sheet(ticker, "annual")
    cashflow_raw = fetch_cashflow(ticker, "annual")
    holders_raw = fetch_holders(ticker)

    if holders_raw is None:
        holders_raw = {}

    # Ensure info is a dict
    info = info_raw if isinstance(info_raw, dict) else {}

    # ── Context ───────────────────────────────────────────────────────────────
    context = {
        "market_cap":        info.get("marketCap"),
        "sector":            info.get("sector"),
        "industry":          info.get("industry"),
        "52w_high":          info.get("fiftyTwoWeekHigh"),
        "52w_low":           info.get("fiftyTwoWeekLow"),
        "analyst_consensus": info.get("recommendationKey"),
    }

    # ── Financial statements ─────────────────────────────────────────────────
    income   = normalize_statements(income_raw)
    balance  = normalize_statements(balance_raw)
    cashflow = normalize_statements(cashflow_raw)

    all_dates = sorted(
        set(list(income) + list(balance) + list(cashflow)),
        reverse=True,
    )
    annual_dates = all_dates[:3]

    mkt_cap = info.get("marketCap")
    ent_val  = info.get("enterpriseValue")

    annual = {}
    for idx, date in enumerate(annual_dates):
        year = date[:4]
        i  = income.get(date,   {})
        b  = balance.get(date,  {})
        cf = cashflow.get(date, {})

        # ── Income statement ────────────────────────────────────────────────
        revenue      = pick(i, "Total Revenue", "TotalRevenue", "Revenue")
        gross_profit = pick(i, "Gross Profit", "GrossProfit")
        op_income    = pick(i, "Operating Income", "OperatingIncome", "EBIT")
        net_income   = pick(i, "Net Income", "NetIncome", "Net Income Common Stockholders")
        interest_exp = pick(i, "Interest Expense", "InterestExpense")
        ebitda       = pick(i, "EBITDA", "Ebitda", "Normalized EBITDA")

        # ── Balance sheet ───────────────────────────────────────────────────
        total_assets   = pick(b, "Total Assets", "TotalAssets")
        total_equity   = pick(b, "Total Stockholder Equity", "StockholdersEquity",
                               "Total Equity Gross Minority Interest",
                               "TotalEquityGrossMinorityInterest")
        total_debt     = pick(b, "Total Debt", "TotalDebt",
                               "Long Term Debt", "LongTermDebt")
        current_assets = pick(b, "Current Assets", "CurrentAssets")
        current_liab   = pick(b, "Current Liabilities", "CurrentLiabilities")
        inventory      = pick(b, "Inventory", "Inventories") or 0

        # ── Cash-flow ───────────────────────────────────────────────────────
        op_cf  = pick(cf, "Operating Cash Flow", "OperatingCashFlow",
                      "Total Cash From Operating Activities")
        capex  = pick(cf, "Capital Expenditure", "CapitalExpenditure",
                      "Capital Expenditures")
        if capex is not None:
            capex = abs(capex)
        fcf = (op_cf - capex) if (op_cf is not None and capex is not None) else None

        # ── Market-based ratios (latest year only from info) ────────────────
        pe      = None
        pb      = None
        ps      = None
        peg     = None
        ev_ebitda = None
        pcf     = None
        div_yield    = None
        payout_ratio = None

        if idx == 0:
            pe        = info.get("trailingPE")    or ratio(mkt_cap, net_income)
            pb        = info.get("priceToBook")   or ratio(mkt_cap, total_equity)
            ps        = info.get("priceToSalesTrailing12Months") or ratio(mkt_cap, revenue)
            peg       = info.get("pegRatio")
            ev_ebitda = info.get("enterpriseToEbitda") or ratio(ent_val, ebitda)
            pcf       = ratio(mkt_cap, op_cf)
            div_yield    = info.get("dividendYield")
            payout_ratio = info.get("payoutRatio")

        # ── Derived ratios ───────────────────────────────────────────────────
        gross_margin = ratio(gross_profit, revenue)
        op_margin    = ratio(op_income, revenue)
        net_margin   = ratio(net_income, revenue)
        roe          = ratio(net_income, total_equity)
        roa          = ratio(net_income, total_assets)

        inv_capital  = ((total_equity or 0) + (total_debt or 0)) or None
        nopat        = (op_income * 0.79) if op_income else None  # approx. 21% tax
        roic         = ratio(nopat, inv_capital)

        d2e          = ratio(total_debt, total_equity)
        current_r    = ratio(current_assets, current_liab)
        quick_assets = (current_assets - inventory) if current_assets is not None else None
        quick_r      = ratio(quick_assets, current_liab)
        equity_r     = ratio(total_equity, total_assets)
        int_cov      = ratio(op_income, interest_exp) if interest_exp else None

        fcf_yield    = ratio(fcf, mkt_cap) if (mkt_cap and idx == 0) else None
        fcf_margin   = ratio(fcf, revenue)
        capex_rev    = ratio(capex, revenue)

        annual[year] = {
            "valuation": {
                "pe":        r2(pe),
                "pb":        r2(pb),
                "ev_ebitda": r2(ev_ebitda),
                "pcf":       r2(pcf),
                "peg":       r2(peg),
                "ps":        r2(ps),
            },
            "profitability": {
                "gross_margin": r2(gross_margin),
                "op_margin":    r2(op_margin),
                "net_margin":   r2(net_margin),
                "roe":          r2(roe),
                "roic":         r2(roic),
                "roa":          r2(roa),
            },
            "balance_sheet": {
                "debt_to_equity":    r2(d2e),
                "current_ratio":     r2(current_r),
                "quick_ratio":       r2(quick_r),
                "equity_ratio":      r2(equity_r),
                "interest_coverage": r2(int_cov),
            },
            "cashflow": {
                "fcf":             r2(fcf),
                "op_cf":           r2(op_cf),
                "fcf_yield":       r2(fcf_yield),
                "fcf_margin":      r2(fcf_margin),
                "capex_to_revenue": r2(capex_rev),
            },
            "dividends": {
                "yield":        r2(div_yield),
                "payout_ratio": r2(payout_ratio),
            },
        }

    # ── Insider sentiment ─────────────────────────────────────────────────────
    insider_sentiment = "NEUTRAL"

    raw_txns = (
        holders_raw if isinstance(holders_raw, list)
        else holders_raw.get("insiderTransactions", holders_raw.get("insider_transactions", []))
    )
    net_shares = 0
    for t in raw_txns:
        if not isinstance(t, dict):
            continue
        side    = str(t.get("Transaction") or t.get("type") or "").lower()
        shares  = float(t.get("Shares") or t.get("shares") or 0)
        is_buy  = "buy" in side or "purchase" in side or "acquisition" in side
        is_sell = "sell" in side or "sale" in side or "disposition" in side
        net_shares += shares if is_buy else (-shares if is_sell else 0)

    if net_shares > 0:
        insider_sentiment = "BULLISH"
    elif net_shares < 0:
        insider_sentiment = "BEARISH"

    result = {
        "context":           context,
        "annual":            annual,
        "insider_sentiment": insider_sentiment,
        "forecast":          {},
    }

    return result


def build_json_envelope(
    signal: Optional[str],
    confidence: Optional[str],
    summary: str,
    details: dict,
    ticker: str,
    date: str,
) -> str:
    """
    Build the JSON envelope per skills/SCHEMA.md.

    Returns JSON string serialized for storage in fundamentals_report.
    """
    envelope = {
        "skill": "fundamental-analyst",
        "ticker": ticker,
        "date": date,
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "details": details,
    }
    return json.dumps(envelope, indent=2)
