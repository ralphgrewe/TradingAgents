#!/usr/bin/env python3
"""
compute_ratios.py — Fundamental ratio computation script (A2)

Usage:
  python compute_ratios.py <ticker_info.json> <income_annual.json> \
      <balance_annual.json> <cashflow_annual.json> [<holders.json>]

All JSON files are raw yfinance MCP tool output (save the tool result text to a file first).

Prints JSON with keys:
  context, annual, quarterly, insider_transactions, insider_sentiment, forecast

The model then reads this table and evaluates value/growth signal + confidence.
Exit 0 on success, 1 on error.
"""

import json, sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)


def r2(x):
    return round(float(x), 4) if x is not None else None


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return r2(a / b)


def normalize_statements(raw):
    """
    Accept yfinance financial data in two common shapes:
      A) {field_name: {date_str: value, ...}, ...}   ← columnar
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


def pick(d, *keys):
    """Return the first non-None value found under any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


# ── memory wiring (issue #9) ─────────────────────────────────────────────────
#
# The fundamental skill is wired into the shared memory core in issue #9,
# mechanically repeating the pattern validated by the quant pilot (issue
# #8). Unlike quant, value/growth `signal`/`confidence` are not produced by
# this script — they are the model's own judgment (SKILL.md Steps 2-3)
# applied to the ratio table `compute()` below computes. Keeping the
# `memory_store_decision` payload construction here as pure functions of
# the already-written envelope `details` (no I/O, no memory-context
# parameter) mirrors quant's `confidence_to_score`/`build_key_drivers`: it
# guarantees these helpers are structurally incapable of feeding Past
# Context back into the value/growth signals, and keeps them exercised by
# the same unit tests as the rest of this module.

# Same HIGH/MEDIUM/LOW -> numeric convention as skills/quant/compute_indicators.py's
# `confidence_to_score` and skills/trader/score_trader.py's `conf_weight` (also
# documented in tradingagents/memory/stats.py's confidence bucket derivation) —
# reused here rather than inventing a second mapping.
CONFIDENCE_SCORE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}


def confidence_to_score(confidence):
    """Map a HIGH/MEDIUM/LOW confidence to the numeric score
    `memory_store_decision`'s `confidence` argument expects. Unknown/missing
    values fall back to 0.3 (LOW), matching `conf_weight`'s default."""
    return CONFIDENCE_SCORE.get(str(confidence).upper(), 0.3)


def build_key_drivers(details):
    """Build the `key_drivers` payload for `memory_store_decision` from the
    envelope's `details`: the `value`/`growth` sub-signals (`signal`,
    `confidence`, `key_ratios`) and `insider_sentiment`, verbatim (issue
    #9). No new derivation — every value here already exists in `details`,
    written by the model in SKILL.md Steps 2-4."""
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


# ── main ─────────────────────────────────────────────────────────────────────

def compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw=None):
    """Compute the fundamental ratio table (`context`, `annual`,
    `insider_sentiment`, `forecast`) from raw yfinance MCP tool output.

    Pure function of its arguments only — deliberately takes no memory /
    past-context input, so the deterministic ratio table computed here can
    never be influenced by injected lessons (issue #9's "Verify"
    requirement, mirroring issue #8's `compute()`). `main()` is a thin CLI
    wrapper around this. The model then evaluates `value`/`growth` signals
    from this table in SKILL.md Steps 2-3.
    """
    if holders_raw is None:
        holders_raw = {}

    # yfinance_get_ticker_info may return the dict directly or wrapped
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


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: compute_ratios.py <ticker_info.json> <income.json> "
            "<balance.json> <cashflow.json> [<holders.json>]",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        info_raw     = load(sys.argv[1])
        income_raw   = load(sys.argv[2])
        balance_raw  = load(sys.argv[3])
        cashflow_raw = load(sys.argv[4])
        holders_raw  = load(sys.argv[5]) if len(sys.argv) > 5 else {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    result = compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
