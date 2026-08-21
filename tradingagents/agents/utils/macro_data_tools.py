import json
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_indicators(
    indicator: Annotated[
        str,
        "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
        "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
        "'real_gdp', 'vix', or a raw FRED series ID such as 'CPIAUCSL'.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
    look_back_days: Annotated[
        int | None, "Trailing window length in days; omit for a 1-year window"
    ] = None,
) -> str:
    """
    Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
    Economic Data): policy rates, Treasury yields, inflation, labor, and growth.
    Returns the series title, units, frequency, the latest value, the change
    over the window, and a recent observation table. Uses the configured
    macro_data vendor.

    Args:
        indicator (str): Friendly alias or raw FRED series ID
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Trailing window length; omit for a 1-year window

    Returns:
        str: A formatted markdown report of the macro series
    """
    return route_to_vendor("get_macro_indicators", indicator, curr_date, look_back_days)


@tool
def get_macro_pack(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the as-of snapshot date"],
) -> str:
    """
    Retrieve the full deterministic macro indicator pack for curr_date: every
    series in the macro indicator universe (policy rates, Treasury yields,
    inflation, growth, labor, dollar index, VIX, consumer sentiment, gold, and
    an explicit "unavailable" slot for CoT), each with Python-computed derived
    features already attached (latest value, prior value, delta, direction,
    trailing z-score) rather than raw series requiring further arithmetic.
    One call replaces the ~15 individual get_macro_indicators calls it would
    otherwise take to build the same picture. A series that fails to fetch
    (including a missing FRED_API_KEY) is marked unavailable in its own entry
    rather than failing the whole pack. Uses the configured macro_data vendor.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format

    Returns:
        str: A JSON-formatted indicator pack (or a plain-text
            "DATA_UNAVAILABLE"/error message when the whole pack could not be
            built).
    """
    pack = route_to_vendor("get_macro_pack", curr_date)
    if isinstance(pack, str):
        # An error/degradation sentinel from the routing layer (e.g. the
        # optional-category DATA_UNAVAILABLE message) — already a plain
        # message, pass it through unchanged rather than re-wrapping it.
        return pack
    return json.dumps(pack, indent=2, sort_keys=True)
