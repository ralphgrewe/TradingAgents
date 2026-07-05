"""Portfolio-level rebalancing for TradingAgents.

Ports the ``skills/portfolio-manager`` reference implementation (style
tables, deterministic allocation, execution ordering) so that
``run_trading_agents.py`` can drive a *portfolio* mode on top of the
existing per-ticker pipeline: run the full five-stage pipeline per ticker,
map the final 5-tier rating to a style-table signal, compute target
allocation, and execute rebalancing trades against the simulator via
``tradingagents.simulation.SimulationClient``.

See ``rebalance.py`` for the pure allocation math and ``runner.py`` for the
orchestration (rating extraction, quote fetching, trade execution, report
envelope).
"""
