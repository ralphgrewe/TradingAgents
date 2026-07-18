#!/usr/bin/env bash
export MCP_TRANSPORT=streamable-http
export FASTMCP_HOST=0.0.0.0
export FASTMCP_PORT=8000
./venv/bin/python mcp_server.py
