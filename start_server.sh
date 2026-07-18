#!/usr/bin/env bash
export MCP_TRANSPORT=streamable-http
export FASTMCP_HOST=127.0.0.1
export FASTMCP_PORT=8001
./venv/bin/python mcp_server.py
