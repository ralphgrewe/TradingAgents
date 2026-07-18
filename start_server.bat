@echo off
set MCP_TRANSPORT=streamable-http
set FASTMCP_HOST=0.0.0.0
set FASTMCP_PORT=8000
venv\Scripts\python.exe mcp_server.py
