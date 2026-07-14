@echo off
set MCP_TRANSPORT=streamable-http
set FASTMCP_HOST=127.0.0.1
set FASTMCP_PORT=8001
venv\Scripts\python.exe mcp_server.py
