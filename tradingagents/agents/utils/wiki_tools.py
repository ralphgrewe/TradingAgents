"""Tool for searching the LLM-wiki strategy knowledge base.

Part of issue #104 (LLM-wiki agent-callable search tool + shared tool-loop helper).
Provides on-demand access to strategy, signal, and risk knowledge for agents who need
to consult the knowledge base during execution (portfolio manager, swing trader).
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def search_strategy_wiki(
    query: Annotated[str, "Free-text search query for strategy knowledge"],
    k: Annotated[int, "Maximum number of results to return"] = 3,
) -> str:
    """
    Search the LLM-wiki strategy knowledge base for relevant articles.

    Consult this tool when you need to understand trading strategies, technical signals,
    risk management approaches, or regime-specific tactics. The knowledge base contains
    academic research and practitioner wisdom on momentum, mean reversion, volatility,
    correlations, and other market factors relevant to decision-making.

    Returns a formatted markdown string with the top-k matching articles from the knowledge base.
    Each result includes the article ID, title, tags, source citation, and relevance score.

    Args:
        query (str): Free-text search query (e.g., "momentum strategy", "risk of gap openings")
        k (int): Maximum number of articles to return (default 3)

    Returns:
        str: Formatted markdown string of results suitable for an LLM to read
    """
    results = route_to_vendor("search_wiki", query, k)

    if not results:
        return "No matching knowledge base articles found for this query."

    markdown = "**Strategy Knowledge Base Results:**\n"
    for item in results:
        markdown += f"\n[{item['id']}] {item['title']}\n"
        if item.get("tags"):
            markdown += f"Tags: {', '.join(str(tag) for tag in item['tags'])}\n"
        if item.get("source"):
            source = item["source"]
            source_str = ""
            if isinstance(source, dict):
                parts = []
                if source.get("authors"):
                    parts.append(source["authors"])
                if source.get("title"):
                    parts.append(f'"{source["title"]}"')
                if source.get("year"):
                    parts.append(f"({source['year']})")
                if parts:
                    source_str = " ".join(parts)
            else:
                source_str = str(source)
            if source_str:
                markdown += f"Source: {source_str}\n"
        markdown += f"Score: {item.get('score', 'N/A'):.3f}\n" if isinstance(
            item.get("score"), (int, float)
        ) else f"Score: {item.get('score', 'N/A')}\n"

    return markdown
