"""Perplexity News Analyst - Uses Perplexity Agent API for news analysis.

This analyst leverages Perplexity's Agent API with built-in web search and URL fetch
capabilities to generate structured news analysis reports. Unlike traditional
news analysts that fetch raw articles and then analyze them, this analyst uses
Perplexity's native tools to search, fetch, and analyze news in a single workflow.

The analyst returns structured JSON with:
- summary: Overall news summary
- sentiment: Numeric sentiment score (-1 to 1)
- key_events: List of important events
- risk_factors: List of identified risks
"""

import json
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)

logger = logging.getLogger(__name__)


def create_perplexity_news_analyst(llm):
    """Create a Perplexity-based news analyst node.
    
    This analyst uses Perplexity's Agent API to search for and analyze
    recent news about a given ticker. It returns a structured report with
    summary, sentiment, key events, and risk factors.
    
    Args:
        llm: The LLM instance (should be a PerplexityClient for full functionality)
        
    Returns:
        A node function that processes state and returns news analysis
    """
    def perplexity_news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        
        # Check if we have a Perplexity agent client available
        perplexity_agent = None
        try:
            # Try to get agent client from LLM if it's a PerplexityClient
            if hasattr(llm, 'get_agent_client'):
                perplexity_agent = llm.get_agent_client()
        except Exception as e:
            logger.warning(f"Could not initialize Perplexity agent client: {e}")
        
        # Define tools for Perplexity agent
        # These match the user's example: web_search with recency and domain filtering
        tools = [
            {
                "type": "web_search",
                "recency": "24h",
                "domain_allowlist": ["reuters.com", "bloomberg.com", "wsj.com", 
                                     "financialtimes.com", "cnbc.com", 
                                     "marketwatch.com", "seekingalpha.com"]
            },
            {"type": "fetch_url"}
        ]
        
        # Prepare messages for Perplexity agent
        messages = [
            {
                "role": "user",
                "content": f"Analyze recent news for {ticker}. Return JSON with summary, sentiment (-1 to 1), key_events[], risk_factors[]" 
            }
        ]
        
        response_format = {"type": "json_object"}
        
        # Try to use Perplexity Agent API directly if available
        if perplexity_agent:
            try:
                result = perplexity_agent.run_agent(
                    model=llm.model if hasattr(llm, 'model') else "sonar-pro",
                    messages=messages,
                    tools=tools,
                    response_format=response_format
                )
                
                # Handle the response - Perplexity returns various formats
                if hasattr(result, 'content'):
                    content = result.content
                elif isinstance(result, dict):
                    content = result.get('content', '')
                else:
                    content = str(result)
                
                # Parse JSON if present
                try:
                    # Try to extract JSON from the response
                    if isinstance(content, str):
                        # Look for JSON in the content
                        import re
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            report_data = json.loads(json_str)
                            report = json.dumps(report_data, indent=2)
                        else:
                            # If no JSON found, use the content as-is
                            report = content
                    else:
                        report = json.dumps(content, indent=2) if isinstance(content, dict) else str(content)
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Could not parse Perplexity response as JSON: {e}")
                    report = content if isinstance(content, str) else str(content)
                
                return {
                    "messages": [{"role": "assistant", "content": report}],
                    "perplexity_news_report": report,
                }
            except Exception as e:
                logger.error(f"Error calling Perplexity Agent API: {e}")
                # Fall back to using the LLM directly
                pass
        
        # Fallback: Use the LLM directly with a structured prompt
        # This works even if we don't have the Perplexity SDK or if there's an error
        
        # Build system message without f-string issues
        lang_instruction = get_language_instruction()
        system_message = (
            "You are a financial news analyst specializing in analyzing recent news for trading decisions. "
            f"Analyze the latest news for {ticker} and provide a comprehensive assessment.\n\n"
            f"{ticker} is the instrument to analyze. Use this exact ticker in your analysis.\n"
            f"Current date: {current_date}\n\n"
            f"{instrument_context}\n\n"
            "Your task is to:\n"
            "1. Search for the most recent news from the past 24 hours\n"
            "2. Identify key events, announcements, and developments\n"
            "3. Assess the overall sentiment and its impact on the stock\n"
            "4. Highlight potential risks and opportunities\n\n"
            "Return your analysis as a JSON object with: summary, sentiment (-1 to 1), key_events[], risk_factors[]"
            + lang_instruction
        )
        
        # Standard agent system prompt
        agent_system_prompt = (
            "You are a helpful AI assistant, collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK; another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, "
            "prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop.\n\n"
            + system_message
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", agent_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        chain = prompt | llm
        result = chain.invoke(state["messages"])
        
        report = ""
        if hasattr(result, 'content'):
            report = result.content
        elif isinstance(result, str):
            report = result
        else:
            report = str(result)
        
        return {
            "messages": [result],
            "perplexity_news_report": report,
        }
    
    return perplexity_news_analyst_node
