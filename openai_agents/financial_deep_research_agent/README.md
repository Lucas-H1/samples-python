# Financial Deep Research Agent

Multi-agent financial research system with specialized roles, extended with Temporal's durable execution and an **explore-exploit pattern** for iterative deep research.

*Adapted from [OpenAI Agents SDK financial research agent](https://github.com/openai/openai-agents-python/tree/main/examples/financial_research_agent)*

## Architecture

This example shows how you might compose a richer financial research agent using the Agents SDK with an **explore-exploit pattern** for comprehensive research. The system iteratively explores sub-topics and then deeply researches each one.

### Explore-Exploit Pattern

The research process follows a sophisticated explore-exploit pattern:

1. **Exploration Phase**: The orchestrator agent identifies 3-5 key sub-topics or themes that need to be researched
2. **Exploitation Phase**: For each sub-topic, the system creates detailed search plans and performs deep research
3. **Iteration**: The process repeats for multiple iterations, allowing for deeper exploration of promising areas
4. **Synthesis**: All research from all iterations is synthesized into a comprehensive final report

### Research Flow

1. **Initial Exploration**: A planner agent identifies high-level sub-topics relevant to the financial analysis
2. **Iterative Exploitation**: For each iteration:
   - Create detailed search plans for each sub-topic
   - Perform parallel web searches for each sub-topic
   - Gather comprehensive information
3. **Sub-analysts**: Specialist agents (fundamentals and risk analysts) are exposed as tools for inline analysis
4. **Comprehensive Writing**: A senior writer agent synthesizes all research from all iterations into a comprehensive report
5. **Verification**: A final verifier agent audits the report for consistency and accuracy

## Running the Example

First, start the worker:
```bash
uv run openai_agents/financial_deep_research_agent/run_worker.py
```

Then run the financial deep research workflow:
```bash
uv run openai_agents/financial_deep_research_agent/run_financial_deep_research_workflow.py
```

Enter a query like:
```
Write up an analysis of Apple Inc.'s most recent quarter.
```

You can also just hit enter to run this query, which is provided as the default.

### Configuration

The `FinancialDeepResearchManager` supports configuration parameters:
- `max_iterations`: Maximum number of research iterations (default: 2)
- `max_depth_per_topic`: Maximum depth to research each sub-topic (default: 2)

### Timeout Configuration

The system is configured with extended timeouts to handle the iterative research process:
- **Activity timeouts**: 10 minutes start-to-close, 2 minutes schedule-to-start, 5 minutes heartbeat
- **Workflow timeouts**: 30 minutes execution timeout, 25 minutes run timeout
- **Expected duration**: 10-15 minutes for complete research

## Components

### Agents

- **Exploration Orchestrator**: Identifies 3-5 key sub-topics for research (exploration mode)
- **Exploitation Orchestrator**: Creates detailed search plans for specific sub-topics (exploitation mode)
- **Search Agent**: Uses web search to gather financial information
- **Financials Agent**: Analyzes company fundamentals (revenue, profit, margins)
- **Risk Agent**: Identifies potential red flags and risk factors
- **Writer Agent**: Synthesizes information from all iterations into a comprehensive report
- **Verifier Agent**: Audits the final report for consistency and accuracy

### Writer Agent Tools

The writer agent has access to tools that invoke the specialist analysts:
- `fundamentals_analysis`: Get financial performance analysis
- `risk_analysis`: Get risk factor assessment

## Temporal Integration

The example demonstrates several Temporal patterns:
- Durable execution of multi-step research workflows
- Parallel execution of web searches using `asyncio.create_task`
- Use of `workflow.as_completed` for handling concurrent tasks
- Proper import handling with `workflow.unsafe.imports_passed_through()`
