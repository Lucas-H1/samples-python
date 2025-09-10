#!/usr/bin/env python3

import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

from openai_agents.financial_deep_research_agent.workflows.financial_deep_research_workflow import (
    FinancialDeepResearchWorkflow,
)


async def main():
    # Get the query from user input
    query = input("Enter a financial deep research query: ")
    if not query.strip():
        query = "Write up an analysis of Apple Inc.'s most recent quarter."
        print(f"Using default query: {query}")

    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    # Extended timeouts for iterative research
                    start_to_close_timeout=timedelta(minutes=10),
                    schedule_to_start_timeout=timedelta(minutes=5),
                    heartbeat_timeout=timedelta(minutes=5),
                )
            ),
        ],
    )

    print(f"Starting financial deep research for: {query}")
    print("This may take 10-15 minutes to complete due to iterative research...\n")

    result = await client.execute_workflow(
        FinancialDeepResearchWorkflow.run,
        query,
        id=f"financial-deep-research-{hash(query)}",
        task_queue="financial-deep-research-task-queue",
        # Extended workflow execution timeout
        execution_timeout=timedelta(minutes=30),
        # Allow longer run timeout for the entire workflow
        run_timeout=timedelta(minutes=25),
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
