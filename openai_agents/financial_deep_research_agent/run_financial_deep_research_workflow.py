#!/usr/bin/env python3

import asyncio

from temporalio.client import Client
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin

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
            OpenAIAgentsPlugin(),
        ],
    )

    print(f"Starting financial deep research for: {query}")
    print("This may take several minutes to complete...\n")

    result = await client.execute_workflow(
        FinancialDeepResearchWorkflow.run,
        query,
        id=f"financial-deep-research-{hash(query)}",
        task_queue="financial-deep-research-task-queue",
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
