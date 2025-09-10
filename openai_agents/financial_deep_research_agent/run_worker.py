#!/usr/bin/env python3

import asyncio

from temporalio.client import Client
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from temporalio.worker import Worker

from openai_agents.financial_deep_research_agent.workflows.financial_deep_research_workflow import (
    FinancialDeepResearchWorkflow,
)


async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(),
        ],
    )

    worker = Worker(
        client,
        task_queue="financial-deep-research-task-queue",
        workflows=[FinancialDeepResearchWorkflow],
    )

    print("Starting financial deepresearch worker...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
