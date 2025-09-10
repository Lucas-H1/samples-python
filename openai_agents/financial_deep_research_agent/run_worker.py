#!/usr/bin/env python3

import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.worker import Worker

from openai_agents.financial_deep_research_agent.workflows.financial_deep_research_workflow import (
    FinancialDeepResearchWorkflow,
)


async def main():
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    # Increased timeout for iterative research process
                    start_to_close_timeout=timedelta(minutes=10),
                    # Allow longer schedule-to-start for complex research
                    schedule_to_start_timeout=timedelta(minutes=5),
                    # Longer heartbeat timeout for long-running research
                    heartbeat_timeout=timedelta(minutes=5),
                )
            ),
        ],
    )

    worker = Worker(
        client,
        task_queue="financial-deep-research-task-queue",
        workflows=[FinancialDeepResearchWorkflow],
    )

    print("Starting financial deep research worker with extended timeouts...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
