from temporalio import workflow

from openai_agents.financial_deep_research_agent.financial_deep_research_manager import (
    FinancialDeepResearchManager,
)


@workflow.defn
class FinancialDeepResearchWorkflow:
    @workflow.run
    async def run(self, query: str) -> str:
        manager = FinancialDeepResearchManager()
        return await manager.run(query)
