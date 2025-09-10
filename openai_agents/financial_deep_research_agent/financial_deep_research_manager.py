from __future__ import annotations

import asyncio
from collections.abc import Sequence

from agents import RunConfig, Runner, RunResult, custom_span, trace
from temporalio import workflow

from openai_agents.financial_deep_research_agent.agents.financials_agent import (
    new_financials_agent,
)
from openai_agents.financial_deep_research_agent.agents.orchestrator_agent import (
    ExplorationPlan,
    FinancialSearchItem,
    FinancialSearchPlan,
    ResearchMode,
    ResearchSubTopic,
    new_orchestrator_agent,
)
from openai_agents.financial_deep_research_agent.agents.risk_agent import new_risk_agent
from openai_agents.financial_deep_research_agent.agents.search_agent import (
    new_search_agent,
)
from openai_agents.financial_deep_research_agent.agents.verifier_agent import (
    VerificationResult,
    new_verifier_agent,
)
from openai_agents.financial_deep_research_agent.agents.writer_agent import (
    FinancialReportData,
    new_writer_agent,
)


async def _summary_extractor(run_result: RunResult) -> str:
    """Custom output extractor for sub-agents that return an AnalysisSummary."""
    # The financial/risk analyst agents emit an AnalysisSummary with a `summary` field.
    # We want the tool call to return just that summary text so the writer can drop it inline.
    return str(run_result.final_output.summary)


class FinancialDeepResearchManager:
    """
    Orchestrates the full flow using an explore-exploit pattern:
    1. Explore: Identify key sub-topics to research
    2. Exploit: Deeply research each sub-topic
    3. Iterate: Continue until sufficient depth is achieved
    """

    def __init__(self, max_iterations: int = 3, max_depth_per_topic: int = 2) -> None:
        self.run_config = RunConfig()
        self.max_iterations = max_iterations
        self.max_depth_per_topic = max_depth_per_topic

        # Create agents for different modes
        self.explore_orchestrator = new_orchestrator_agent(ResearchMode.EXPLORE)
        self.exploit_orchestrator = new_orchestrator_agent(ResearchMode.EXPLOIT)
        self.search_agent = new_search_agent()
        self.financials_agent = new_financials_agent()
        self.risk_agent = new_risk_agent()
        self.writer_agent = new_writer_agent()
        self.verifier_agent = new_verifier_agent()

    async def run(self, query: str) -> str:
        with trace("Financial deep research trace"):
            # Phase 1: Initial exploration to identify sub-topics
            exploration_plan = await self._explore_topics(query)

            # Phase 2: Iterative exploitation of each sub-topic
            all_search_results = []
            research_summary = []

            for iteration in range(self.max_iterations):
                with custom_span(f"Research iteration {iteration + 1}"):
                    iteration_results = await self._exploit_subtopics(
                        query, exploration_plan.sub_topics, iteration, research_summary
                    )
                    all_search_results.extend(iteration_results["search_results"])
                    research_summary.append(iteration_results["summary"])

                    # Check if we should continue iterating
                    if not self._should_continue_iteration(
                        iteration_results, iteration
                    ):
                        break

            # Phase 3: Synthesize all research into final report
            report = await self._write_comprehensive_report(
                query, all_search_results, research_summary
            )
            verification = await self._verify_report(report)

        # Return formatted output
        result = f"""=====RESEARCH SUMMARY=====

{self._format_research_summary(research_summary)}

=====FINAL REPORT=====

{report.markdown_report}

=====FOLLOW UP QUESTIONS=====

{chr(10).join(report.follow_up_questions)}

=====VERIFICATION=====

Verified: {verification.verified}
Issues: {verification.issues}"""

        return result

    async def _explore_topics(self, query: str) -> ExplorationPlan:
        """Phase 1: Explore and identify key sub-topics to research."""
        result = await Runner.run(
            self.explore_orchestrator,
            f"Query: {query}",
            run_config=self.run_config,
        )
        return result.final_output_as(ExplorationPlan)

    async def _perform_searches(
        self, search_plan: FinancialSearchPlan
    ) -> Sequence[str]:
        with custom_span("Search the web"):
            tasks = [
                asyncio.create_task(self._search(item)) for item in search_plan.searches
            ]
            results: list[str] = []
            for task in workflow.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
            return results

    async def _search(self, item: FinancialSearchItem) -> str | None:
        input_data = f"Search term: {item.query}\nReason: {item.reason}"
        try:
            result = await Runner.run(
                self.search_agent,
                input_data,
                run_config=self.run_config,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(
        self, query: str, search_results: Sequence[str]
    ) -> FinancialReportData:
        # Expose the specialist analysts as tools so the writer can invoke them inline
        # and still produce the final FinancialReportData output.
        fundamentals_tool = self.financials_agent.as_tool(
            tool_name="fundamentals_analysis",
            tool_description="Use to get a short write-up of key financial metrics",
            custom_output_extractor=_summary_extractor,
        )
        risk_tool = self.risk_agent.as_tool(
            tool_name="risk_analysis",
            tool_description="Use to get a short write-up of potential red flags",
            custom_output_extractor=_summary_extractor,
        )
        writer_with_tools = self.writer_agent.clone(
            tools=[fundamentals_tool, risk_tool]
        )

        input_data = (
            f"Original query: {query}\nSummarized search results: {search_results}"
        )
        result = await Runner.run(
            writer_with_tools,
            input_data,
            run_config=self.run_config,
        )
        return result.final_output_as(FinancialReportData)

    async def _verify_report(self, report: FinancialReportData) -> VerificationResult:
        result = await Runner.run(
            self.verifier_agent,
            report.markdown_report,
            run_config=self.run_config,
        )
        return result.final_output_as(VerificationResult)

    async def _exploit_subtopics(
        self,
        query: str,
        sub_topics: list[ResearchSubTopic],
        iteration: int,
        research_summary: list[str],
    ) -> dict:
        """Phase 2: Deeply research each sub-topic in exploitation mode."""
        iteration_results = {
            "search_results": [],
            "summary": f"Iteration {iteration + 1} Results:",
        }

        # Sort sub-topics by priority
        sorted_topics = sorted(sub_topics, key=lambda x: x.priority)

        for topic in sorted_topics:
            with custom_span(f"Researching sub-topic: {topic.name}"):
                # Create exploitation plan for this specific sub-topic
                exploit_plan = await self._create_exploit_plan(
                    query, topic, iteration, research_summary
                )

                # Perform searches for this sub-topic
                topic_results = await self._perform_searches(exploit_plan)
                iteration_results["search_results"].extend(topic_results)

                # Add to summary with more detail
                iteration_results[
                    "summary"
                ] += f"\n- {topic.name}: {len(topic_results)} search results"

                # Add brief summary of what was found (first 200 chars of first result)
                if topic_results:
                    first_result_preview = (
                        topic_results[0][:200] + "..."
                        if len(topic_results[0]) > 200
                        else topic_results[0]
                    )
                    iteration_results[
                        "summary"
                    ] += f"\n  Preview: {first_result_preview}"

        return iteration_results

    async def _create_exploit_plan(
        self,
        query: str,
        sub_topic: ResearchSubTopic,
        iteration: int,
        research_summary: list[str],
    ) -> FinancialSearchPlan:
        """Create a detailed search plan for a specific sub-topic."""
        # Truncate research summary if too long (keep last 2 iterations)
        truncated_summary = self._truncate_research_summary(
            research_summary, max_iterations=2
        )

        input_prompt = f"""
        Original Query: {query}
        Sub-topic to research: {sub_topic.name}
        Sub-topic description: {sub_topic.description}
        Research iteration: {iteration + 1}
        
        Previous Research Summary:
        {truncated_summary}
        
        Based on the previous research findings, create detailed searches to:
        1. Fill any gaps in the current research for this sub-topic
        2. Deepen understanding of key findings from previous iterations
        3. Explore new angles or recent developments not yet covered
        4. Verify or contradict previous findings if needed
        
        Focus on searches that will add new value beyond what has already been researched.
        """

        result = await Runner.run(
            self.exploit_orchestrator,
            input_prompt,
            run_config=self.run_config,
        )

        search_plan = result.final_output_as(FinancialSearchPlan)
        search_plan.sub_topic = sub_topic.name
        return search_plan

    def _should_continue_iteration(
        self, iteration_results: dict, iteration: int
    ) -> bool:
        """Determine if we should continue with more iterations."""
        # Stop if we've reached max iterations
        if iteration >= self.max_iterations - 1:
            return False

        # Stop if we didn't get many new results (diminishing returns)
        search_count = len(iteration_results["search_results"])
        if search_count < 3:  # Very few new results
            return False

        # Continue if we're still in early iterations and getting good results
        return iteration < 2

    async def _write_comprehensive_report(
        self, query: str, all_search_results: list[str], research_summary: list[str]
    ) -> FinancialReportData:
        """Write a comprehensive report using all research from all iterations."""
        # Expose the specialist analysts as tools
        fundamentals_tool = self.financials_agent.as_tool(
            tool_name="fundamentals_analysis",
            tool_description="Use to get a short write-up of key financial metrics",
            custom_output_extractor=_summary_extractor,
        )
        risk_tool = self.risk_agent.as_tool(
            tool_name="risk_analysis",
            tool_description="Use to get a short write-up of potential red flags",
            custom_output_extractor=_summary_extractor,
        )
        writer_with_tools = self.writer_agent.clone(
            tools=[fundamentals_tool, risk_tool]
        )

        # Create comprehensive input with all research
        research_context = "\n\n".join(research_summary)
        input_data = f"""
        Original query: {query}
        
        Research Summary:
        {research_context}
        
        All search results from {len(research_summary)} research iterations:
        {all_search_results}
        
        Please synthesize all this information into a comprehensive financial report.
        """

        result = await Runner.run(
            writer_with_tools,
            input_data,
            run_config=self.run_config,
        )
        return result.final_output_as(FinancialReportData)

    def _truncate_research_summary(
        self, research_summary: list[str], max_iterations: int = 2
    ) -> str:
        """Truncate research summary to keep only the most recent iterations."""
        if not research_summary:
            return "No previous research available."

        # Keep only the last max_iterations
        recent_summaries = (
            research_summary[-max_iterations:]
            if len(research_summary) > max_iterations
            else research_summary
        )

        formatted = "Recent Research Findings:\n"
        for i, summary in enumerate(
            recent_summaries, len(research_summary) - len(recent_summaries) + 1
        ):
            formatted += f"\nIteration {i}:\n{summary}\n"

        return formatted

    def _format_research_summary(self, research_summary: list[str]) -> str:
        """Format the research summary for the final output."""
        if not research_summary:
            return "No research iterations completed."

        formatted = "Research Process:\n"
        for i, summary in enumerate(research_summary, 1):
            formatted += f"\nIteration {i}:\n{summary}\n"

        return formatted
