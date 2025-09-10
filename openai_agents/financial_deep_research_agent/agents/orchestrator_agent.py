from agents import Agent
from pydantic import BaseModel
from enum import Enum


class ResearchMode(str, Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


# Generate a plan of searches to ground the financial analysis.
# For a given financial question or company, we want to search for
# recent news, official filings, analyst commentary, and other
# relevant background.
EXPLORE_PROMPT = (
    "You are a financial deep research orchestrator in EXPLORATION mode. Given a request for financial analysis, "
    "identify 3-5 high-level sub-topics or themes that need to be researched. Focus on broad areas that would "
    "provide comprehensive coverage of the topic. Each search should explore a different aspect or angle. "
    "Output between 3 and 5 search terms that represent major research directions."
)

EXPLOIT_PROMPT = (
    "You are a financial deep research orchestrator in EXPLOITATION mode. Given a specific sub-topic or theme "
    "that was identified during exploration, create detailed searches to deeply research this specific area. "
    "You will be provided with previous research findings to help you avoid duplication and build upon existing knowledge. "
    "Focus on recent news, official filings, analyst commentary, and specific data points that add new value. "
    "Avoid searches that would duplicate previous findings. Instead, look for gaps, contradictions, or deeper insights. "
    "Output between 5 and 10 search terms that dive deep into this specific sub-topic while building on previous research."
)


class FinancialSearchItem(BaseModel):
    reason: str
    """Your reasoning for why this search is relevant."""

    query: str
    """The search term to feed into a web (or file) search."""

    priority: int = 1
    """Priority level for this search (1=high, 2=medium, 3=low)."""


class FinancialSearchPlan(BaseModel):
    searches: list[FinancialSearchItem]
    """A list of searches to perform."""

    mode: ResearchMode
    """The research mode used to generate this plan."""

    sub_topic: str | None = None
    """The specific sub-topic being researched (only for exploit mode)."""


class ResearchSubTopic(BaseModel):
    """Represents a sub-topic identified during exploration."""

    name: str
    """The name/title of the sub-topic."""
    description: str
    """Brief description of what this sub-topic covers."""
    priority: int = 1
    """Priority level (1=high, 2=medium, 3=low)."""


class ExplorationPlan(BaseModel):
    """Plan generated during exploration phase."""

    sub_topics: list[ResearchSubTopic]
    """List of sub-topics to research in detail."""
    mode: ResearchMode = ResearchMode.EXPLORE


def new_orchestrator_agent(mode: ResearchMode = ResearchMode.EXPLORE) -> Agent:
    """Create an orchestrator agent for the specified research mode."""
    if mode == ResearchMode.EXPLORE:
        instructions = EXPLORE_PROMPT
        output_type = ExplorationPlan
    else:
        instructions = EXPLOIT_PROMPT
        output_type = FinancialSearchPlan

    return Agent(
        name=f"FinancialOrchestratorAgent_{mode.value}",
        instructions=instructions,
        model="o3-mini",
        output_type=output_type,
    )
