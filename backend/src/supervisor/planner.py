"""
Analysis Planner for M&A Due Diligence Supervisor.

This module creates execution plans based on classified intent,
determining which agents to invoke and in what order.
"""

from typing import Optional, List, Dict
from datetime import datetime
import uuid

from src.supervisor.models import (
    AnalysisPlan,
    AnalysisScope,
    EnhancedIntentResult,
    DealType,
)
from src.common.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# All available domain agents (each agent calls RAG internally for its data needs)
ALL_AGENTS = [
    "finance_agent",
    "legal_agent",
    "hr_agent",
    "analyst_agent",
]

# Domain to agent mapping
DOMAIN_TO_AGENT = {
    "finance": "finance_agent",
    "legal": "legal_agent",
    "hr": "hr_agent",
    "compliance": "legal_agent",  # Compliance handled by legal agent
    "strategic": "analyst_agent",
}

# Agent dependencies - now empty since each agent handles its own RAG calls
# Domain agents are independent and can run in any order
AGENT_DEPENDENCIES = {
    "finance_agent": [],
    "legal_agent": [],
    "hr_agent": [],
    "analyst_agent": [],
}

# Agents that can run in parallel (all domain agents are now parallel-capable)
PARALLEL_CAPABLE_AGENTS = ["finance_agent", "legal_agent", "hr_agent"]


# =============================================================================
# PLAN TEMPLATES BY SCOPE
# =============================================================================

SCOPE_PLAN_TEMPLATES = {
    # NOTE: Each domain agent handles its own RAG calls internally for data retrieval
    # Supervisor only orchestrates domain agents, not RAG
    
    AnalysisScope.FULL_DUE_DILIGENCE: {
        "required_agents": ["finance_agent", "legal_agent", "hr_agent"],
        "optional_agents": ["analyst_agent"],
        "execution_mode": "parallel",  # All domain agents can run in parallel
        "agent_order": [
            ["finance_agent", "legal_agent", "hr_agent"],     # Phase 1: Parallel domain analysis
            # Phase 2: risk_aggregator + master_analyst (handled by graph, not plan)
        ],
        "require_recommendation": True,
        "report_format": "full",
    },
    AnalysisScope.FINANCIAL_ONLY: {
        "required_agents": ["finance_agent"],
        "optional_agents": [],
        "execution_mode": "sequential",
        "agent_order": [
            ["finance_agent"],
        ],
        "require_recommendation": False,
        "report_format": "summary",
    },
    AnalysisScope.LEGAL_ONLY: {
        "required_agents": ["legal_agent"],
        "optional_agents": [],
        "execution_mode": "sequential",
        "agent_order": [
            ["legal_agent"],
        ],
        "require_recommendation": False,
        "report_format": "summary",
    },
    AnalysisScope.HR_ONLY: {
        "required_agents": ["hr_agent"],
        "optional_agents": [],
        "execution_mode": "sequential",
        "agent_order": [
            ["hr_agent"],
        ],
        "require_recommendation": False,
        "report_format": "summary",
    },
    AnalysisScope.COMPLIANCE_ONLY: {
        "required_agents": ["legal_agent"],  # Legal handles compliance
        "optional_agents": [],
        "execution_mode": "sequential",
        "agent_order": [
            ["legal_agent"],
        ],
        "require_recommendation": False,
        "report_format": "summary",
    },
    AnalysisScope.STRATEGIC_ONLY: {
        "required_agents": ["analyst_agent"],
        "optional_agents": [],
        "execution_mode": "sequential",
        "agent_order": [
            ["analyst_agent"],
        ],
        "require_recommendation": True,
        "report_format": "summary",
    },
    AnalysisScope.RISK_ASSESSMENT: {
        "required_agents": ["finance_agent", "legal_agent", "hr_agent"],
        "optional_agents": [],
        "execution_mode": "parallel",
        "agent_order": [
            ["finance_agent", "legal_agent", "hr_agent"],
        ],
        "require_recommendation": True,
        "report_format": "summary",
    },
    AnalysisScope.COMPARISON: {
        "required_agents": ["finance_agent", "legal_agent", "hr_agent", "analyst_agent"],
        "optional_agents": [],
        "execution_mode": "hybrid",
        "agent_order": [
            ["finance_agent", "legal_agent", "hr_agent"],
            ["analyst_agent"],
        ],
        "require_recommendation": True,
        "report_format": "full",
    },
    AnalysisScope.QUICK_OVERVIEW: {
        "required_agents": ["finance_agent", "legal_agent"],
        "optional_agents": ["hr_agent"],
        "execution_mode": "parallel",
        "agent_order": [
            ["finance_agent", "legal_agent"],
        ],
        "require_recommendation": False,
        "report_format": "executive",
    },
}


# =============================================================================
# PLANNER FUNCTIONS
# =============================================================================

def generate_plan_id() -> str:
    """Generate unique plan identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"plan_{timestamp}_{unique_id}"


def build_dependencies(agent_order: List[List[str]]) -> Dict[str, List[str]]:
    """
    Build dependency map from agent execution order.
    
    Args:
        agent_order: List of agent groups in execution order
        
    Returns:
        Dictionary mapping each agent to its dependencies
    """
    dependencies = {}
    previous_agents = []
    
    for group in agent_order:
        for agent in group:
            # Each agent depends on all agents from previous phases
            dependencies[agent] = previous_agents.copy()
        
        # Add all agents from this group to previous for next phase
        previous_agents.extend(group)
    
    return dependencies


def estimate_duration(agent_order: List[List[str]]) -> int:
    """
    Estimate execution duration in seconds.
    
    Args:
        agent_order: List of agent groups
        
    Returns:
        Estimated duration in seconds
    """
    # Rough estimates per agent type (in seconds)
    AGENT_DURATIONS = {
        "rag_agent": 5,
        "finance_agent": 15,
        "legal_agent": 15,
        "hr_agent": 10,
        "analyst_agent": 20,
    }
    
    total_duration = 0
    for group in agent_order:
        # Parallel execution takes time of longest agent in group
        group_duration = max(AGENT_DURATIONS.get(agent, 10) for agent in group)
        total_duration += group_duration
    
    return total_duration


def create_analysis_plan(intent_result: EnhancedIntentResult) -> AnalysisPlan:
    """
    Create execution plan based on enhanced intent classification.
    
    This is the main planning function that determines:
    - Which agents to invoke
    - Execution order (sequential, parallel, hybrid)
    - Dependencies between agents
    - Output format requirements
    
    Args:
        intent_result: Enhanced intent classification result
        
    Returns:
        AnalysisPlan with execution details
    """
    scope = intent_result.analysis_scope
    
    # Get template for this scope
    template = SCOPE_PLAN_TEMPLATES.get(scope, SCOPE_PLAN_TEMPLATES[AnalysisScope.FULL_DUE_DILIGENCE])
    
    # Adjust based on required domains if specified
    required_agents = template["required_agents"].copy()
    optional_agents = template["optional_agents"].copy()
    agent_order = [group.copy() for group in template["agent_order"]]
    
    # If specific domains were requested, filter agents
    if intent_result.required_domains and scope == AnalysisScope.FULL_DUE_DILIGENCE:
        domain_agents = set()
        domain_agents.add("rag_agent")  # Always include RAG
        
        for domain in intent_result.required_domains:
            if domain in DOMAIN_TO_AGENT:
                domain_agents.add(DOMAIN_TO_AGENT[domain])
        
        # Filter required agents
        required_agents = [a for a in required_agents if a in domain_agents]
        
        # Rebuild agent order
        agent_order = rebuild_agent_order(required_agents)
    
    # Adjust report format based on depth
    report_format = template["report_format"]
    if intent_result.depth == "quick":
        report_format = "executive"
    elif intent_result.depth == "deep":
        report_format = "full"
    
    # Build dependencies
    dependencies = build_dependencies(agent_order)
    
    # Estimate duration
    estimated_duration = estimate_duration(agent_order)
    
    # Create plan
    plan = AnalysisPlan(
        plan_id=generate_plan_id(),
        analysis_scope=scope,
        required_agents=required_agents,
        optional_agents=optional_agents,
        execution_mode=template["execution_mode"],
        agent_order=agent_order,
        dependencies=dependencies,
        require_risk_score=True,
        require_recommendation=template["require_recommendation"],
        report_format=report_format,
        estimated_duration_seconds=estimated_duration,
    )
    
    logger.info(
        f"Created analysis plan: {plan.plan_id}, "
        f"scope={scope.value}, agents={required_agents}, "
        f"mode={template['execution_mode']}"
    )
    
    return plan


def rebuild_agent_order(required_agents: List[str]) -> List[List[str]]:
    """
    Rebuild agent execution order based on required agents.
    
    Args:
        required_agents: List of required agent names
        
    Returns:
        Ordered list of agent groups
    """
    order = []
    
    # Phase 1: RAG always first
    if "rag_agent" in required_agents:
        order.append(["rag_agent"])
    
    # Phase 2: Domain agents (can be parallel)
    domain_agents = [a for a in required_agents if a in PARALLEL_CAPABLE_AGENTS]
    if domain_agents:
        order.append(domain_agents)
    
    # Phase 3: Analyst/consolidation
    if "analyst_agent" in required_agents:
        order.append(["analyst_agent"])
    
    return order


def get_next_agents(
    plan: AnalysisPlan,
    completed_agents: List[str],
    failed_agents: List[str] = None
) -> List[str]:
    """
    Determine which agents should run next based on plan and completion status.
    
    Args:
        plan: The analysis plan
        completed_agents: List of completed agent names
        failed_agents: List of failed agent names
        
    Returns:
        List of agent names to run next (empty if all done)
    """
    if failed_agents is None:
        failed_agents = []
    
    # Find the next phase of agents to run
    for group in plan.agent_order:
        # Check if all agents in this group are done
        group_done = all(
            agent in completed_agents or agent in failed_agents
            for agent in group
        )
        
        if not group_done:
            # Return agents from this group that haven't been run
            return [
                agent for agent in group
                if agent not in completed_agents and agent not in failed_agents
            ]
    
    # All phases complete
    return []


def is_plan_complete(
    plan: AnalysisPlan,
    completed_agents: List[str],
    failed_agents: List[str] = None
) -> bool:
    """
    Check if the analysis plan is complete.
    
    Args:
        plan: The analysis plan
        completed_agents: List of completed agents
        failed_agents: List of failed agents
        
    Returns:
        True if plan is complete
    """
    if failed_agents is None:
        failed_agents = []
    
    # Check if all required agents are done (completed or failed)
    for agent in plan.required_agents:
        if agent not in completed_agents and agent not in failed_agents:
            return False
    
    return True


def get_pending_agents(
    plan: AnalysisPlan,
    completed_agents: List[str],
    failed_agents: List[str] = None
) -> List[str]:
    """
    Get list of all pending agents.
    
    Args:
        plan: The analysis plan
        completed_agents: List of completed agents
        failed_agents: List of failed agents
        
    Returns:
        List of pending agent names
    """
    if failed_agents is None:
        failed_agents = []
    
    return [
        agent for agent in plan.required_agents
        if agent not in completed_agents and agent not in failed_agents
    ]
