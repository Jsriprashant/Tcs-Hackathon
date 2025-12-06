/**
 * Type definitions for Due Diligence Report data
 * Matches the consolidated_result structure from backend's ConsolidatedResult.to_dict()
 */

// Finding from agent analysis
export interface Finding {
  category: string;
  title: string;
  description: string;
  severity: "low" | "medium" | "high" | "critical";
  impact?: string;
  data_points?: string[];
  source?: string;
}

// Risk factor identified by agents
export interface RiskFactor {
  factor_id: string;
  name: string;
  description: string;
  severity: "low" | "medium" | "high" | "critical";
  probability: number;
  impact_score: number;
  is_deal_breaker: boolean;
  mitigation?: string;
}

// Individual domain agent output
export interface AgentOutput {
  agent_name: string;
  domain: string;
  summary: string;
  findings: Finding[];
  key_findings: string[];
  risk_score: number;
  risk_level: "low" | "medium" | "high" | "critical";
  risk_factors: RiskFactor[];
  recommendations: string[];
  red_flags: string[];
  positive_factors: string[];
  confidence: number;
  data_quality: "high" | "medium" | "low";
  documents_analyzed: string[];
  timestamp: string;
  raw_response?: string | null;
}

// Scoring table row for each domain
export interface ScoringTableRow {
  domain: string;
  agent: string;
  score: number;
  max_score: number;
  risk_score: number;
  status: string;
  color: "green" | "yellow" | "orange" | "red";
  key_findings: string[];
  confidence: number;
}

// Overall score summary
export interface OverallScore {
  company_id: string;
  company_name: string;
  overall_health_score: number;
  overall_risk_score: number;
  recommendation: "GO" | "CONDITIONAL" | "CAUTION" | "NO-GO";
  recommendation_color: "green" | "yellow" | "orange" | "red";
  domains_analyzed: number;
}

// Company info
export interface CompanyInfo {
  id: string;
  name: string;
}

// Domain details section
export interface DomainDetails {
  finance: AgentOutput | null;
  legal: AgentOutput | null;
  hr: AgentOutput | null;
}

// Main consolidated result structure
export interface DueDiligenceData {
  company: CompanyInfo;
  overall: OverallScore;
  scoring_table: ScoringTableRow[];
  domain_details: DomainDetails;
}

// Chart color mappings
export const RECOMMENDATION_COLORS = {
  GO: "#22c55e", // green-500
  CONDITIONAL: "#eab308", // yellow-500
  CAUTION: "#f97316", // orange-500
  "NO-GO": "#ef4444", // red-500
} as const;

export const RISK_COLORS = {
  low: "#22c55e", // green-500
  medium: "#eab308", // yellow-500
  high: "#f97316", // orange-500
  critical: "#ef4444", // red-500
} as const;

export const STATUS_COLORS = {
  green: "#22c55e",
  yellow: "#eab308",
  orange: "#f97316",
  red: "#ef4444",
} as const;

export const DOMAIN_COLORS = {
  Financial: "#3b82f6", // blue-500
  Legal: "#8b5cf6", // violet-500
  HR: "#06b6d4", // cyan-500
} as const;

// Helper to check if data is valid DueDiligenceData
export function isDueDiligenceData(data: unknown): data is DueDiligenceData {
  if (typeof data !== "object" || data === null) return false;
  
  const d = data as Record<string, unknown>;
  
  return (
    typeof d.company === "object" &&
    typeof d.overall === "object" &&
    Array.isArray(d.scoring_table) &&
    typeof d.domain_details === "object"
  );
}
