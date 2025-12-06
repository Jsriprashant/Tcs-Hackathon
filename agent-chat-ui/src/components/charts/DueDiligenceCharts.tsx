"use client";

import { cn } from "@/lib/utils";
import { DueDiligenceData, isDueDiligenceData } from "./types";
import { HealthGauge } from "./HealthGauge";
import { RiskMeter } from "./RiskMeter";
import { DomainComparisonChart } from "./DomainComparisonChart";
import { FindingsSeverityChart } from "./FindingsSeverityChart";
import { RecommendationBadge } from "./RecommendationBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DueDiligenceChartsProps {
  data: DueDiligenceData;
  className?: string;
}

export function DueDiligenceCharts({ data, className }: DueDiligenceChartsProps) {
  // Validate data structure
  if (!isDueDiligenceData(data)) {
    console.error("Invalid DueDiligenceData structure", data);
    return null;
  }

  const { company, overall, scoring_table, domain_details } = data;

  return (
    <div className={cn("my-6 space-y-6", className)}>
      {/* Header with company name and recommendation */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <CardTitle className="text-lg">
                M&A Due Diligence Report
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                {company.name} ({company.id})
              </p>
            </div>
            <RecommendationBadge recommendation={overall.recommendation} />
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            {overall.domains_analyzed} domain{overall.domains_analyzed !== 1 ? "s" : ""} analyzed
          </p>
        </CardContent>
      </Card>

      {/* Gauges Row */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-center justify-center gap-8">
            <HealthGauge
              score={overall.overall_health_score}
              label="Overall Health"
              size="lg"
            />
            <RiskMeter
              riskScore={overall.overall_risk_score}
              label="Overall Risk"
              size="lg"
            />
          </div>
        </CardContent>
      </Card>

      {/* Domain Comparison Chart */}
      {scoring_table && scoring_table.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <DomainComparisonChart scoringTable={scoring_table} />
          </CardContent>
        </Card>
      )}

      {/* Findings Severity Chart */}
      {domain_details && (
        <Card>
          <CardContent className="pt-6">
            <FindingsSeverityChart domainDetails={domain_details} />
          </CardContent>
        </Card>
      )}

      {/* Key Findings Summary */}
      {scoring_table && scoring_table.some((row) => row.key_findings?.length > 0) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Key Findings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {scoring_table.map((row) => (
                row.key_findings && row.key_findings.length > 0 && (
                  <div key={row.domain}>
                    <h4 className="text-sm font-medium mb-2">{row.domain}</h4>
                    <ul className="space-y-1">
                      {row.key_findings.map((finding, idx) => (
                        <li
                          key={idx}
                          className="text-sm text-muted-foreground flex items-start gap-2"
                        >
                          <span className="text-yellow-500">•</span>
                          <span>{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Export all chart components
export { HealthGauge } from "./HealthGauge";
export { RiskMeter } from "./RiskMeter";
export { DomainComparisonChart } from "./DomainComparisonChart";
export { FindingsSeverityChart } from "./FindingsSeverityChart";
export { RecommendationBadge } from "./RecommendationBadge";
export * from "./types";
