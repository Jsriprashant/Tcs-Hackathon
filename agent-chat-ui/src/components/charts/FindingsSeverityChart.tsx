"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { cn } from "@/lib/utils";
import { DomainDetails, Finding, RISK_COLORS } from "./types";

interface FindingsSeverityChartProps {
  domainDetails: DomainDetails;
  className?: string;
}

interface SeverityCount {
  domain: string;
  critical: number;
  high: number;
  medium: number;
  low: number;
  total: number;
}

function countFindingsBySeverity(findings: Finding[]): Record<string, number> {
  const counts = { critical: 0, high: 0, medium: 0, low: 0 };
  
  findings.forEach((finding) => {
    const severity = finding.severity.toLowerCase() as keyof typeof counts;
    if (severity in counts) {
      counts[severity]++;
    }
  });
  
  return counts;
}

export function FindingsSeverityChart({ domainDetails, className }: FindingsSeverityChartProps) {
  // Build severity counts per domain
  const chartData: SeverityCount[] = [];
  
  if (domainDetails.finance?.findings) {
    const counts = countFindingsBySeverity(domainDetails.finance.findings);
    chartData.push({
      domain: "Financial",
      critical: counts.critical,
      high: counts.high,
      medium: counts.medium,
      low: counts.low,
      total: Object.values(counts).reduce((a, b) => a + b, 0),
    });
  }
  
  if (domainDetails.legal?.findings) {
    const counts = countFindingsBySeverity(domainDetails.legal.findings);
    chartData.push({
      domain: "Legal",
      critical: counts.critical,
      high: counts.high,
      medium: counts.medium,
      low: counts.low,
      total: Object.values(counts).reduce((a, b) => a + b, 0),
    });
  }
  
  if (domainDetails.hr?.findings) {
    const counts = countFindingsBySeverity(domainDetails.hr.findings);
    chartData.push({
      domain: "HR",
      critical: counts.critical,
      high: counts.high,
      medium: counts.medium,
      low: counts.low,
      total: Object.values(counts).reduce((a, b) => a + b, 0),
    });
  }

  // Don't render if no findings
  if (chartData.length === 0 || chartData.every(d => d.total === 0)) {
    return null;
  }

  return (
    <div className={cn("w-full", className)}>
      <h3 className="mb-4 text-sm font-semibold text-foreground">Findings by Severity</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 5 }}
          >
            <XAxis dataKey="domain" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="rounded-lg border bg-background p-3 shadow-md">
                      <p className="font-semibold">{label}</p>
                      {payload.map((entry) => (
                        <p
                          key={entry.dataKey}
                          className="text-sm capitalize"
                          style={{ color: entry.fill }}
                        >
                          {entry.dataKey}: {entry.value}
                        </p>
                      ))}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend
              iconType="circle"
              iconSize={8}
              wrapperStyle={{ fontSize: "12px" }}
            />
            <Bar
              dataKey="critical"
              stackId="severity"
              fill={RISK_COLORS.critical}
              name="Critical"
              radius={[0, 0, 0, 0]}
            />
            <Bar
              dataKey="high"
              stackId="severity"
              fill={RISK_COLORS.high}
              name="High"
              radius={[0, 0, 0, 0]}
            />
            <Bar
              dataKey="medium"
              stackId="severity"
              fill={RISK_COLORS.medium}
              name="Medium"
              radius={[0, 0, 0, 0]}
            />
            <Bar
              dataKey="low"
              stackId="severity"
              fill={RISK_COLORS.low}
              name="Low"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
