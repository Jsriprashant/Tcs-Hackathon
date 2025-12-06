"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";
import { cn } from "@/lib/utils";
import { ScoringTableRow, DOMAIN_COLORS, STATUS_COLORS } from "./types";

interface DomainComparisonChartProps {
  scoringTable: ScoringTableRow[];
  className?: string;
}

export function DomainComparisonChart({ scoringTable, className }: DomainComparisonChartProps) {
  // Transform data for the chart
  const chartData = scoringTable.map((row) => ({
    domain: row.domain,
    score: row.score,
    maxScore: row.max_score,
    riskScore: Math.round(row.risk_score * 100),
    status: row.status,
    color: STATUS_COLORS[row.color] || STATUS_COLORS.yellow,
    domainColor: DOMAIN_COLORS[row.domain as keyof typeof DOMAIN_COLORS] || "#6b7280",
    confidence: Math.round(row.confidence * 100),
  }));

  return (
    <div className={cn("w-full", className)}>
      <h3 className="mb-4 text-sm font-semibold text-foreground">Domain Analysis Scores</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
          >
            <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}`} />
            <YAxis 
              type="category" 
              dataKey="domain" 
              tick={{ fontSize: 12 }}
              width={60}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="rounded-lg border bg-background p-3 shadow-md">
                      <p className="font-semibold">{data.domain}</p>
                      <p className="text-sm">Score: {data.score}/{data.maxScore}</p>
                      <p className="text-sm">Risk: {data.riskScore}%</p>
                      <p className="text-sm">Status: {data.status}</p>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {data.confidence}%
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
              <LabelList
                dataKey="score"
                position="right"
                formatter={(value: number) => `${value}/100`}
                className="fill-foreground text-xs"
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend / Status indicators */}
      <div className="mt-4 flex flex-wrap gap-4">
        {chartData.map((row) => (
          <div key={row.domain} className="flex items-center gap-2">
            <div
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: row.color }}
            />
            <span className="text-xs text-muted-foreground">
              {row.domain}: {row.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
