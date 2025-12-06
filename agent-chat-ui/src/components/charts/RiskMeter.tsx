"use client";

import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from "recharts";
import { cn } from "@/lib/utils";

interface RiskMeterProps {
  riskScore: number; // 0-1 scale
  label?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

function getRiskColor(riskScore: number): string {
  if (riskScore <= 0.3) return "#22c55e"; // green-500 - Low risk
  if (riskScore <= 0.5) return "#eab308"; // yellow-500 - Medium risk
  if (riskScore <= 0.7) return "#f97316"; // orange-500 - High risk
  return "#ef4444"; // red-500 - Critical risk
}

function getRiskLabel(riskScore: number): string {
  if (riskScore <= 0.3) return "Low Risk";
  if (riskScore <= 0.5) return "Moderate";
  if (riskScore <= 0.7) return "High Risk";
  return "Critical";
}

const SIZE_CONFIG = {
  sm: { width: 120, height: 120, fontSize: "text-lg", labelSize: "text-xs" },
  md: { width: 160, height: 160, fontSize: "text-2xl", labelSize: "text-sm" },
  lg: { width: 200, height: 200, fontSize: "text-3xl", labelSize: "text-base" },
};

export function RiskMeter({ riskScore, label = "Risk Score", size = "md", className }: RiskMeterProps) {
  const config = SIZE_CONFIG[size];
  const color = getRiskColor(riskScore);
  const riskLabel = getRiskLabel(riskScore);
  const displayValue = Math.round(riskScore * 100);
  
  const data = [
    {
      name: "risk",
      value: displayValue,
      fill: color,
    },
  ];

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <div style={{ width: config.width, height: config.height }} className="relative">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="70%"
            outerRadius="100%"
            barSize={12}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background={{ fill: "#e5e7eb" }}
              dataKey="value"
              cornerRadius={6}
              angleAxisId={0}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("font-bold", config.fontSize)} style={{ color }}>
            {displayValue}%
          </span>
          <span className={cn("text-muted-foreground", config.labelSize)}>
            {riskLabel}
          </span>
        </div>
      </div>
      
      <span className="mt-2 text-sm font-medium text-muted-foreground">{label}</span>
    </div>
  );
}
