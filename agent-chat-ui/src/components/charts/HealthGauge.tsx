"use client";

import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from "recharts";
import { cn } from "@/lib/utils";

interface HealthGaugeProps {
  score: number; // 0-100
  label?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

function getHealthColor(score: number): string {
  if (score >= 70) return "#22c55e"; // green-500
  if (score >= 40) return "#eab308"; // yellow-500
  return "#ef4444"; // red-500
}

function getHealthLabel(score: number): string {
  if (score >= 70) return "Healthy";
  if (score >= 40) return "Moderate";
  return "At Risk";
}

const SIZE_CONFIG = {
  sm: { width: 120, height: 120, fontSize: "text-xl", labelSize: "text-xs" },
  md: { width: 160, height: 160, fontSize: "text-3xl", labelSize: "text-sm" },
  lg: { width: 200, height: 200, fontSize: "text-4xl", labelSize: "text-base" },
};

export function HealthGauge({ score, label = "Health Score", size = "md", className }: HealthGaugeProps) {
  const config = SIZE_CONFIG[size];
  const color = getHealthColor(score);
  const healthLabel = getHealthLabel(score);
  
  const data = [
    {
      name: "score",
      value: score,
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
            {score}
          </span>
          <span className={cn("text-muted-foreground", config.labelSize)}>
            {healthLabel}
          </span>
        </div>
      </div>
      
      <span className="mt-2 text-sm font-medium text-muted-foreground">{label}</span>
    </div>
  );
}
