"use client";

import { cn } from "@/lib/utils";
import { RECOMMENDATION_COLORS } from "./types";

interface RecommendationBadgeProps {
  recommendation: "GO" | "CONDITIONAL" | "CAUTION" | "NO-GO";
  className?: string;
}

const RECOMMENDATION_LABELS = {
  GO: "Proceed with Deal",
  CONDITIONAL: "Proceed with Conditions",
  CAUTION: "Review Required",
  "NO-GO": "Do Not Proceed",
} as const;

const RECOMMENDATION_ICONS = {
  GO: "✓",
  CONDITIONAL: "⚠",
  CAUTION: "⚡",
  "NO-GO": "✕",
} as const;

export function RecommendationBadge({ recommendation, className }: RecommendationBadgeProps) {
  const color = RECOMMENDATION_COLORS[recommendation] || RECOMMENDATION_COLORS.CONDITIONAL;
  const label = RECOMMENDATION_LABELS[recommendation] || recommendation;
  const icon = RECOMMENDATION_ICONS[recommendation] || "•";
  
  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold text-white shadow-sm",
        className
      )}
      style={{ backgroundColor: color }}
    >
      <span className="text-lg">{icon}</span>
      <span>{label}</span>
    </div>
  );
}
