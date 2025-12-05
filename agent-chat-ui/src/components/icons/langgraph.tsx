export function LangGraphLogoSVG({
  className,
  width,
  height,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <img
      src="/logo.webp"
      alt="LangGraph logo"
      className={className}
      width={width}
      height={height}
      style={width || height ? { width: width ?? undefined, height: height ?? undefined } : undefined}
    />
  );
}
