import React from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Layout, Data } from "plotly.js";
import { useTheme } from "next-themes";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface CandleData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface StockChartProps {
  data: CandleData[] | null;
  ticker: string;
  isLoading?: boolean;
}

export function StockChart({ data, ticker, isLoading }: StockChartProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark" || theme === "system";

  if (isLoading) {
    return (
      <Card className="w-full h-[400px] flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">
          Loading Chart Data...
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card className="w-full h-[400px] flex items-center justify-center">
        <div className="text-muted-foreground">No chart data available</div>
      </Card>
    );
  }

  const trace: Data = {
    x: data.map((d) => d.date),
    close: data.map((d) => d.close),
    high: data.map((d) => d.high),
    low: data.map((d) => d.low),
    open: data.map((d) => d.open),

    // Cutomize colors
    increasing: { line: { color: "#10b981" } }, // Emerald-500
    decreasing: { line: { color: "#ef4444" } }, // Red-500

    type: "candlestick",
    xaxis: "x",
    yaxis: "y",
  };

  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "#334155" : "#e2e8f0";

  const layout: Partial<Layout> = {
    dragmode: "zoom",
    autosize: true,
    margin: { l: 50, r: 20, b: 40, t: 20 },
    showlegend: false,
    xaxis: {
      rangeslider: { visible: false },
      title: { text: "Date" },
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      color: textColor,
      type: "date",
    },
    yaxis: {
      autorange: true,
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      color: textColor,
      title: { text: "Price ($)" },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      color: textColor,
    },
  };

  return (
    <Card className="w-full h-full min-h-[400px] border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl font-light tracking-wide text-primary/90">
          {ticker} Price Action
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[350px] w-full p-2">
        <Plot
          data={[trace]}
          layout={layout}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}

