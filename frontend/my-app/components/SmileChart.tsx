import React from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Layout, Data } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface SmileChartProps {
  data: {
    strikes: number[];
    ivs: number[];
    expiry: string;
  }[];
  isLoading?: boolean;
}

export function SmileChart({ data, isLoading }: SmileChartProps) {
  if (isLoading) {
    return (
      <Card className="w-full h-[400px] flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">
          Loading Smile Data...
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card className="w-full h-[400px] flex items-center justify-center">
        <div className="text-muted-foreground">No smile data available</div>
      </Card>
    );
  }

  // Create traces for each expiry slice
  const traces: Data[] = data.map((slice) => ({
    type: "scatter",
    mode: "lines+markers",
    name: slice.expiry,
    x: slice.strikes,
    y: slice.ivs,
    line: { shape: "spline", width: 2 },
    marker: { size: 4 },
  }));

  // Plotly.js layout type definition is strict about title being an object or string,
  // but sometimes conflicts with React props. We force the type here to be compatible.
  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 40, r: 20, b: 40, t: 20 },
    xaxis: {
      title: { text: "Strike Price" },
      gridcolor: "#333",
      zerolinecolor: "#333",
      color: "#94a3b8",
    },
    yaxis: {
      title: { text: "Implied Volatility" },
      gridcolor: "#333",
      zerolinecolor: "#333",
      color: "#94a3b8",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    legend: {
      font: { color: "#94a3b8" },
      orientation: "h",
      y: 1.1,
    },
    hovermode: "x unified",
  };

  return (
    <Card className="w-full h-full min-h-[400px] border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl font-light tracking-wide text-primary/90">
          Volatility Smiles (By Expiry)
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[350px] w-full p-2">
        <Plot
          data={traces}
          layout={layout}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}
