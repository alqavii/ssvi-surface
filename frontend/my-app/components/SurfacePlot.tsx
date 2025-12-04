import React from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Layout, Data } from "plotly.js";
import { useTheme } from "next-themes";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface SurfacePlotProps {
  data: {
    x: number[];
    y: number[]; // This is assumed to be fractional years now
    z: number[][];
  } | null;
  isLoading?: boolean;
}

export function SurfacePlot({ data, isLoading }: SurfacePlotProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark" || theme === "system";

  if (isLoading) {
    return (
      <Card className="w-full h-[500px] flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">
          Loading Surface Data...
        </div>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="w-full h-[500px] flex items-center justify-center">
        <div className="text-muted-foreground">No data available</div>
      </Card>
    );
  }

  // Transform Y-axis (fractional years) to Days for readability
  // y values are like 0.1, 0.5 etc.
  const yDays = data.y.map((val) => Math.round(val * 365));

  const plotData: Data[] = [
    {
      type: "surface",
      x: data.x,
      y: yDays, // Use Days instead of Years
      z: data.z,
      colorscale: "Jet",
      showscale: true,
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: "#424242",
          project: { z: true },
        },
      } as any,
    },
  ];

  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "#334155" : "#e2e8f0";

  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: {
      aspectratio: { x: 1, y: 1, z: 0.6 },
      xaxis: {
        title: { text: "Strike" },
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        showbackground: false,
        backgroundcolor: "rgba(0,0,0,0)",
        color: textColor,
      },
      yaxis: {
        title: { text: "Expiry (Days)" }, // Updated Title
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        showbackground: false,
        backgroundcolor: "rgba(0,0,0,0)",
        color: textColor,
      },
      zaxis: {
        title: { text: "Implied Volatility" },
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        showbackground: false,
        backgroundcolor: "rgba(0,0,0,0)",
        color: textColor,
      },
      camera: {
        eye: { x: 1.4, y: 1.4, z: 0.8 },
        center: { x: 0, y: 0, z: -0.1 },
      },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      color: textColor,
    },
  };

  return (
    <Card className="w-full h-full min-h-[600px] border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl font-light tracking-wide text-primary/90">
          Implied Volatility Surface
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[500px] w-full p-0">
        <Plot
          data={plotData}
          layout={layout}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}
