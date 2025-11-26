"use client";

import React, { useState, useEffect } from "react";
import { SurfacePlot } from "@/components/SurfacePlot";
import { SmileChart } from "@/components/SmileChart";
import { StockChart } from "@/components/StockChart";
import { TickerSearch } from "@/components/TickerSearch";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { motion } from "framer-motion";
import { Activity, Calendar as CalendarIcon, TrendingUp } from "lucide-react";
import { API_BASE_URL } from "@/lib/config";

interface DashboardData {
  surface: any;
  smiles: any[];
  candles: any[];
  tickerInfo: {
    price: number;
    change: number;
    ivRank: number;
  };
}

export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [selectedOptionType, setSelectedOptionType] = useState("CALL");
  const [selectedDuration, setSelectedDuration] = useState("1y");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<DashboardData | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/analytics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: selectedTicker,
          optionType: selectedOptionType,
          duration: selectedDuration,
        }),
      });
      
      if (!response.ok) throw new Error("Failed to fetch data");
      
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error("Error fetching analytics:", error);
      // Handle error state visually if needed
    } finally {
      setLoading(false);
    }
  };

  // Fetch data when any parameter changes
  useEffect(() => {
    fetchData();
  }, [selectedTicker, selectedOptionType, selectedDuration]);

  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker);
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-6 space-y-6 font-sans selection:bg-primary/20">
      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8"
      >
        <div>
          <h1 className="text-3xl font-light tracking-tight flex items-center gap-3">
            <div className="h-8 w-1 bg-primary rounded-full" />
            <span className="font-bold text-primary">PETRAL</span>
            <span className="text-muted-foreground">Quant Analytics</span>
          </h1>
          <p className="text-muted-foreground mt-1 ml-4 text-sm">
            Real-time Volatility Surface & Smile Analysis
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 bg-card/30 p-2 rounded-lg border border-border/40 backdrop-blur-md shadow-sm">
          <TickerSearch
            onSelect={handleTickerChange}
            initialValue={selectedTicker}
          />

          <div className="h-8 w-px bg-border mx-1 hidden md:block" />

          <Select
            value={selectedOptionType}
            onValueChange={setSelectedOptionType}
          >
            <SelectTrigger className="w-[100px] border-primary/20 bg-background/50">
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="CALL">Calls</SelectItem>
              <SelectItem value="PUT">Puts</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedDuration} onValueChange={setSelectedDuration}>
            <SelectTrigger className="w-[140px] border-primary/20 bg-background/50">
              <SelectValue placeholder="Duration" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1m">1 Month</SelectItem>
              <SelectItem value="3m">3 Months</SelectItem>
              <SelectItem value="6m">6 Months</SelectItem>
              <SelectItem value="1y">1 Year</SelectItem>
              <SelectItem value="2y">2 Years</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </motion.div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {[
          {
            label: "Spot Price",
            value: data?.tickerInfo.price ? `$${data.tickerInfo.price.toFixed(2)}` : "---",
            icon: TrendingUp,
            color: "text-emerald-400",
          },
          {
            label: "IV Rank",
            value: data?.tickerInfo.ivRank ? `${data.tickerInfo.ivRank}%` : "---",
            icon: Activity,
            color: "text-cornflower-400",
          },
          {
            label: "Term Structure",
            value: "Contango", // Placeholder logic
            icon: CalendarIcon,
            color: "text-purple-400",
          },
        ].map((metric, i) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <Card className="bg-card/40 border-border/40 backdrop-blur-sm hover:bg-card/60 transition-colors">
              <CardContent className="p-6 flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground font-medium">
                    {metric.label}
                  </p>
                  <h3 className="text-2xl font-bold mt-1 tracking-tight">
                    {loading ? "..." : metric.value}
                  </h3>
                </div>
                <div
                  className={`p-3 rounded-full bg-background/50 ${metric.color}`}
                >
                  <metric.icon className="h-6 w-6" />
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="h-[600px]"
        >
          <SurfacePlot data={data?.surface} isLoading={loading} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="h-[600px]"
        >
          {/* Note: Passing empty candles for now as we focus on Options/IV */}
          <StockChart
            data={data?.candles || []}
            ticker={selectedTicker}
            isLoading={loading}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2"
        >
          <SmileChart data={data?.smiles || []} isLoading={loading} />
        </motion.div>
      </div>
    </div>
  );
}
