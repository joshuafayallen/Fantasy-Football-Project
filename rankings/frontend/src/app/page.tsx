"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import * as d3 from "d3";

interface RankingData {
  team: string;
  mean: number;
  hdi_lower: number;
  hdi_upper: number;
  "Team Rank": number;
  logo_url: string;
}

function useResizeObserver<T extends HTMLElement>(ref: React.RefObject<T | null>) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const { contentRect } of entries) {
        const { width, height } = contentRect;
        setDimensions({ width, height });
      }
    });

    observer.observe(ref.current);

    return () => observer.disconnect();
  }, [ref]);

  return dimensions;
}

export default function Rankings() {
  const [data, setData] = useState<RankingData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [season, setSeason] = useState(2023);

  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const dimensions = useResizeObserver(wrapperRef);

  const fetchSeasonData = async (season: number) => {
    setLoading(true);
    setError(null);

    try {
      // Try to fetch the local JSON first
      const res = await fetch("/season_rankings.json");
      if (!res.ok) throw new Error("Could not load local rankings JSON");

      const allData = await res.json();
      const seasonKey = `${season} Season`;

      if (seasonKey in allData) {
        const seasonData = allData[seasonKey];

        const formattedData = seasonData.team.map((team: string, i: number) => ({
          team,
          mean: seasonData.mean[i],
          hdi_lower: seasonData.hdi_lower[i],
          hdi_upper: seasonData.hdi_upper[i],
          "Team Rank": seasonData["Team Rank"][i],
          logo_url: seasonData.logo_url[i],
        }));

        setData(formattedData);
        return;
      }

      // If not in JSON, fall back to backend API
      const apiRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/fit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season }),
      });

      if (!apiRes.ok) throw new Error(`API error! status: ${apiRes.status}`);

      const apiData = await apiRes.json();

      setData(apiData);

    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSeasonData(season);
  }, [season]);

  const drawChart = (chartData: RankingData[]) => {
    if (!chartData.length || !svgRef.current || !wrapperRef.current) return;

    const width = wrapperRef.current.clientWidth;
    const sortedData = [...chartData].sort((a, b) => a["Team Rank"] - b["Team Rank"]);
    const height = Math.max(400, sortedData.length * 35 + 100);
    const margin = { top: 40, right: 60, bottom: 60, left: 80 };

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "xMidYMid meet");

    const xExtent = d3.extent([...sortedData.map(d => d.hdi_lower), ...sortedData.map(d => d.hdi_upper)]) as [number, number];
    const x = d3.scaleLinear().domain(xExtent).nice().range([margin.left, width - margin.right]);
    const y = d3.scaleBand().domain(sortedData.map(d => d.team)).range([margin.top, height - margin.bottom]).padding(0.2);

    const g = svg.append("g");

    // Title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .style("font-size", "18px")
      .style("font-weight", "bold")
      .text(`NFL Power Rankings ${season} - Bradley-Terry Model`);

    // X-axis
    g.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickFormat(d3.format(".2f")))
      .selectAll("text")
      .style("font-size", "12px");

    // Y-axis with ranks
    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickSize(0).tickFormat(d => {
        const team = sortedData.find(t => t.team === d);
        return team ? `#${team["Team Rank"]}` : "";
      }))
      .select(".domain")
      .remove();

    // Team logos
    g.selectAll(".team-logo")
      .data(sortedData)
      .enter()
      .append("image")
      .attr("class", "team-logo")
      .attr("x", margin.left - 40)
      .attr("y", d => (y(d.team) || 0) + y.bandwidth() / 2 - 15)
      .attr("width", 30)
      .attr("height", 30)
      .attr("href", d => d.logo_url);

    // HDI lines & mean points
    const teamGroups = g.selectAll(".team-group").data(sortedData).enter().append("g").attr("class", "team-group");

    teamGroups.append("line")
      .attr("class", "hdi-line")
      .attr("x1", d => x(d.hdi_lower))
      .attr("x2", d => x(d.hdi_upper))
      .attr("y1", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("y2", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("stroke", "#f7c267")
      .attr("stroke-width", 3)
      .attr("opacity", 0.7);

    teamGroups.append("circle")
      .attr("class", "mean-point")
      .attr("cx", d => x(d.mean))
      .attr("cy", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("r", 6)
      .attr("fill", "#cd0f0fff")
      .attr("stroke", "white")
      .attr("stroke-width", 2);
  };

  useEffect(() => {
    drawChart(data);
  }, [data, dimensions]);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white-900 mb-4">
          Josh Allen&apos;s Total Correct NFL Power Rankings
        </h1>

        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label htmlFor="season" className="text-sm font-medium text-white-700">Season:</label>
            <input
              id="season"
              type="number"
              min="1999"
              max="2024"
              value={season}
              onChange={(e) => setSeason(parseInt(e.target.value))}
              className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <button
            onClick={() => fetchSeasonData(season)}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Loading..." : "Load Rankings"}
          </button>
        </div>

        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">Error: {error}</div>}
      </div>

      <div ref={wrapperRef} className="bg-black border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
}
