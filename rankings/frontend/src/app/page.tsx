"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import * as d3 from "d3";

// TypeScript interface
interface RankingData {
  team: string;
  mean: number;
  hdi_lower: number;
  hdi_upper: number;
  "Team Rank": number;
  logo_url: string;
}

// Custom hook for responsive dimensions
function useResizeObserver<T extends HTMLElement>(ref: React.RefObject<T | null>) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const { contentRect } of entries) {
        setDimensions({ width: contentRect.width, height: contentRect.height });
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

  const drawChart = (chartData: RankingData[], containerWidth: number) => {
    if (!chartData.length || !svgRef.current) return;

    const sortedData = [...chartData].sort((a, b) => a["Team Rank"] - b["Team Rank"]);
    const width = containerWidth || 900;
    const height = Math.max(400, sortedData.length * 35 + 100);
    const margin = { top: 40, right: 60, bottom: 60, left: 100 };

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("width", "100%")
      .style("height", "auto");

    // Scales
    const xExtent = d3.extent([...sortedData.map(d => d.hdi_lower), ...sortedData.map(d => d.hdi_upper)]) as [number, number];
    const x = d3.scaleLinear().domain(xExtent).nice().range([margin.left, width - margin.right]);

    const y = d3.scaleBand().domain(sortedData.map(d => d.team)).range([margin.top, height - margin.bottom]).padding(0.3);

    const g = svg.append("g");

    // Title
    svg
      .append("text")
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
      .call(
        d3
          .axisLeft(y)
          .tickSize(0)
          .tickFormat(d => {
            const team = sortedData.find(t => t.team === d);
            return team ? `#${team["Team Rank"]}` : "";
          })
      )
      .select(".domain")
      .remove();

    // HDI and mean points
    const teamGroups = g
      .selectAll(".team-group")
      .data(sortedData)
      .enter()
      .append("g")
      .attr("class", "team-group");

    // HDI lines
    teamGroups
      .append("line")
      .attr("class", "hdi-line")
      .attr("x1", d => x(d.hdi_lower))
      .attr("x2", d => x(d.hdi_upper))
      .attr("y1", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("y2", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("stroke", "#f7c267")
      .attr("stroke-width", 3)
      .attr("opacity", 0.7);

    // Mean points
    teamGroups
      .append("circle")
      .attr("class", "mean-point")
      .attr("cx", d => x(d.mean))
      .attr("cy", d => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("r", 6)
      .attr("fill", "#cd0f0fff")
      .attr("stroke", "white")
      .attr("stroke-width", 2);

    // Team logos
    g.selectAll(".team-logo")
      .data(sortedData)
      .enter()
      .append("image")
      .attr("class", "team-logo")
      .attr("x", margin.left - 45)
      .attr("y", d => (y(d.team) || 0) + y.bandwidth() / 2 - 15)
      .attr("width", 30)
      .attr("height", 30)
      .attr("href", d => d.logo_url)
      .on("error", function (event, d) {
        const parent = this.parentNode;
        if (parent && parent instanceof SVGElement) {
          d3.select(parent)
            .append("text")
            .attr("x", margin.left - 25)
            .attr("y", (y(d.team) || 0) + y.bandwidth() / 2)
            .attr("dy", "0.35em")
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .text(d.team);
          d3.select(this).remove();
        }
      });

    // Hover tooltip
    const tooltip = d3.select("body").append("div").attr("id", "tooltip").style("position", "absolute").style("padding", "6px 10px").style("background", "rgba(0,0,0,0.8)").style("color", "#fff").style("border-radius", "4px").style("pointer-events", "none").style("opacity", 0);

    teamGroups
      .on("mouseover", function (event, d) {
        d3.select(this).select(".mean-point").transition().duration(200).attr("r", 8);
        d3.select(this).select(".hdi-line").transition().duration(200).attr("stroke-width", 4).attr("opacity", 1);
        tooltip.style("opacity", 1).html(`${d.team} - Rank: #${d["Team Rank"]}`).style("left", event.pageX + 10 + "px").style("top", event.pageY - 20 + "px");
      })
      .on("mouseout", function () {
        d3.select(this).select(".mean-point").transition().duration(200).attr("r", 6);
        d3.select(this).select(".hdi-line").transition().duration(200).attr("stroke-width", 3).attr("opacity", 0.7);
        tooltip.style("opacity", 0);
      });
  };

  // Redraw chart on data, season, or dimensions change
  useEffect(() => {
    const width = dimensions?.width || 900;
    drawChart(data, width);
  }, [data, season, dimensions]);

  // Fetch data
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/fit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season }),
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-white-900 mb-4">NFL Power Rankings</h1>

      <div className="flex items-center gap-4 mb-4 flex-wrap">
        <label htmlFor="season">Season:</label>
        <input
          id="season"
          type="number"
          min="1999"
          max="2025"
          value={season}
          onChange={(e) => setSeason(parseInt(e.target.value))}
          className="px-3 py-1 border rounded"
        />
        <button onClick={fetchData} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">
          {loading ? "Loading..." : "Load Rankings"}
        </button>
      </div>

      {error && <div className="text-red-600 mb-4">{error}</div>}

      <div ref={wrapperRef} className="bg-black p-4 rounded overflow-x-auto">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
}
