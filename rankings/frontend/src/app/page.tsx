"use client";

import { useState, useEffect, useRef } from "react";
import Image from 'next/image';
import * as d3 from "d3";
// Type definitions for better TypeScript support
interface RankingData {
  team: string;
  mean: number;
  hdi_lower: number;
  hdi_upper: number;
  'Team Rank': number;
  logo_url: string;
  
}

function useResizeObserver(ref: React.RefObject<HTMLElement>) {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
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


  const drawChart = (chartData: RankingData[], containerWidth: number) => {
  if (!chartData.length || !svgRef.current) return;

  // Sort by team rank ascending (rank #1 at top)
  const sortedData = [...chartData].sort((a, b) => a["Team Rank"] - b["Team Rank"]);

  const width = containerWidth || 900;
  const height = Math.max(400, sortedData.length * 35 + 100);
  const margin = { top: 40, right: 60, bottom: 60, left: 80 };

  const svg = d3.select(svgRef.current);

  svg.selectAll("*").remove();

  svg.attr("viewBox", `0 0 ${width} ${height}`)
     .attr("preserveAspectRatio", "xMidYMid meet")
     .style("width", "100%")
     .style("height", "auto");

  const xExtent = d3.extent([
    ...sortedData.map(d => d.hdi_lower),
    ...sortedData.map(d => d.hdi_upper)
  ]) as [number, number];

  const x = d3.scaleLinear()
    .domain(xExtent)
    .nice()
    .range([margin.left, width - margin.right]);

  const y = d3.scaleBand()
    .domain(sortedData.map(d => d.team))
    .range([margin.top, height - margin.bottom])
    .padding(0.2);

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

  // Y-axis: show rank ticks
  g.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y)
      .tickSize(0)
      .tickFormat((d) => {
        const team = sortedData.find(t => t.team === d);
        return team ? `#${team["Team Rank"]}` : "";
      })
    )
    .select(".domain")
    .remove();

  // Add team logos
  g.selectAll(".team-logo")
    .data(sortedData)
    .enter()
    .append("image")
    .attr("class", "team-logo")
    .attr("x", margin.left - 40)
    .attr("y", d => (y(d.team) || 0) + y.bandwidth() / 2 - 15)
    .attr("width", 30)
    .attr("height", 30)
    .attr("href", d => d.logo_url)
    .on("error", function(event, d) {
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

  // HDI lines
  const teamGroups = g.selectAll(".team-group")
    .data(sortedData)
    .enter()
    .append("g")
    .attr("class", "team-group");

  teamGroups.append("line")
    .attr("class", "hdi-line")
    .attr("x1", d => x(d.hdi_lower))
    .attr("x2", d => x(d.hdi_upper))
    .attr("y1", d => (y(d.team) || 0) + y.bandwidth() / 2)
    .attr("y2", d => (y(d.team) || 0) + y.bandwidth() / 2)
    .attr("stroke", "#f7c267")
    .attr("stroke-width", 3)
    .attr("opacity", 0.7);

  // Mean points
  teamGroups.append("circle")
    .attr("class", "mean-point")
    .attr("cx", d => x(d.mean))
    .attr("cy", d => (y(d.team) || 0) + y.bandwidth() / 2)
    .attr("r", 6)
    .attr("fill", "#cd0f0fff")
    .attr("stroke", "white")
    .attr("stroke-width", 2);

  // Hover shows team rank
  teamGroups
    .style("cursor", "pointer")
    .on("mouseover", function(event, d) {
      d3.select(this).select(".mean-point")
        .transition()
        .duration(200)
        .attr("r", 8);

      d3.select(this).select(".hdi-line")
        .transition()
        .duration(200)
        .attr("stroke-width", 4)
        .attr("opacity", 1);

      // Optional: show tooltip with rank
      const tooltip = d3.select("#tooltip");
      if (!tooltip.empty()) {
        tooltip.style("opacity", 1)
               .html(`${d.team} - Rank: #${d["Team Rank"]}`)
               .style("left", (event.pageX + 10) + "px")
               .style("top", (event.pageY - 20) + "px");
      }
    })
    .on("mouseout", function() {
      d3.select(this).select(".mean-point")
        .transition()
        .duration(200)
        .attr("r", 6);

      d3.select(this).select(".hdi-line")
        .transition()
        .duration(200)
        .attr("stroke-width", 3)
        .attr("opacity", 0.7);

      d3.select("#tooltip").style("opacity", 0);
    });
};

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/fit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const json = await res.json();
      console.log("Raw fetched data:", json);
      
      // Log the first few items to understand the structure
      if (Array.isArray(json) && json.length > 0) {
        console.log("First item structure:", json[0]);
        console.log("All keys in first item:", Object.keys(json[0]));
        setData(json);
      } else {
        throw new Error("No data returned from API");
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white-900 mb-4">
          Josh Allen&apos;s Total Correct NFL Power Rankings
        </h1>
        
        <div className="flex items-center gap-4 mb-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label htmlFor="season" className="text-sm font-medium text-white-700">
              Season:
            </label>
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
            onClick={fetchData}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Loading..." : "Load Rankings"}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            Error: {error}
          </div>
        )}

        {data.length > 0 && (
          <div className="text-sm text-white-600 mb-4">
            Points represent mean skill estimates from a Bradley-Terry Model, 
            lines show 97% High Density Intervals.
          </div>
        )}
      </div>

      <div className="bg-black border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <svg ref={svgRef}></svg>
      </div>

      {data.length > 0 && (
        <div className="mt-6 bg-black-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3">Top 5 Teams</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {data
              .sort((a, b) => b.mean - a.mean)
              .slice(0, 5)
              .map((team, index) => (
                <div key={team.team} className="bg-black rounded p-4 text-center shadow-sm">
                  <div className="text-2xl font-bold text-red-600 mb-2">#{index + 1}</div>
                  <div className="flex flex-col items-center mb-3">
                    <Image
                      src={team.logo_url} 
                      alt={`${team.team} logo`}
                      className="w-12 h-12 mb-2"
                      onError={(e) => {
                        // Fallback if logo fails to load
                        e.currentTarget.style.display = 'none';
                        const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                        if (fallback) fallback.style.display = 'block';
                      }}
                    />
                    <div 
                      className="font-bold text-2xl text-red-800 hidden"
                      style={{ display: 'none' }}
                    >
                      {team.team}
                    </div>
                  </div>
                  <div className="font-semibold text-red-800 mb-1">{team.team}</div>
                  <div className="text-sm text-red-600">
                    Est Skill: {team.mean.toFixed(3)}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}