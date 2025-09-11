"use client";
import { useState, useEffect, useRef } from "react";
import * as d3 from "d3";

interface RankingData {
  team: string;
  mean: number;
  hdi_lower: number;
  hdi_upper: number;
  'Team Rank': number;
  logo_url: string;
}

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
  const [season, setSeason] = useState(1999);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dimensions = useResizeObserver(wrapperRef);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/seasons_rankings.json");
      const json = await res.json();
      const seasonData = json[`${season} Season`];

      if (seasonData) {
        const rankingData: RankingData[] = seasonData.team.map((team: string, i: number) => ({
          team,
          mean: seasonData.mean[i],
          hdi_lower: seasonData.hdi_lower[i],
          hdi_upper: seasonData.hdi_upper[i],
          'Team Rank': seasonData["Team Rank"][i],
          logo_url: seasonData.logo_url[i],
        }));
        setData(rankingData);
      } else {
        const apiRes = await fetch("/api/fit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ season }),
        });
        const apiData = await apiRes.json();
        setData(apiData);
      }
    } catch (err: any) {
      console.error("Failed to fetch rankings:", err);
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [season]);

  useEffect(() => {
    if (data.length && dimensions) drawChart(data, dimensions.width);
  }, [data, dimensions]);

  const drawChart = (chartData: RankingData[], width: number) => {
    if (!svgRef.current) return;

    const sortedData = [...chartData].sort((a, b) => a["Team Rank"] - b["Team Rank"]);
    const height = Math.max(400, sortedData.length * 35 + 100);
    const margin = { top: 40, right: 60, bottom: 60, left: 80 };

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("width", "100%")
      .style("height", "auto");

    const x = d3
      .scaleLinear()
      .domain([d3.min(sortedData, (d) => d.hdi_lower)!, d3.max(sortedData, (d) => d.hdi_upper)!])
      .nice()
      .range([margin.left, width - margin.right]);

    const y = d3
      .scaleBand()
      .domain(sortedData.map((d) => d.team))
      .range([margin.top, height - margin.bottom])
      .padding(0.2);

    const g = svg.append("g");

    // Title
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .style("font-size", "18px")
      .style("font-weight", "bold")
      .text(`NFL Power Rankings ${season}`);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickFormat(d3.format(".2f")));

    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickSize(0))
      .select(".domain")
      .remove();

    // HDI lines
    g.selectAll(".team-line")
      .data(sortedData)
      .enter()
      .append("line")
      .attr("x1", (d) => x(d.hdi_lower))
      .attr("x2", (d) => x(d.hdi_upper))
      .attr("y1", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("y2", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("stroke", "#f7c267")
      .attr("stroke-width", 3)
      .attr("opacity", 0.7);

    // Mean points
    g.selectAll(".team-circle")
      .data(sortedData)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.mean))
      .attr("cy", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("r", 6)
      .attr("fill", "#cd0f0fff")
      .attr("stroke", "white")
      .attr("stroke-width", 2);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto" ref={wrapperRef}>
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
            Points represent mean skill estimates from a Bradley-Terry Model, lines show 97% High Density Intervals.
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
                    <img
                      src={team.logo_url}
                      alt={`${team.team} logo`}
                      className="w-12 h-12 mb-2"
                      onError={(e) => (e.currentTarget.style.display = "none")}
                    />
                    <div className="font-bold text-2xl text-red-800">{team.team}</div>
                  </div>
                  <div className="font-semibold text-red-800 mb-1">{team.team}</div>
                  <div className="text-sm text-red-600">Est Skill: {team.mean.toFixed(3)}</div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
