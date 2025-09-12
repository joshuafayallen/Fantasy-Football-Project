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
  const [season, setSeason] = useState(2024);
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
      let rankingData: RankingData[] = [];


      if (seasonData) {
         rankingData = seasonData.team.map((team: string, i: number) => ({
          team,
          mean: seasonData.mean[i],
          hdi_lower: seasonData.hdi_lower[i],
          hdi_upper: seasonData.hdi_upper[i],
          'Team Rank': seasonData["Team Rank"][i],
          logo_url: seasonData.logo_url[i],
        }));
      } else {
      console.log("=== FRONTEND API CALL START ===");
      console.log("Making API call to:", `${process.env.NEXT_PUBLIC_API_URL}/fit`);
      console.log("Request body:", JSON.stringify({ season }));
      console.log("Season value:", season);
      console.log("Season type:", typeof season);
        const apiRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/fit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ season }),
        });

      const apiData = await apiRes.json();
      console.log("=== API RESPONSE ANALYSIS ===");
console.log("API Response data:", apiData);
console.log("API Response keys:", Object.keys(apiData));
console.log("apiData.team exists:", 'team' in apiData);
console.log("apiData.team type:", typeof apiData.team);
console.log("apiData.team is array:", Array.isArray(apiData.team));
if (apiData.team) {
  console.log("apiData.team length:", apiData.team.length);
  console.log("First few teams:", apiData.team.slice(0, 3));
}

// Safety check
  if (!Array.isArray(apiData)) {
    console.error("❌ API response is not an array:", apiData);
    throw new Error("API response should be an array of team objects");
  }
  
console.log("✅ API response structure looks good, proceeding with mapping...");

    rankingData = apiData.map((teamData) => ({
    team: teamData.team,
    mean: teamData.mean,
    hdi_lower: teamData.hdi_lower,
    hdi_upper: teamData.hdi_upper,
    'Team Rank': teamData['Team Rank'],
    logo_url: teamData.logo_url,
  }));
      }
      setData(rankingData);
    } catch (err: unknown) {
  console.error("Failed to fetch rankings:", err);

  if (err instanceof Error) {
    setError(err.message);
  } else {
    setError("Unknown error");
  }
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
      .style("font-size", "24px")
      .style("font-weight", "bold")
      .style("fill", "white")
      .text(`NFL Power Rankings ${season}`);

    // Axes
    const xAxis = g
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickFormat(d3.format(".2f")));
    
    xAxis.selectAll('text')
    .style("font-size", "14px")

    const yAxis = g.append("g")
  .attr("transform", `translate(${margin.left},0)`)
  .call(d3.axisLeft(y).tickSize(0));
  yAxis.select(".domain").remove();
  yAxis.selectAll("text").remove();

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
      .attr("fill", "#fbfbfbff")
      .attr("stroke", "white")
      .attr("stroke-width", 2);


  const yAxisGroup = g.append("g").attr("class", "y-axis-custom");

  sortedData.forEach((d) => {
    const yPos = (y(d.team) || 0) + y.bandwidth() / 2;
    
    // Create a group for each team's y-axis elements
    const teamGroup = yAxisGroup.append("g");
    
    // Add rank text

    teamGroup
      .append("text")
      .attr("x", 15) // Position for rank
      .attr("y", yPos)
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("font-size", "18px")
      .style("font-weight", "bold")
      .style("fill", "#f0f0f0ff")
      .text(`${d["Team Rank"]}`);
    
    // Add team logo
    teamGroup
      .append("image")
      .attr("x", 30) // Position for logo (after rank)
      .attr("y", yPos - 12) // Center the logo vertically
      .attr("width", 24)
      .attr("height", 24)
      .attr("href", d.logo_url)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .on("error", function() {
        // If image fails to load, add team abbreviation as fallback
        d3.select(this.parentNode)
          .append("text")
          .attr("x", 42) // Same x as logo center
          .attr("y", yPos)
          .attr("text-anchor", "middle")
          .attr("dy", "0.35em")
          .style("font-size", "10px")
          .style("font-weight", "bold")
          .style("fill", "#666")
          .text(d.team);
      });
    
    // Add team abbreviation next to logo for clarity
    teamGroup
      .append("text")
      .attr("x", 58) // Position after logo
      .attr("y", yPos)
      .attr("dy", "0.35em")
      .style("font-size", "18px")
      .style("fill", "#f7f7f7ff")
      .text(d.team);
  });

  // Add a vertical line to separate y-axis from chart area
  
};


  return (
  <div ref={wrapperRef} className="min-h-screen bg-background text-foreground">
    <div className="p-6 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <header>
        <h1 className="text-3xl font-bold mb-2">
          Josh Allen&apos;s Total Correct NFL Power Rankings
        </h1>
        <p className="text-sm text-white-400">
          Points represent mean skill estimates from a Bradley-Terry Model, lines show 97% High Density Intervals.
        </p>
      </header>

      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <label htmlFor="season" className="text-sm font-medium">
            Season:
          </label>
          <input
            id="season"
            type="number"
            min="1999"
            max="2025"
            value={season}
            onChange={(e) => setSeason(parseInt(e.target.value))}
            className="px-3 py-1 bg-gray-900 border border-gray-700 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <button
          onClick={fetchData}
          disabled={loading}
          className="px-4 py-2 rounded bg-purple-400 text-white font-medium shadow hover:bg-blue-700 disabled:bg-purple-400 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Loading..." : "Load Rankings"}
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500 text-red-400 px-4 py-3 rounded">
          Error: {error}
        </div>
      )}

      {/* Chart */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 overflow-x-auto shadow">
        <svg ref={svgRef}></svg>
      </div>

      {/* Top 5 Section */}
      {data.length > 0 && (
        <section className="bg-gray-900 rounded-lg p-4 shadow">
          <h3 className="text-lg font-semibold mb-3">Top 5 Teams</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {data
              .sort((a, b) => b.mean - a.mean)
              .slice(0, 5)
              .map((team, index) => (
                <div
                  key={team.team}
                  className="bg-gray-800 rounded p-4 text-center shadow hover:shadow-lg transition"
                >
                  <div className="text-2xl font-bold text-white-400 mb-2">
                    #{index + 1}
                  </div>
                  <div className="flex flex-col items-center mb-3">
                    <img
                      src={team.logo_url}
                      alt={`${team.team} logo`}
                      className="w-12 h-12 mb-2"
                      onError={(e) => (e.currentTarget.style.display = "none")}
                    />
                    <div className="font-bold text-lg">{team.team}</div>
                  </div>
                  <div className="text-sm text-gray-400">
                    Est Skill: {team.mean.toFixed(3)}
                  </div>
                </div>
              ))}
          </div>
        </section>
      )}
    </div>
  </div>
)};
