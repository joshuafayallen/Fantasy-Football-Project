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

function useResizeObserver<T extends HTMLElement>(ref: React.RefObject<T>) {
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
  const wrapperRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dimensions = useResizeObserver(wrapperRef);

  useEffect(() => {
    async function fetchData() {
      try {
        // Try to load from public JSON first
        const res = await fetch("/seasons_rankings.json");
        const json = await res.json();
        const seasonData = json[`${season} Season`];

        if (seasonData) {
          // Transform data into array of objects for D3
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
          // Fallback: call API if season missing
          const apiRes = await fetch("/api/fit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ season }),
          });
          const apiData = await apiRes.json();
          setData(apiData); // adapt as needed based on API shape
        }
      } catch (err) {
        console.error("Failed to fetch rankings:", err);
      }
    }

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
    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .attr("preserveAspectRatio", "xMidYMid meet")
       .style("width", "100%")
       .style("height", "auto");

    const x = d3.scaleLinear()
      .domain([d3.min(sortedData, d => d.hdi_lower)!, d3.max(sortedData, d => d.hdi_upper)!])
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
      .text(`NFL Power Rankings ${season}`);

    // Axes
    g.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickFormat(d3.format(".2f")));

    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickSize(0))
      .select(".domain").remove();

    // HDI lines
    g.selectAll(".team-group")
      .data(sortedData)
      .enter()
      .append("line")
      .attr("x1", d => x(d.hdi_lower))
      .attr("x2", d => x(d.hdi_upper))
      .attr("y1", d => (y(d.team) || 0) + y.bandwidth()/2)
      .attr("y2", d => (y(d.team) || 0) + y.bandwidth()/2)
      .attr("stroke", "#f7c267")
      .attr("stroke-width", 3)
      .attr("opacity", 0.7);

    // Mean points
    g.selectAll(".team-group-circle")
      .data(sortedData)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.mean))
      .attr("cy", d => (y(d.team) || 0) + y.bandwidth()/2)
      .attr("r", 6)
      .attr("fill", "#cd0f0fff")
      .attr("stroke", "white")
      .attr("stroke-width", 2);
  };

  return (
    <div ref={wrapperRef} style={{ width: "100%", minHeight: "500px" }}>
      <svg ref={svgRef}></svg>
    </div>
  );
}
