"use client";
import { useState, useEffect, useRef } from "react";
import { FaBluesky, FaGithub, FaLinkedin } from "react-icons/fa6";
import * as d3 from "d3";

interface RankingData {
  team: string;
  team_ability: number;
  hdi_low_ability: number;
  hdi_high_ability: number;
  'Team Rank': number;
  logo_url: string;
  off_epa_per_play: number;
  def_epa_per_play: number;
  record: string;
  team_win_prob: number;
  hdi_low_prob: number;
  hdi_high_prob: number;
}

const NFL_TEAM_COLOR: Record<string, string> = {
'ARI':'#97233F',
'ATL':'#A71930',
'BAL':'#241773',
'BUF':'#00338D',
'CAR':'#0085CA',
'CHI':'#0B162A',
'CIN':'#FB4F14',
'CLE':'#FF3C00',
'DAL':'#002244',
'DEN':'#002244',
'DET':'#0076B6',
'GB':'#203731',
'HOU':'#03202F',
'IND':'#002C5F',
'JAX':'#006778',
'KC':'#E31837',
'LA':'#003594',
'LAC':'#007BC7',
'LV':'#000000',
'MIA':'#008E97',
'MIN':'#4F2683',
'NE':'#002244',
'NO':'#D3BC8D',
'NYG':'#0B2265',
'NYJ':'#003F2D',
'PHI':'#004C54',
'PIT':'#000000',
'SEA':'#002244',
'SF':'#AA0000',
'TB':'#A71930',
'TEN':'#002244',
'WAS':'#5A1414'}

interface TimelineData {
  team: string;
  seasons: {
    season: number;
    rank: number;
    ability: number;
    off_epa_per_play: number;
    def_epa_per_play: number;
  }[];
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
  const [startYear, setStartYear] = useState(1999);
  const [endYear, setEndYear] = useState(2024);
  const [viewMode, setViewMode] = useState<'current' | 'timeline'>('current');
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const timelineSvgRef = useRef<SVGSVGElement>(null);
  const [timelineData, setTimelineData] = useState<Record<string, TimelineData>>({});

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
          team_ability: seasonData.team_ability[i],
          hdi_low_ability: seasonData.hdi_low_ability[i],
          hdi_high_ability: seasonData.hdi_high_ability[i],
          'Team Rank': seasonData["Team Rank"][i],
          logo_url: seasonData.logo_url[i],
          off_epa_per_play: seasonData.off_epa_per_play[i],
          def_epa_per_play: seasonData.def_epa_per_play[i],
          record: seasonData.record[i],
          team_win_prob: seasonData.team_win_prob[i],
          hdi_low_prob: seasonData.hdi_low_prob[i],
          hdi_high_prob: seasonData.hdi_high_prob[i]
        }));
      } else {
        const apiRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/fit`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ season }),
        });
        const apiData = await apiRes.json();
        if (!Array.isArray(apiData)) {
          throw new Error("API response should be an array of team objects");
        }
        rankingData = apiData.map((teamData) => ({
          team: teamData.team,
          team_ability: teamData.team_ability,
          hdi_low_ability: teamData.hdi_low_ability,
          hdi_high_ability: teamData.hdi_high_ability,
          'Team Rank': teamData['Team Rank'],
          logo_url: teamData.logo_url,
          off_epa_per_play: teamData.off_epa_per_play,
          def_epa_per_play: teamData.def_epa_per_play,
          record: teamData.record,
          team_win_prob: teamData.team_win_prob,
          hdi_high_prob: teamData.hdi_high_prob,
          hdi_low_prob: teamData.hdi_low_prob
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

  const fetchTimelineData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/seasons_rankings.json");
      const json = await res.json();
      const teamTimelines: Record<string, TimelineData> = {};
      for (let year = startYear; year <= endYear; year++) {
        const seasonKey = `${year} Season`;
        const seasonData = json[seasonKey];
        if (seasonData) {
          seasonData.team.forEach((team: string, i: number) => {
            if (!teamTimelines[team]) {
              teamTimelines[team] = {
                team,
                seasons: [],
              };
            }
            teamTimelines[team].seasons.push({
              season: year,
              rank: seasonData["Team Rank"][i],
              ability: seasonData.team_ability[i],
              off_epa_per_play: seasonData.off_epa_per_play[i],
              def_epa_per_play: seasonData.def_epa_per_play[i],
            });
          });
        }
      }
      setTimelineData(teamTimelines);
    } catch (err: unknown) {
      console.error("Failed to fetch timeline data:", err);
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
    if (viewMode === 'current') {
      fetchData();
    } else {
      fetchTimelineData();
    }
  }, [season, viewMode, startYear, endYear]);

  useEffect(() => {
    if (data.length && dimensions && viewMode === 'current') {
      drawChart(data, dimensions.width);
    }
  }, [data, dimensions, viewMode]);

  useEffect(() => {
    if (Object.keys(timelineData).length > 0 && dimensions && viewMode === 'timeline') {
      drawTimelineChart(timelineData, dimensions.width);
    }
  }, [timelineData, dimensions, viewMode, selectedTeams]);

  const drawChart = (chartData: RankingData[], width: number) => {
    if (!svgRef.current) return;
    const sortedData = [...chartData].sort((a, b) => a["Team Rank"] - b["Team Rank"]);
    const height = Math.max(400, sortedData.length * 35 + 100);
    const margin = { top: 40, right: 60, bottom: 60, left: 80 };
    const tooltip = d3.select("#tooltip");
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("width", "100%")
      .style("height", "auto");
    const x = d3
      .scaleLinear()
      .domain([d3.min(sortedData, (d) => d.hdi_low_ability)!, d3.max(sortedData, (d) => d.hdi_high_ability)!])
      .nice()
      .range([margin.left, width - margin.right]);
    const y = d3
      .scaleBand()
      .domain(sortedData.map((d) => d.team))
      .range([margin.top, height - margin.bottom])
      .padding(0.2);
    const g = svg.append("g");




    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", 25)
      .attr("text-anchor", "middle")
      .style("font-size", "24px")
      .style("font-weight", "bold")
      .style("fill", "white")
      .text(`NFL Power Rankings ${season}`);


    const xAxis = g
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickFormat(d3.format(".2f")));
    xAxis.selectAll('text').style("font-size", "14px");
    const yAxis = g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickSize(0));
    yAxis.select(".domain").remove();
    yAxis.selectAll("text").remove();
    g.selectAll(".team-line")
      .data(sortedData)
      .enter()
      .append("line")
      .attr("x1", (d) => x(d.hdi_low_ability))
      .attr("x2", (d) => x(d.hdi_high_ability))
      .attr("y1", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("y2", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("stroke", "#f7c267")
      .attr("stroke-width", 3)
      .attr("opacity", 0.7);
    g.selectAll(".team-circle")
      .data(sortedData)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.team_ability))
      .attr("cy", (d) => (y(d.team) || 0) + y.bandwidth() / 2)
      .attr("r", 6)
      .attr("fill", "#fbfbfbff")
      .attr("stroke", "white")
      .attr("stroke-width", 2)
      .on("mouseover", function(event, d) {
        tooltip
          .style('left', `${event.pageX + 12}px`)
          .style('top', `${event.pageY - 20}px`)
          .style('position', "absolute")
          .style("z-index", '10')
          .html(`<div><strong>${d.team}</strong></div>
            <div>Off EPA/play: ${d.off_epa_per_play.toFixed(3)}</div>
            <div>Def EPA/play: ${d.def_epa_per_play.toFixed(3)}</div>
            <div>Record: ${d.record}</div>
            <div>Prob beats average team ${d.hdi_low_prob.toFixed(2)}-${d.hdi_high_prob.toFixed(2)}</div>`)
          .classed('hidden', false);
      })
      .on("mousemove", function(event) {
        tooltip
          .style("left", `${event.pageX + 12}px`)
          .style("top", `${event.pageY - 20}px`);
      })
      .on("mouseout", function() {
        tooltip.classed("hidden", true);
      });
    const yAxisGroup = g.append("g").attr("class", "y-axis-custom");
    sortedData.forEach((d) => {
      const yPos = (y(d.team) || 0) + y.bandwidth() / 2;
      const teamGroup = yAxisGroup.append("g");
      teamGroup
        .append("text")
        .attr("x", 15)
        .attr("y", yPos)
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .style("font-size", "18px")
        .style("font-weight", "bold")
        .style("fill", "#f0f0f0ff")
        .text(`${d["Team Rank"]}`);
      teamGroup
        .append("image")
        .attr("x", 30)
        .attr("y", yPos - 12)
        .attr("width", 24)
        .attr("height", 24)
        .attr("href", d.logo_url)
        .attr("preserveAspectRatio", "xMidYMid meet");
      teamGroup
        .append("text")
        .attr("x", 58)
        .attr("y", yPos)
        .attr("dy", "0.5em")
        .style("font-size", "18px")
        .style("fill", "#f7f7f7ff")
        .text(d.team);
    });
  };

  const drawTimelineChart = (teamData: Record<string, TimelineData>, width: number) => {
    if (!timelineSvgRef.current) return;
    
    const teamsToShow = selectedTeams.length > 0
      ? selectedTeams.filter(t => teamData[t])
      : Object.keys(teamData).slice(0, 10);
    
    const height = 600;
    const margin = { top: 60, right: 120, bottom: 60, left: 60 };
    const svg = d3.select(timelineSvgRef.current);
    svg.selectAll('*').remove();
    
    svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%')
      .style('height', 'auto');
    
    const seasons = Array.from(
      new Set(
        Object.values(teamData)
          .flatMap(t => t.seasons.map(s => s.season))
      )
    ).sort();
    
    const x = d3
      .scaleLinear()
      .domain([Math.min(...seasons), Math.max(...seasons)])
      .range([margin.left, width - margin.right]);
    
    const y = d3
      .scaleLinear()
      .domain([1, 32])
      .range([margin.top, height - margin.bottom]);
    

    const g = svg.append('g');
    
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .style('font-size', '24px')
      .style('font-weight', 'bold')
      .style('fill', 'white')
      .text(`Team Rankings Over Time (${startYear} - ${endYear})`);

      g.append('g')
      .attr('class', 'grid')
      .attr("transform", `translate(${margin.left},0)`)
      .call(
        d3.axisLeft(y)
        .tickValues([1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15, 16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31, 32])
        .tickSize(-(width - margin.left - margin.right))
        .tickFormat(() => "")
      )
      .selectAll("line")
      .style('stroke', "#ffff")
      .style("stroke-opacity", 0.3)
      .style('stroke-dash', "3,3");

      g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0, ${height - margin.bottom})`)
      .call(
        d3.axisBottom(x)
        .tickFormat(d3.format("d"))
        .tickSize(-(height- margin.top - margin.bottom))
      )
      .selectAll('line')
      .style("stroke", "#ffff")
      .style("stroke-opacity", 0.3)
      .style("stroke-dash", "3,3");

    g.selectAll(".grid .domain").remove();

    
    
    g.append("g")
      .attr("transform", `translate(${margin.left}, 0)`)
      .call(d3.axisLeft(y).tickValues([1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15, 16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31, 32]))
      .selectAll('text')
      .style("font-size", "14px")
      .style("fill", 'white');

    
    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 15)
      .attr("x", -(height / 2))
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("fill", "white")
    
    
    const line = d3.line<{ season: number; rank: number }>()
      .x(d => x(d.season))
      .y(d => y(d.rank))
      .curve(d3.curveMonotoneX);
    
    const timelinetooltip = d3.select("#timeline-tooltip");
    
    teamsToShow.forEach((team, i) => {
      const data = teamData[team];
      const teamColor = NFL_TEAM_COLOR[team] || '#88888888';
      if (!data || data.seasons.length === 0) return;
      
      g.append("path")
        .datum(data.seasons)
        .attr("fill", "none")
        .attr("stroke", teamColor)
        .attr("stroke-width", 3)
        .attr("d", line)
        .style("opacity", 0.8);
      
      g.selectAll(`.point-${team}`)
        .data(data.seasons)
        .enter()
        .append("circle")
        .attr("class", `point-${team}`)
        .attr("cx", d => x(d.season))
        .attr("cy", d => y(d.rank))
        .attr("r", 5)
        .attr("fill",teamColor)
        .attr("stroke", "white")
        .attr("stroke-width", 2)
        .on("mouseover", function(event, d) {
          timelinetooltip
            .style("left", `${event.pageX + 12}px`)
            .style("top", `${event.pageY - 20}px`)
            .style("position", "absolute")
            .style("z-index", '10')
            .html(`<div><strong>${team}</strong></div>
              <div>Season: ${d.season}</div>
              <div>Rank: #${d.rank}</div>
              <div>Off EPA/play: ${d.off_epa_per_play.toFixed(3)}</div>
              <div>Def EPA/play: ${d.def_epa_per_play.toFixed(3)}</div>`)
            .classed("hidden", false);
        })
        .on("mouseout", function() {
          timelinetooltip.classed("hidden", true);
        });
      
      const lastPoint = data.seasons[data.seasons.length - 1];
      g.append("text")
        .attr("x", width - margin.right + 10)
        .attr("y", y(lastPoint.rank))
        .attr("dy", "0.35em")
        .style("font-size", "12px")
        .style("fill", 'white')
        .style("font-weight", "bold")
        .text(team);
    });
  };

  const toggleTeamSelection = (team: string) => {
    setSelectedTeams(prev =>
      prev.includes(team)
        ? prev.filter(t => t !== team)
        : [...prev, team]
    );
  };

  return (
    <div ref={wrapperRef} className="min-h-screen bg-gray-950 text-gray-100">
      <div className="p-6 max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-3xl font-bold mb-2">
            Josh Allen&apos;s Total Correct NFL Power Rankings
          </h1>
          <p className="text-sm text-gray-400">
            Points represent skill estimates from a Bradley-Terry Model, lines show 97% high density intervals.
          </p>
        </header>
        <div className="flex gap-4 mb-4">
          <button
            onClick={() => setViewMode('current')}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              viewMode === 'current'
                ? 'bg-purple-500 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            Current Season
          </button>
          <button
            onClick={() => setViewMode('timeline')}
            className={`px-4 py-2 rounded font-medium transition-colors ${
              viewMode === 'timeline'
                ? 'bg-purple-500 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            Rankings Timeline
          </button>
        </div>
        {viewMode === 'current' ? (
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label htmlFor="season" className="text-sm font-medium">Season:</label>
              <input
                id="season"
                type="number"
                min="1999"
                max="2025"
                value={season}
                onChange={(e) => setSeason(parseInt(e.target.value))}
                className="px-3 py-1 bg-gray-900 border border-gray-700 rounded text-sm focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <button
              onClick={fetchData}
              disabled={loading}
              className="px-4 py-2 rounded bg-purple-500 text-white font-medium hover:bg-purple-600 disabled:bg-purple-400 disabled:cursor-not-allowed"
            >
              {loading ? "Loading..." : "Load Rankings"}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Start Year:</label>
                <input
                  type="number"
                  min="1999"
                  max="2025"
                  value={startYear}
                  onChange={(e) => setStartYear(parseInt(e.target.value))}
                  className="px-3 py-1 bg-gray-900 border border-gray-700 rounded text-sm focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">End Year:</label>
                <input
                  type="number"
                  min="1999"
                  max="2025"
                  value={endYear}
                  onChange={(e) => setEndYear(parseInt(e.target.value))}
                  className="px-3 py-1 bg-gray-900 border border-gray-700 rounded text-sm focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
            {Object.keys(timelineData).length > 0 && (
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3">Select Teams to Compare (max 10)</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2 max-h-96 overflow-y-auto">
                  {Object.keys(timelineData).sort().map(team => (
                    <button
                      key={team}
                      onClick={() => toggleTeamSelection(team)}
                      disabled={selectedTeams.length >= 10 && !selectedTeams.includes(team)}
                      className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                        selectedTeams.includes(team)
                          ? 'bg-purple-500 text-white'
                          : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {team}
                    </button>
                  ))}
                </div>
                {selectedTeams.length > 0 && (
                  <button
                    onClick={() => setSelectedTeams([])}
                    className="mt-3 px-4 py-2 rounded bg-red-500 text-white text-sm hover:bg-red-600"
                  >
                    Clear Selection
                  </button>
                )}
              </div>
            )}
          </div>
        )}
        {error && (
          <div className="bg-red-500/10 border border-red-500 text-red-400 px-4 py-3 rounded">
            Error: {error}
          </div>
        )}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 overflow-x-auto shadow">
          {viewMode === 'current' ? (
            <>
              <svg ref={svgRef}></svg>
              <div
                id="tooltip"
                className="absolute hidden bg-gray-900 text-white text-sm px-3 py-2 rounded shadow-lg pointer-events-none border border-gray-700"
              ></div>
            </>
          ) : (
            <>
              <svg ref={timelineSvgRef}></svg>
              <div
                id="timeline-tooltip"
                className="absolute hidden bg-gray-900 text-white text-sm px-3 py-2 rounded shadow-lg pointer-events-none border border-gray-700"
              ></div>
            </>
          )}
        </div>
        {data.length > 0 && viewMode === 'current' && (
          <section className="bg-gray-900 rounded-lg p-4 shadow">
            <h3 className="text-lg font-semibold mb-3">Top 5 Teams</h3>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
              {data
                .sort((a, b) => b.team_ability - a.team_ability)
                .slice(0, 5)
                .map((team, index) => (
                  <div
                    key={team.team}
                    className="bg-gray-800 rounded p-4 text-center shadow hover:shadow-lg transition"
                  >
                    <div className="text-2xl font-bold text-purple-400 mb-2">
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
                      Est Skill: {team.team_ability.toFixed(3)} <br />
                      OFF EPA/play: {team.off_epa_per_play.toFixed(3)} <br />
                      DEF EPA/play: {team.def_epa_per_play.toFixed(3)} <br />
                      Record: {team.record} <br />
                      Prob beats average team {team.team_win_prob.toFixed(2)}
                    </div>
                  </div>
                ))}
            </div>
          </section>
        )}
      </div>
      <footer className="mt-8 p-4 bg-gray-900 text-gray-300 flex justify-center gap-6">
        <a href="https://bsky.app/profile/joshfallen.bsky.social" target="_blank" rel="noopener noreferrer">
          <FaBluesky size={24} className="hover:text-white transition-colors" />
        </a>
        <a href="https://github.com/joshuafayallen/Fantasy-Football-Project/tree/main/rankings" target="_blank" rel="noopener noreferrer">
          <FaGithub size={24} className="hover:text-white transition-colors" />
        </a>
        <a href="https://www.linkedin.com/in/joshua-allen-112b81119/" target="_blank" rel="noopener noreferrer">
          <FaLinkedin size={24} className="hover:text-white transition-colors" />
        </a>
      </footer>
    </div>
  );
}