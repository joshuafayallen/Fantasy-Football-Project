"use client";
import { FaArrowLeft, FaChartLine, FaCalculator, FaCircleInfo } from "react-icons/fa6";
import { MathJax, MathJaxContext } from "better-react-mathjax";

export default function MethodologyPage() {
  return (
    <MathJaxContext
      config={{
        tex: {
          inlineMath: [["$", "$"], ["\\(", "\\)"]],
          displayMath: [["$$", "$$"], ["\\[", "\\]"]],
        },
      }}
    >
      <div className="min-h-screen bg-gray-900 text-gray-100">
        <div className="max-w-4xl mx-auto p-6 space-y-8">
          {/* Header */}
          <header className="border-b border-gray-700 pb-6">
            <button
              onClick={() => window.history.back()}
              className="flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-4 transition-colors"
            >
              <FaArrowLeft size={16} />
              Back to Rankings
            </button>
            <h1 className="text-4xl font-bold mb-3">Methodology</h1>
            <p className="text-xl text-gray-400">
              Understanding the statistical models behind NFL Power Rankings
            </p>
          </header>

          {/* Overview */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <FaCircleInfo className="text-blue-400" size={24} />
              <h2 className="text-2xl font-semibold">Overview</h2>
            </div>
            <p className="text-gray-300 leading-relaxed">
             I originally got the idea to do this from <a href="https://blog.damoncroberts.io/posts/baseball_paired/content" target = "_blank" rel ="noopener no reffere" className="text-blue-400 hover:text-blue-300 underline"> Damon&apos; blog
             </a>   where he uses it for MLB rankings. Ranking things is an interesting problem and the source of a lot of anger for fans.
             One of the problems is that we have incomplete information and sometimes bad information. or example, during the 2024 season the Buffalo Bills never played the eventual Super Bowl champions the Philadelphia Eagles. To take another example just because a team loses to a bad team one week does not mean that they are bad. In the last game of the season good teams tend to rest their players if they have clinched a playoff spot and can&apos;t improve their standings. In addition, things sometimes happen in the NFL where a bad team wins against a good team. Just because the 2024 Ravens lost to the Raiders does not mean that the Ravens were bad or the Raiders were good. 

             </p>
            <p className="text-gray-300 leading-relaxed">
             The NFL puts a lot of effort into making the schedule because they have to balance a lot of stakeholders: the owners, broadcasting partners, the fans, and the players. There are also structural factors that they have to work around. Under the current 17 game scheduling rules: each team has to play their division opponents twice, each team must play every team in a different division within your conference, and each team must play every team in another division outside your conference. Then each team must play two teams within in your conference that finished in the same place that you did in your division and one game against a team in another conference that finished in the same place in the division as you did. What this means is that there is a lot of incomplete information since the season is short and not every team plays eachother. Fortunately this is well worn territory in the statistical world. We can use paired comparision models to estimate a team&apos; latent ability over the course of a season based on who they lose against and who they win against.
            </p>


          </section>

          {/* Bradley-Terry Model */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <FaChartLine className="text-green-400" size={24} />
              <h2 className="text-2xl font-semibold">Bradley-Terry Model and The Davidson Model</h2>
            </div>

            <div className="space-y-4 text-gray-300">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  What it is:
                </h3>
                <p>
                  The simplest model is known as a Bradley-Terry model where we
                  estimate a team&apos;s ability as:
                </p>

                <MathJax className="block bg-gray-900 p-2 rounded mt-2">
                  {`$$ y_{i} \\sim Bernoulli\\left(\\text{logit}^{-1}(\\log(\\alpha_{home}) - \\log(\\alpha_{away}))\\right) $$`}
                </MathJax>

                <p className="mt-2">
                Where $\alpha$ is the estimated latent ability of the team
                indexed by whether they are home or away and $y_i$ is abinary outcome variable where 1 indicates the home team lost and zero denotes an away team win. Sticking with convention we are also going to add a home advantage parameter that we assume is normally distributed.  
                </p>

                <p className="mt-2">

                One of the things that discerning readers may have picked up on is that this does not deal with ties and margin of victory. While there are instances where Baseball games have ended in a tie these are relatively rare in its history. Generally, Baseball just reschedules these things because sometimes whether gets in the way. Over the course of a 162 game season these will tend to get drowned out. In Football ties are, relatively speaking, more common and due to the scaricity of Football games potentially more problematic.
                </p>

                <p className="mt-2">
                Additionally, we would expect good teams to blow a few teams out. If we think about the 2024 Kansas City Chiefs and the 2022 Minnesota Vikings they were two teams that won a lot of one score games. 
                  
                </p>

                <p className="mt-2">

                The Davidson model is a straightforward extension of the Bradley-Teryy model that allows us to use an outcome variable with multiple categories. In this case we are simply subbing in an ordered outcome to the likelihood. In this case we are not only prediciting ties and wins but we can add magnitudes of wins as categories.  
                </p>

                <p className="mt-2">
                The most straightforward way to think about these estimates is &quot;what is the probablity that this team beats an average team&quot; where league average is kind of just a hypothetical team.  While the rankings, for the most part, pass the eye test the skills parameter is from a logistic regression(logit). By using a logit we introduce non-linearity that we need to account for when interpreting the skill estimate. What this means is that if we move from 0.5 to 0.75 &quot;skill&quot; this change does not mean that one team has a 75% probability of beating the league average team and the other team has a 50% probability of beating that same team.

                </p>

                <p className="mt-2">
                  Instead what this means is that to get the probability of beating an average team we have to pass this through the equation below

                  <MathJax className="block bg-gray-900 p-2 rounded mt-2">
                    {
                      `$$
                      \\text{P(Bills > Average Team)} = \\frac{e^{Skill}}{1 + e^{Skill}}
                      $$`}
 

                  </MathJax>
                  Where e is the base fo the natural logorithm. So when we go and sub in our skill values. This will just turn our odds into probabilities making them, only slightly, more interpretable. For simplicity instead of having you calculate it yourself you should see this appear in the tooltip. 
                </p>

              </div>
            </div>
          </section>
        </div>
      </div>
    </MathJaxContext>
  );
}
