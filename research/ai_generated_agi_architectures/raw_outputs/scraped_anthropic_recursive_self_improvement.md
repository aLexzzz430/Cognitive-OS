# Scraped Content - anthropic_recursive_self_improvement

**Source URL**: https://www.anthropic.com/institute/recursive-self-improvement
**Collection Date**: 2026-07-11
**Note**: Public web content, not direct AI output

---

# When AI builds itself

Our progress toward recursive self-improvement, and its implications.

For most of AI’s history, humans drove every step in its development cycle. But at Anthropic, we are delegating a growing share of AI development to AI systems themselves, which is speeding up our work.

Taken far enough, and given enough compute, that trend points to an AI system capable of fully autonomously designing and developing its own successor. This is called _recursive self-improvement_. We are not there yet, and recursive self-improvement is not inevitable. But it could come sooner than most institutions are prepared for.

Using public benchmarks and previously unreported data from within Anthropic, [The Anthropic Institute](https://www.anthropic.com/institute) is showing that AI is already accelerating the development of AI systems. To take just one example: today, Anthropic engineers on average ship 8x as much code per quarter as they did from 2021-2025.

The technical trends discussed in this piece suggest that AI systems are going to become much more capable in coming years. These trends have huge implications. AI that can build itself would be a major development in the history of technology—one that could bring [enormous good for the world](https://www.darioamodei.com/essay/machines-of-loving-grace) in science, healthcare, and beyond. But full recursive self-improvement also might increase the [risks](https://www.darioamodei.com/essay/the-adolescence-of-technology) of humans losing control over AI systems. If systems are capable of fully building their own successors, the ways we secure them, monitor them, and shape their behavior all grow much more important.

personcomputerpersoncomputerchatbotpersoncomputerchatbotagentpersoncomputerchatbotagentworkers

2021–2023

Building the first Claude

In the early days, work at Anthropic looked like work at any other tech company: people writing code and docs on laptops.

2023–2025

Chatbots

People used early chatbots to help with parts of the process, like generating short code snippets and copying the output into text editors.

2025–2026

Coding agents

As the agents became more capable, they were able to write and edit code on their own, sometimes entire files.

Today

Autonomous agents

Agents can now run code themselves and delegate hours of work to other agents.

20XX?

Closing the loop

In the future, agents could become capable enough to build and train models themselves. If this happens, future versions of Claude could be continuously improved by Claude itself.

### **Evidence from the outside world**

The rate at which AI models improve is accelerating. The length of tasks that they can [reliably complete on their own](https://metr.org/time-horizons/) has been doubling roughly every four months, up from an earlier trend of [doubling](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) every seven months. In March 2024, Claude Opus 3 could complete software tasks that take humans about four minutes to complete. A year later, Claude Sonnet 3.7 managed tasks that took about an hour and a half. A year after that, Claude Opus 4.6 managed 12-hour tasks.1 If this trend holds, tasks that take a skilled person days could come into range this year. In 2027, AI systems could be capable of tasks that take a person weeks.

The same pattern appears on coding and research benchmarks. Benchmarks measure the performance of models in a given domain, and they’re “saturated” when models achieve close to 100% performance.2 [SWE-bench](https://www.swebench.com/) is a standard test of real-world software engineering: it hands a model an actual open-source codebase and a real bug report, and asks it to write a code change that fixes the issue and passes the project’s own tests. Models have gone from scoring in the low single digits to saturating the benchmark in two years.

[CORE-Bench](https://arxiv.org/abs/2409.11363) tests whether a model can reproduce existing research, a prerequisite for them to conduct original research. It gives an AI model the code and data behind a published paper, and asks it to rerun everything and confirm it can replicate the paper’s results. AI systems went from succeeding at reproducing the results roughly 20% of the time in 2024 to saturating the benchmark fifteen months later. METR, which runs the benchmark measuring how well models can complete long-duration tasks, [found](https://x.com/METR_Evals/status/2052896621760004602) that Claude Mythos Preview could work for “at least” 16 hours and was “at the upper end of what \[METR\] can measure without new tasks.”

Public benchmarks say a lot about the capabilities of these systems. But they can’t reveal the impact AI systems are having on speeding up AI development itself. For that, we need direct evidence from within AI companies like Anthropic.

### **Evidence from within Anthropic**

Building a frontier model takes two broad categories of work. There is _engi
