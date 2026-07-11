# AGI Architecture Content - deepseek (Public Source)

**Source**: https://arxiv.org/html/2505.09343v1
**Title**: Insights into DeepSeek-V3: Scaling Challenges and ...
**Collection Date**: 2026-07-11
**Method**: Firecrawl web search and scrape
**Note**: This is publicly available content, not a direct model response to the standardized prompt.

---

HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

- failed: eso-pic

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2505.09343v1 \[cs.DC\] 14 May 2025

# Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures

Report issue for preceding element

Chenggang Zhao,
Chengqi Deng,
Chong Ruan,
Damai Dai,
Huazuo Gao,
Jiashi Li,
Liyue Zhang,
Panpan Huang,
Shangyan Zhou,
Shirong Ma,
Wenfeng Liang,
Ying He,
Yuqing Wang,
Yuxuan Liu,
Y.X. Wei
DeepSeek-AIBeijingChina

Report issue for preceding element

(2025)

###### Abstract.

Report issue for preceding element

The rapid scaling of large language models (LLMs) has unveiled critical limitations in current hardware architectures, including constraints in memory capacity, computational efficiency, and interconnection bandwidth. DeepSeek-V3, trained on 2,048 NVIDIA H800 GPUs, demonstrates how hardware-aware model co-design can effectively address these challenges, enabling cost-efficient training and inference at scale. This paper presents an in-depth analysis of the DeepSeek-V3/R1 model architecture and its AI infrastructure, highlighting key innovations such as Multi-head Latent Attention (MLA) for enhanced memory efficiency, Mixture of Experts (MoE) architectures for optimized computation-communication trade-offs, FP8 mixed-precision training to unlock the full potential of hardware capabilities, and a Multi-Plane Network Topology to minimize cluster-level network overhead. Building on the hardware bottlenecks encountered during DeepSeek-V3’s development, we engage in a broader discussion with academic and industry peers on potential future hardware directions, including precise low-precision computation units, scale-up and scale-out convergence, and innovations in low-latency communication fabrics. These insights underscore the critical role of hardware and model co-design in meeting the
escalating demands of AI workloads, offering a practical blueprint for innovation in next-generation AI systems.

Report issue for preceding element

Large Language Model, Mixture-of-Experts, Deep Learning, FP8 Mixed-Precision Training, Multi-Plane Network, Co-Design

††journalyear: 2025††copyright: acmlicensed††conference: Proceedings of the 52nd Annual International Symposium on Computer Architecture; June 21–25, 2025; Tokyo, Japan††booktitle: Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA ’25), Ju
