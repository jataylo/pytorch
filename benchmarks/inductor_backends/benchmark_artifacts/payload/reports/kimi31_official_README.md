---
tags:
- compressed-tensors
- conversational
license: other
license_name: "kimi-k3.1-pro"
library_name: transformers
pipeline_tag: image-text-to-text
---

<div align="center">
  <picture>
      <img src="assets/kimi-logo.png" width="30%" alt="Kimi K3.1 Pro">
  </picture>
</div>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.ai" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/Chat-Kimi%20K3.1%20Pro-1783ff"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>
<div align="center" style="line-height:1">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-Moonshot%20AI-ffc107"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="X" src="https://img.shields.io/badge/X-Kimi.ai-white?logo=x&logoColor=white"/></a>
  <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
  <a href="https://modelscope.cn/organization/moonshotai" target="_blank"><img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-Moonshot%20AI-white?labelColor=rgb(99%2C%2074%2C%255)"/></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Kimi_K3.1_Pro-f5de53?&color=f5de53"/></a>
</div>

<p align="center">
📰&nbsp;&nbsp;<a href="https://www.kimi.ai/blog/kimi-k3-1-pro">Tech Blog</a> | &nbsp;&nbsp;&nbsp; <b>🚀&nbsp;&nbsp;<a href="https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart">API Quickstart</a></b>
</p>



## 1. Model Introduction

Kimi K3.1 Pro is an open-weight, natively multimodal agentic model, and our most capable 3T-class model to date. Post-trained from Kimi K3 with large-scale joint reinforcement learning across hundreds of downstream domains, it is the first model in the K3.1 series. With a 1-million-token context window, Kimi K3.1 Pro delivers substantial improvements in coding, 3D creation, financial research, factual accuracy, and other agentic capabilities.

## 2. Model Summary

<div align="center">
<table>
<tbody>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Architecture</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">Mixture-of-Experts (MoE)</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Total Parameters</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2.8T</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Activated Parameters</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">104B</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Layers</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">93</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Dense Layers</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention-Layer Composition</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">69 KDA + 24 Gated DSA<sup>*</sup></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention Hidden Dimension</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">7168</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Attention Heads</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">96</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>DSA Index Head Dimension</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">128</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>DSA Index Number of Heads</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">32</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>DSA Index Top-K</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2048</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Latent MoE Dimension</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">3584</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>MoE Hidden Dimension</strong> (per Expert)</td>
<td align="center" style="vertical-align: middle; text-align: center">3072</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Experts</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">896</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Selected Experts per Token</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">16</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Number of Shared Experts</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Vocabulary Size</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">160K</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Context Length</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">1048576</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Attention Mechanism</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">KDA &amp; Gated DSA</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Activation Function</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">SiTU-GLU</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Vision Encoder</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">MoonViT-V2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Parameters of Vision Encoder</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">401M</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Quantization</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">MXFP4 weights / MXFP8 activations<br>(quantization-aware training)</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center"><strong>Modality</strong></td>
<td align="center" style="vertical-align: middle; text-align: center">Text, Image</td>
</tr>
</tbody>
</table>
</div>

<p><sup>*</sup> Compared with Kimi K3, we replace the full-attention Gated MLA layers with Gated DeepSeek Sparse Attention (Gated DSA) layers for more efficient long-context modeling.</p>

## 3. Evaluation Results

Unless otherwise specified, all Kimi K3.1 Pro results use reasoning effort `max`, `temperature = 1.0`, and `top-p = 0.95`. A dash (`—`) means that the source table does not report a score. Higher scores are better except where the benchmark name explicitly states `lower is better`; Elo scores and other benchmark-specific metrics are not directly comparable with accuracy or pass-rate scores.

<div align="center">
<table>
<thead>
<tr>
<th align="center" style="text-align: center">Benchmark</th>
<th align="center" style="text-align: center"><sup>Kimi K3.1 Pro<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Fable 5.1<br><sup>(max, w/ fallback)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GPT-6 Astra<br><sup>(max, w/ fallback)</sup></sup></th>
<th align="center" style="text-align: center"><sup>Claude Opus 5.5<br><sup>(max, w/ fallback)</sup></sup></th>
<th align="center" style="text-align: center"><sup>DeepSeek-V4.1-Flash-1017<br><sup>(max)</sup></sup></th>
<th align="center" style="text-align: center"><sup>GLM-5.3<br><sup>(max)</sup></sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Terminal-Bench 4.0</td>
<td align="center" style="vertical-align: middle; text-align: center">65.7</td>
<td align="center" style="vertical-align: middle; text-align: center">57.9</td>
<td align="center" style="vertical-align: middle; text-align: center">58.2</td>
<td align="center" style="vertical-align: middle; text-align: center">66.4</td>
<td align="center" style="vertical-align: middle; text-align: center">42.3</td>
<td align="center" style="vertical-align: middle; text-align: center">41.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Terminal-Bench-Science 0.1</td>
<td align="center" style="vertical-align: middle; text-align: center">38.1</td>
<td align="center" style="vertical-align: middle; text-align: center">40.0</td>
<td align="center" style="vertical-align: middle; text-align: center">68.1</td>
<td align="center" style="vertical-align: middle; text-align: center">63.3</td>
<td align="center" style="vertical-align: middle; text-align: center">15.7</td>
<td align="center" style="vertical-align: middle; text-align: center">8.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">SWE Marathon v1.1</td>
<td align="center" style="vertical-align: middle; text-align: center">51.7</td>
<td align="center" style="vertical-align: middle; text-align: center">45.6</td>
<td align="center" style="vertical-align: middle; text-align: center">42.5</td>
<td align="center" style="vertical-align: middle; text-align: center">46.3</td>
<td align="center" style="vertical-align: middle; text-align: center">40.0</td>
<td align="center" style="vertical-align: middle; text-align: center">42.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BenchCAD w/o tools</td>
<td align="center" style="vertical-align: middle; text-align: center">72.8</td>
<td align="center" style="vertical-align: middle; text-align: center">60.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">73.0</td>
<td align="center" style="vertical-align: middle; text-align: center">41.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BenchCAD w/ tools</td>
<td align="center" style="vertical-align: middle; text-align: center">92.7</td>
<td align="center" style="vertical-align: middle; text-align: center">92.6</td>
<td align="center" style="vertical-align: middle; text-align: center">95.9</td>
<td align="center" style="vertical-align: middle; text-align: center">96.2</td>
<td align="center" style="vertical-align: middle; text-align: center">81.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Program Bench</td>
<td align="center" style="vertical-align: middle; text-align: center">31.5</td>
<td align="center" style="vertical-align: middle; text-align: center">53.5</td>
<td align="center" style="vertical-align: middle; text-align: center">50.0</td>
<td align="center" style="vertical-align: middle; text-align: center">65.0</td>
<td align="center" style="vertical-align: middle; text-align: center">29.1</td>
<td align="center" style="vertical-align: middle; text-align: center">19.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DeepSWE v1.1</td>
<td align="center" style="vertical-align: middle; text-align: center">75.5</td>
<td align="center" style="vertical-align: middle; text-align: center">67.4</td>
<td align="center" style="vertical-align: middle; text-align: center">74.1</td>
<td align="center" style="vertical-align: middle; text-align: center">74.2</td>
<td align="center" style="vertical-align: middle; text-align: center">77.0</td>
<td align="center" style="vertical-align: middle; text-align: center">66.9</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PostTrain Bench v1.2</td>
<td align="center" style="vertical-align: middle; text-align: center">36.5</td>
<td align="center" style="vertical-align: middle; text-align: center">44.6</td>
<td align="center" style="vertical-align: middle; text-align: center">41.9</td>
<td align="center" style="vertical-align: middle; text-align: center">43.8</td>
<td align="center" style="vertical-align: middle; text-align: center">30.5</td>
<td align="center" style="vertical-align: middle; text-align: center">34.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Code Bench 2.0 (Internal)</td>
<td align="center" style="vertical-align: middle; text-align: center">77.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">76.1</td>
<td align="center" style="vertical-align: middle; text-align: center">61.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Code Bench 3.0 Long-horizon (Internal)</td>
<td align="center" style="vertical-align: middle; text-align: center">61.2</td>
<td align="center" style="vertical-align: middle; text-align: center">58.8</td>
<td align="center" style="vertical-align: middle; text-align: center">66.5</td>
<td align="center" style="vertical-align: middle; text-align: center">58.5</td>
<td align="center" style="vertical-align: middle; text-align: center">51.5</td>
<td align="center" style="vertical-align: middle; text-align: center">45.5</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Agentic</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HLE-Full w/ tools</td>
<td align="center" style="vertical-align: middle; text-align: center">70.0</td>
<td align="center" style="vertical-align: middle; text-align: center">65.6</td>
<td align="center" style="vertical-align: middle; text-align: center">57.2</td>
<td align="center" style="vertical-align: middle; text-align: center">67.7</td>
<td align="center" style="vertical-align: middle; text-align: center">58.8</td>
<td align="center" style="vertical-align: middle; text-align: center">62.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BrowseComp</td>
<td align="center" style="vertical-align: middle; text-align: center">92.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">91.5</td>
<td align="center" style="vertical-align: middle; text-align: center">90.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BioMysteryBench</td>
<td align="center" style="vertical-align: middle; text-align: center">80.6</td>
<td align="center" style="vertical-align: middle; text-align: center">81.6</td>
<td align="center" style="vertical-align: middle; text-align: center">78.9</td>
<td align="center" style="vertical-align: middle; text-align: center">81.9</td>
<td align="center" style="vertical-align: middle; text-align: center">80.0</td>
<td align="center" style="vertical-align: middle; text-align: center">79.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CharXiv (RQ) w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">93.4</td>
<td align="center" style="vertical-align: middle; text-align: center">92.7</td>
<td align="center" style="vertical-align: middle; text-align: center">92.8</td>
<td align="center" style="vertical-align: middle; text-align: center">93.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMMU-Pro w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">90.0</td>
<td align="center" style="vertical-align: middle; text-align: center">88.4</td>
<td align="center" style="vertical-align: middle; text-align: center">87.8</td>
<td align="center" style="vertical-align: middle; text-align: center">89.8</td>
<td align="center" style="vertical-align: middle; text-align: center">82.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GDP.pdf w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">85.7</td>
<td align="center" style="vertical-align: middle; text-align: center">83.8</td>
<td align="center" style="vertical-align: middle; text-align: center">85.5</td>
<td align="center" style="vertical-align: middle; text-align: center">83.9</td>
<td align="center" style="vertical-align: middle; text-align: center">73.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ChartoGraphy w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">90.2</td>
<td align="center" style="vertical-align: middle; text-align: center">88.4</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
<td align="center" style="vertical-align: middle; text-align: center">89.0</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">BabyVision w/ python</td>
<td align="center" style="vertical-align: middle; text-align: center">95.4</td>
<td align="center" style="vertical-align: middle; text-align: center">95.1</td>
<td align="center" style="vertical-align: middle; text-align: center">96.6</td>
<td align="center" style="vertical-align: middle; text-align: center">97.7</td>
<td align="center" style="vertical-align: middle; text-align: center">92.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ZeroBench_main w/ python (pass@5)</td>
<td align="center" style="vertical-align: middle; text-align: center">63.0</td>
<td align="center" style="vertical-align: middle; text-align: center">59.0</td>
<td align="center" style="vertical-align: middle; text-align: center">58.0</td>
<td align="center" style="vertical-align: middle; text-align: center">61.0</td>
<td align="center" style="vertical-align: middle; text-align: center">51.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Agents' Last Exam (ALL)</td>
<td align="center" style="vertical-align: middle; text-align: center">43.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">41.8</td>
<td align="center" style="vertical-align: middle; text-align: center">42.6</td>
<td align="center" style="vertical-align: middle; text-align: center">37.7</td>
<td align="center" style="vertical-align: middle; text-align: center">26.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Agents' Last Exam (CLI)</td>
<td align="center" style="vertical-align: middle; text-align: center">44.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">44.2</td>
<td align="center" style="vertical-align: middle; text-align: center">40.7</td>
<td align="center" style="vertical-align: middle; text-align: center">36.0</td>
<td align="center" style="vertical-align: middle; text-align: center">29.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Automation Bench-AA</td>
<td align="center" style="vertical-align: middle; text-align: center">73.3</td>
<td align="center" style="vertical-align: middle; text-align: center">59.4</td>
<td align="center" style="vertical-align: middle; text-align: center">68.5</td>
<td align="center" style="vertical-align: middle; text-align: center">69.5</td>
<td align="center" style="vertical-align: middle; text-align: center">66.6</td>
<td align="center" style="vertical-align: middle; text-align: center">62.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GDPval-AA v2.1 (Elo-score)</td>
<td align="center" style="vertical-align: middle; text-align: center">1846</td>
<td align="center" style="vertical-align: middle; text-align: center">1754</td>
<td align="center" style="vertical-align: middle; text-align: center">1573</td>
<td align="center" style="vertical-align: middle; text-align: center">1866</td>
<td align="center" style="vertical-align: middle; text-align: center">1746</td>
<td align="center" style="vertical-align: middle; text-align: center">1650</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">AA-Briefcase v1.1 (Elo-score)</td>
<td align="center" style="vertical-align: middle; text-align: center">1678</td>
<td align="center" style="vertical-align: middle; text-align: center">1672</td>
<td align="center" style="vertical-align: middle; text-align: center">1569</td>
<td align="center" style="vertical-align: middle; text-align: center">1803</td>
<td align="center" style="vertical-align: middle; text-align: center">1570</td>
<td align="center" style="vertical-align: middle; text-align: center">1507</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">APEX-Agents</td>
<td align="center" style="vertical-align: middle; text-align: center">69.4</td>
<td align="center" style="vertical-align: middle; text-align: center">68.6</td>
<td align="center" style="vertical-align: middle; text-align: center">64.7</td>
<td align="center" style="vertical-align: middle; text-align: center">73.5</td>
<td align="center" style="vertical-align: middle; text-align: center">69.8</td>
<td align="center" style="vertical-align: middle; text-align: center">56.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OfficeQA Pro v2</td>
<td align="center" style="vertical-align: middle; text-align: center">64.4</td>
<td align="center" style="vertical-align: middle; text-align: center">54.5</td>
<td align="center" style="vertical-align: middle; text-align: center">54.4</td>
<td align="center" style="vertical-align: middle; text-align: center">62.2</td>
<td align="center" style="vertical-align: middle; text-align: center">40.4</td>
<td align="center" style="vertical-align: middle; text-align: center">31.1</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OfficeQA Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">67.4</td>
<td align="center" style="vertical-align: middle; text-align: center">68.6</td>
<td align="center" style="vertical-align: middle; text-align: center">66.2</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">62.4</td>
<td align="center" style="vertical-align: middle; text-align: center">58.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Frontier Finance</td>
<td align="center" style="vertical-align: middle; text-align: center">58.6</td>
<td align="center" style="vertical-align: middle; text-align: center">60.8</td>
<td align="center" style="vertical-align: middle; text-align: center">45.0</td>
<td align="center" style="vertical-align: middle; text-align: center">56.7</td>
<td align="center" style="vertical-align: middle; text-align: center">54.5</td>
<td align="center" style="vertical-align: middle; text-align: center">57.5</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Harvey Legal Agent</td>
<td align="center" style="vertical-align: middle; text-align: center">23.7</td>
<td align="center" style="vertical-align: middle; text-align: center">19.1</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">21.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HANDBOOK.md</td>
<td align="center" style="vertical-align: middle; text-align: center">44.6</td>
<td align="center" style="vertical-align: middle; text-align: center">38.8</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">41.5</td>
<td align="center" style="vertical-align: middle; text-align: center">28.9</td>
<td align="center" style="vertical-align: middle; text-align: center">20.8</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">OSWorld V2.1</td>
<td align="center" style="vertical-align: middle; text-align: center">79.6</td>
<td align="center" style="vertical-align: middle; text-align: center">80.7</td>
<td align="center" style="vertical-align: middle; text-align: center">75.8</td>
<td align="center" style="vertical-align: middle; text-align: center">81.8</td>
<td align="center" style="vertical-align: middle; text-align: center">69.3</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Business Operation (Internal)</td>
<td align="center" style="vertical-align: middle; text-align: center">37.8</td>
<td align="center" style="vertical-align: middle; text-align: center">30.7</td>
<td align="center" style="vertical-align: middle; text-align: center">32.3</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">8.9</td>
<td align="center" style="vertical-align: middle; text-align: center">18.6</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">DECK-Arena (Internal, Elo-score)</td>
<td align="center" style="vertical-align: middle; text-align: center">1456</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">1423</td>
<td align="center" style="vertical-align: middle; text-align: center">1253</td>
<td align="center" style="vertical-align: middle; text-align: center">1231</td>
<td align="center" style="vertical-align: middle; text-align: center">1119</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Design Bench (Internal, Elo-score)</td>
<td align="center" style="vertical-align: middle; text-align: center">957</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">1076</td>
<td align="center" style="vertical-align: middle; text-align: center">927</td>
<td align="center" style="vertical-align: middle; text-align: center">863</td>
<td align="center" style="vertical-align: middle; text-align: center">718</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Hallu Bench Hard w/ tools (Internal, lower is better)</td>
<td align="center" style="vertical-align: middle; text-align: center">8.3</td>
<td align="center" style="vertical-align: middle; text-align: center">23.3</td>
<td align="center" style="vertical-align: middle; text-align: center">5.5</td>
<td align="center" style="vertical-align: middle; text-align: center">15.0</td>
<td align="center" style="vertical-align: middle; text-align: center">23.3</td>
<td align="center" style="vertical-align: middle; text-align: center">22.6</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Reasoning &amp; Knowledge &amp; Multimodal</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GDP.pdf</td>
<td align="center" style="vertical-align: middle; text-align: center">34.6</td>
<td align="center" style="vertical-align: middle; text-align: center">27.6</td>
<td align="center" style="vertical-align: middle; text-align: center">34.2</td>
<td align="center" style="vertical-align: middle; text-align: center">30.6</td>
<td align="center" style="vertical-align: middle; text-align: center">17.2</td>
<td align="center" style="vertical-align: middle; text-align: center">11.2</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HLE-Full</td>
<td align="center" style="vertical-align: middle; text-align: center">51.1</td>
<td align="center" style="vertical-align: middle; text-align: center">57.6</td>
<td align="center" style="vertical-align: middle; text-align: center">53.6</td>
<td align="center" style="vertical-align: middle; text-align: center">59.8</td>
<td align="center" style="vertical-align: middle; text-align: center">42.6</td>
<td align="center" style="vertical-align: middle; text-align: center">43.0*</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">HLE-Diamond</td>
<td align="center" style="vertical-align: middle; text-align: center">39.4</td>
<td align="center" style="vertical-align: middle; text-align: center">55.1</td>
<td align="center" style="vertical-align: middle; text-align: center">65.8</td>
<td align="center" style="vertical-align: middle; text-align: center">60.9</td>
<td align="center" style="vertical-align: middle; text-align: center">21.6</td>
<td align="center" style="vertical-align: middle; text-align: center">25.1*</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">GPQA-Diamond</td>
<td align="center" style="vertical-align: middle; text-align: center">94.2</td>
<td align="center" style="vertical-align: middle; text-align: center">93.7</td>
<td align="center" style="vertical-align: middle; text-align: center">96.1</td>
<td align="center" style="vertical-align: middle; text-align: center">93.9</td>
<td align="center" style="vertical-align: middle; text-align: center">92.2</td>
<td align="center" style="vertical-align: middle; text-align: center">91.7</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">CharXiv (RQ)</td>
<td align="center" style="vertical-align: middle; text-align: center">89.0</td>
<td align="center" style="vertical-align: middle; text-align: center">88.8</td>
<td align="center" style="vertical-align: middle; text-align: center">90.0</td>
<td align="center" style="vertical-align: middle; text-align: center">92.1</td>
<td align="center" style="vertical-align: middle; text-align: center">77.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">MMMU-Pro</td>
<td align="center" style="vertical-align: middle; text-align: center">84.1</td>
<td align="center" style="vertical-align: middle; text-align: center">86.3</td>
<td align="center" style="vertical-align: middle; text-align: center">86.9</td>
<td align="center" style="vertical-align: middle; text-align: center">87.7</td>
<td align="center" style="vertical-align: middle; text-align: center">80.6</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ZeroBench_main (avg@5 / pass@5)</td>
<td align="center" style="vertical-align: middle; text-align: center">29.8 / 39.0</td>
<td align="center" style="vertical-align: middle; text-align: center">23.0 / 35.0</td>
<td align="center" style="vertical-align: middle; text-align: center">43.2 / 52.0</td>
<td align="center" style="vertical-align: middle; text-align: center">35.0 / 43.0</td>
<td align="center" style="vertical-align: middle; text-align: center">11.8 / 23.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">PerceptionBench</td>
<td align="center" style="vertical-align: middle; text-align: center">68.3</td>
<td align="center" style="vertical-align: middle; text-align: center">63.0</td>
<td align="center" style="vertical-align: middle; text-align: center">77.6</td>
<td align="center" style="vertical-align: middle; text-align: center">72.1</td>
<td align="center" style="vertical-align: middle; text-align: center">47.7</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">Kimi Hallu Bench w/o tools (Internal, lower is better)</td>
<td align="center" style="vertical-align: middle; text-align: center">5.4</td>
<td align="center" style="vertical-align: middle; text-align: center">11.4</td>
<td align="center" style="vertical-align: middle; text-align: center">4.2</td>
<td align="center" style="vertical-align: middle; text-align: center">6.8</td>
<td align="center" style="vertical-align: middle; text-align: center">13.1</td>
<td align="center" style="vertical-align: middle; text-align: center">14.8</td>
</tr>
<tr>
<td align="center" colspan=7 style="text-align: center"><strong>Cyber Security</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ExploitBench</td>
<td align="center" style="vertical-align: middle; text-align: center">74.7</td>
<td align="center" style="vertical-align: middle; text-align: center">83.0</td>
<td align="center" style="vertical-align: middle; text-align: center">100</td>
<td align="center" style="vertical-align: middle; text-align: center">91.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">54.4</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">ExploitGym</td>
<td align="center" style="vertical-align: middle; text-align: center">20.7</td>
<td align="center" style="vertical-align: middle; text-align: center">30.4</td>
<td align="center" style="vertical-align: middle; text-align: center">42.4</td>
<td align="center" style="vertical-align: middle; text-align: center">34.5</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">15.0</td>
</tr>
<tr>
<td align="center" style="vertical-align: middle; text-align: center">xdev-bench-hard (Internal)</td>
<td align="center" style="vertical-align: middle; text-align: center">81.0</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">—</td>
<td align="center" style="vertical-align: middle; text-align: center">85.7</td>
<td align="center" style="vertical-align: middle; text-align: center">57.1</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>Evaluation Footnotes</b></summary>

#### Coding benchmarks

1. **Terminal-Bench 4.0.** Kimi K3.1 Pro is evaluated with the Claude Code harness. The DeepSeek-V4.1-Flash-1017 score is taken from its [release blog](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). All remaining scores are the maximum values reported in model blog posts or on the [official Terminal-Bench leaderboard](https://www.tbench.ai/).
2. **Terminal-Bench-Science 0.1.** Kimi K3.1 Pro is evaluated with Claude Code. Internal evaluation of DeepSeek-V4.1-Flash-1017 yields 9.3; the table instead reports 15.7 for the older DeepSeek-V4.1-Flash from the official leaderboard. All remaining scores are from the [Terminal-Bench-Science leaderboard](https://www.terminal-bench-science.ai/).
3. **SWE-Marathon v1.1.** Kimi K3.1 Pro and DeepSeek-V4.1-Flash-1017 are evaluated with Claude Code. Other scores are from the [official leaderboard](https://www.swe-marathon.org/).
4. **ProgramBench.** The table reports binary scores: a task receives 1 if its ProgramBench score is at least 95%, and 0 otherwise. K3.1 Pro is evaluated with Claude Code. DeepSeek and GLM scores are taken from the [DeepSeek release blog](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017); remaining scores are from [Vals AI](https://www.vals.ai/benchmarks/programbench).
5. **DeepSWE v1.1.** K3.1 Pro is evaluated with Claude Code. Other scores come from the respective technical reports for [GPT-6 Astra](https://openai.com/index/gpt-6-astra/), [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5-system-card), [Claude Fable 5.1](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card), [DeepSeek-V4.1-Flash-1017](https://api-docs.deepseek.com/news/news261017/), and [GLM-5.3](https://z.ai/blog/glm-5.3).
6. **PostTrainBench.** K3.1 Pro and DeepSeek-V4.1-Flash-1017 are evaluated internally on v1.2 with Claude Code and H20 GPUs rather than H100 GPUs. For stable evaluation, all judge models use DeepSeek-V4.1-Flash-1017 at maximum effort, which best matched the official GPT-5.6-Terra (xhigh) judge in testing. Other scores are from the [official leaderboard](https://posttrainbench.com/).
7. **Kimi Code Bench 2.0 and Kimi Code Bench 3.0 Long-horizon.** These are internal benchmarks. K3.1 Pro is evaluated with Claude Code; all other models use their maximum-effort setting. Proprietary-model testing may involve fallback behavior.
8. **BenchCAD.** Open models are evaluated on 1,000-case subsets randomly sampled from the 17,900-case Vision2Code dataset at maximum reasoning effort. With-tools results use Claude Code; without-tools results use single-shot generation with official prompts and scoring. Claude scores are from the Opus 5.5 System Card, and GPT-6 Astra's with-tools score is from its official report. GPT-6 Astra has no reported without-tools score.

#### Agentic benchmarks

1. **BioMysteryBench.** Uses the [official v11 release](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full/commit/b5a889c), with Kimi K3 as judge. K3.1 Pro uses Kimi Code; GLM-5.3 and DeepSeek-V4.1-Flash-1017 use Claude Code; GPT-6 Astra uses Codex. Claude scores are from official system cards.
2. **Agents' Last Exam.** Uses the [official v1.1 release](https://github.com/rdi-berkeley/agents-last-exam), comprising 122 tasks, including 86 CLI tasks. Runs that bypass stated requirements receive zero. DeepSeek and Kimi are evaluated with Claude Code and Kimi Code, respectively; other results are from the official benchmark website.
3. **OfficeQA Pro and OfficeQA Pro v2.** GPT-6 Astra uses Codex; other models use Claude Code. Each task provides the complete PDF corpus as page images without machine-readable text. External-model trajectories that retrieve reference answers online receive zero for those tasks.
4. **Frontier Finance.** Evaluated with the official Finance Agent v2 harness.
5. **HANDBOOK.md.** Evaluated with the official OpenHands-based harness, using tasks from the benchmark's early open-source v1 release.
6. **Harvey Legal Agent.** Evaluated on 1,251 public tasks with Artificial Analysis's Stirrup harness. The reported score is the all-pass rate: the percentage of tasks passing every rubric criterion.
7. **OSWorld V2.1.** Kimi models use the Kimi Agent harness in the official OSWorld v2 repository. GPT-6 Astra's score is taken from Muse Wonder's technical report.
8. **DECK-Arena and Design Bench.** DECK-Arena evaluates presentation generation on real-world queries; Design Bench covers graphic design through direct and reference-image few-shot prompting. Both evaluate python-pptx, pptxjs, and Kimi's in-house DSL in a harness with integrated search and image generation. The automated grader is calibrated on more than 10,000 human judgments.
9. **BrowseComp.** Uses context compaction triggered at 200K tokens and a 500-step maximum. If the limit is reached, the agent is forced to answer before judging. All runs use `temperature = 1.0` and `top-p = 1.0`.
10. **GDPval-AA v2.1, AA-Briefcase v1.1, and AutomationBench-AA.** Scores are cited from [Artificial Analysis](https://artificialanalysis.ai/). **APEX-Agents.** Scores are cited from the [official leaderboard](https://www.mercor.com/apex/apex-agents-leaderboard/).
11. **Tool-augmented multimodal evaluations.** GDP.pdf w/ python, ZeroBench w/ python, and ChartoGraphy w/ python are averaged over five runs; other multimodal benchmarks with Python are averaged over four. For GDP.pdf w/ python, each original PDF is sent to a sandbox for IPython processing. Responses are graded against each rubric criterion using Gemini-3.5-Flash at medium reasoning effort and the official judging prompt. The reported score is the task-macro mean rubric pass rate.

#### Reasoning, knowledge, and multimodal benchmarks

1. **GDP.pdf.** Each PDF is provided as page images resized to a longest edge of 1,152 pixels, interleaved with Docling-extracted text in page order. Responses are graded against each rubric criterion using Gemini-3.5-Flash Medium and the official judging prompt. The table reports the strict pass rate among fully scored responses. GLM-5.3 Max and DeepSeek-V4.1-Flash-1017 Max scores are cited from [Artificial Analysis](https://artificialanalysis.ai/).
2. **HLE-Full and HLE-Diamond.** The full evaluation sets contain 2,500 and 1,000 questions, respectively. For scores marked `*`, accuracy is reported on the text-only subset.
3. **Run counts.** GDP.pdf, ZeroBench, and ChartoGraphy are averaged over five runs; other multimodal benchmarks are averaged over four.
4. **MMMU-Pro.** Evaluated using the official protocol, preserving the original input order and prepending images to the text input.
5. **PerceptionBench.** An in-house benchmark focused on atomic visual perception capabilities.
6. **Kimi Hallucination Benchmark.** Evaluates factual accuracy on an internal set of challenging, de-identified prompts drawn from practical work tasks. The table reports claim-level hallucination rates assessed by fact-checking agents.

#### Cybersecurity benchmarks

1. **ExploitBench.** K3.1 Pro is evaluated with the official harness under a 300-turn budget. The table reports the pass@5 maximum-capability score: for each environment, the best of five seeds is scored by capability flags reached out of 16; arbitrary code execution counts as all 16 capabilities. Comparison scores are from [OpenAI](https://openai.com/index/gpt-6-astra/), [Anthropic](https://www.anthropic.com/claude-opus-5-5-system-card), and [GLM-5.3](https://z.ai/blog/glm-5.3).
2. **ExploitGym.** K3.1 Pro is evaluated with Kimi Code under a six-hour wall-clock budget per task. GPT-6 Astra, Claude, and GLM comparison scores are taken from their respective official reports, with Claude results using the same six-hour budget.
3. **xdev-bench-hard.** K3.1 Pro is evaluated with Kimi Code; GLM-5.3 and DeepSeek-V4.1-Flash-1017 use Claude Code. The internal benchmark contains expert-curated, source-code-based vulnerability discovery and exploitation tasks. The final model is evaluated at pass@3 over a random subset of runs, without hard timeouts or output-token limits.

</details>

## 4. Deployment and Availability

> [!Note]
> You can access Kimi K3.1 Pro's API on https://platform.kimi.ai by selecting `kimi-k3.1-pro`, and we provide OpenAI/Anthropic-compatible API for you.
> Currently, Kimi K3.1 Pro is recommended to run on the following inference engines:

- [vLLM](https://github.com/vllm-project/vllm) — see [recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro)
- [SGLang](https://github.com/sgl-project/sglang) — see [cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro)
- [TokenSpeed](https://github.com/lightseekorg/tokenspeed) — see [recipes](https://lightseek.org/tokenspeed/recipes/models#kimi-k3-1-pro)

---
## 5. Model Usage
The thinking effort of Kimi K3.1 Pro is configured with the top-level `reasoning_effort` request field, which supports `"low"`, `"high"`, and `"max"` (default `"max"`).

For multi-turn conversations and tool calls, Kimi K3.1 Pro requires the complete assistant message returned by the API to be passed back to `messages` as-is — including `reasoning_content` and `tool_calls`, not just `content`:

```python
import openai

def chat_with_preserved_thinking(client: openai.OpenAI, model_name: str):
    messages = [
        {
            "role": "user",
            "content": "Tell me three random numbers."
        },
        {
            "role": "assistant",
            "reasoning_content": "I'll start by listing five numbers: 473, 921, 235, 215, 222, and I'll tell you the first three.",
            "content": "473, 921, 235"
        },
        {
            "role": "user",
            "content": "What are the other two numbers you have in mind?"
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        reasoning_effort="max",
    )
    # the assistant should mention 215 and 222 that appear in the prior reasoning content
    print(f"response: {response.choices[0].message.reasoning}")
    return response.choices[0].message.content
```
For full guides and examples (vision input, structured output, partial mode, tool choice, dynamic tool loading, context caching), see the [Kimi K3.1 Pro Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart) and [Thinking Effort](https://platform.kimi.ai/docs/guide/use-thinking-effort).


## 6. License

Kimi K3.1 Pro is released under the [Kimi K3.1 Pro License](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE).

## 7. Contact Us

If you have any questions, please reach out at [support@moonshot.ai](mailto:support@moonshot.ai).