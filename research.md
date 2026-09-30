---
layout: default
title: Research Overview
---

# Research Overview

This page collects selected technical work from my research, open-source projects, and engineering experiments. The throughline is practical machine learning systems: making models more reliable, explainable, efficient, and useful in settings where the surrounding system matters as much as the model itself.

## Selected Work

### AI Security, Robustness, and Trustworthy ML

- [Forensic Energy Map](https://github.com/paperwhite/forensic-energy-map)  
  Real or synthetic image classification using spatial forensic energy and noisiness maps that expose low-level traces of AI image generation.

- [Adversarial Prompt Guard](https://github.com/paperwhite/adversarial-prompt-guard)  
  A small guardrail project for screening prompts and identifying adversarial inputs.

- [Polaris](https://github.com/paperwhite/Polaris)  
  A Python tool focused on preventing cross-site scripting attacks.

- [ARMORY](https://github.com/paperwhite/armory) and [Obfuscated Gradients](https://github.com/paperwhite/obfuscated-gradients)  
  Adversarial robustness evaluation and attack/defense research references connected to trustworthy machine learning systems.

### ML Systems and Developer Tooling

- [ML Run Doctor](https://github.com/paperwhite/ml_run_doctor)  
  A training-log diagnosis project for understanding machine learning run behavior.

- [Distributed Training Visualizer](https://github.com/paperwhite/distributed-training-visualizer)  
  A TypeScript visualization project for reasoning about distributed training.

- [Portfolio Project RAG](https://github.com/paperwhite/portfolio-project-rag)  
  A safety-first retrieval augmented generation module for portfolio project Q&A.

- [Spark Perf](https://github.com/paperwhite/spark-perf), [Machine Learning with Spark](https://github.com/paperwhite/Machine-Learning-with-Spark), and [Cluster Ease](https://github.com/paperwhite/Cluster_Ease)  
  Systems-oriented work around distributed analytics, Spark workloads, and cluster workflows.

### Scientific Computing, Geometry, and Medical Imaging

- [Conformal Mapping](https://github.com/paperwhite/Conformal_Mapping)  
  C++ implementation for conformal mapping of 3D meshes onto a sphere using Tutte energy minimization and harmonic energy minimization.

- [Generalised Multidimensional Scaling Method](https://github.com/paperwhite/Generalised-Multidimensional-Scaling-Method)  
  MATLAB work connected to multidimensional scaling and geometric representation.

- [FDG-PET Alzheimer’s Classification](https://github.com/paperwhite/DL_Classification_of_FDG-PET_Regional_Segmentation_PCA_MLP)  
  Deep learning code for classifying FDG-PET images into Alzheimer’s diagnostic categories.

- [BTC](https://github.com/paperwhite/BTC)  
  Bitcoin price prediction work using machine learning methods.

## Selected Work by Theme

<figure>
  <svg viewBox="0 0 720 260" role="img" aria-labelledby="theme-chart-title theme-chart-desc">
    <title id="theme-chart-title">Selected work grouped by research theme</title>
    <desc id="theme-chart-desc">A horizontal bar chart showing four selected works in AI security and trustworthy ML, four in ML systems and developer tooling, and four in scientific computing and applied ML.</desc>
    <style>
      .chart-label { font: 14px sans-serif; fill: #222; }
      .chart-value { font: 13px sans-serif; fill: #333; }
      .axis { stroke: #c9d1d9; stroke-width: 1; }
    </style>
    <line class="axis" x1="210" y1="30" x2="210" y2="220" />
    <text class="chart-label" x="20" y="67">AI security &amp; trustworthy ML</text>
    <rect x="210" y="45" width="360" height="34" fill="#2f6f73" />
    <text class="chart-value" x="585" y="67">4 projects</text>
    <text class="chart-label" x="20" y="127">ML systems &amp; tooling</text>
    <rect x="210" y="105" width="360" height="34" fill="#6f5a8f" />
    <text class="chart-value" x="585" y="127">4 projects</text>
    <text class="chart-label" x="20" y="187">Scientific &amp; applied ML</text>
    <rect x="210" y="165" width="360" height="34" fill="#9b5f3d" />
    <text class="chart-value" x="585" y="187">4 projects</text>
  </svg>
  <figcaption>Selected projects are grouped by the role they play in the broader research story, not by repository count alone.</figcaption>
</figure>

## Public GitHub Snapshot

Public metadata from [github.com/paperwhite](https://github.com/paperwhite) shows 51 repositories in the current query, spanning machine learning, systems, security, scientific computing, and older coursework or experiments.

<figure>
  <svg viewBox="0 0 720 320" role="img" aria-labelledby="language-chart-title language-chart-desc">
    <title id="language-chart-title">Public repository language mix</title>
    <desc id="language-chart-desc">A bar chart of public paperwhite repositories by primary language: Python 13, C++ 5, Java 4, JavaScript 2, Jupyter Notebook 2, Shell 2, and TypeScript 1.</desc>
    <style>
      .bar-label { font: 13px sans-serif; fill: #222; }
      .bar-value { font: 12px sans-serif; fill: #333; }
      .grid { stroke: #e3e6ea; stroke-width: 1; }
    </style>
    <line class="grid" x1="75" y1="250" x2="660" y2="250" />
    <g>
      <rect x="85" y="55" width="58" height="195" fill="#2f6f73" />
      <text class="bar-value" x="104" y="45">13</text>
      <text class="bar-label" x="82" y="276">Python</text>
    </g>
    <g>
      <rect x="180" y="175" width="58" height="75" fill="#6f5a8f" />
      <text class="bar-value" x="202" y="165">5</text>
      <text class="bar-label" x="194" y="276">C++</text>
    </g>
    <g>
      <rect x="275" y="190" width="58" height="60" fill="#9b5f3d" />
      <text class="bar-value" x="298" y="180">4</text>
      <text class="bar-label" x="292" y="276">Java</text>
    </g>
    <g>
      <rect x="370" y="220" width="58" height="30" fill="#4f6d9a" />
      <text class="bar-value" x="393" y="210">2</text>
      <text class="bar-label" x="354" y="276">JavaScript</text>
    </g>
    <g>
      <rect x="465" y="220" width="58" height="30" fill="#7b6f45" />
      <text class="bar-value" x="488" y="210">2</text>
      <text class="bar-label" x="430" y="276">Jupyter Notebook</text>
    </g>
    <g>
      <rect x="560" y="220" width="58" height="30" fill="#7d4e57" />
      <text class="bar-value" x="583" y="210">2</text>
      <text class="bar-label" x="570" y="276">Shell</text>
    </g>
    <g>
      <rect x="635" y="235" width="34" height="15" fill="#49735a" />
      <text class="bar-value" x="647" y="225">1</text>
      <text class="bar-label" x="608" y="300">TypeScript</text>
    </g>
  </svg>
  <figcaption>Primary-language counts exclude repositories with no detected primary language and collapse the long tail for readability.</figcaption>
</figure>

## How the Work Connects

The selected projects form three related lines of inquiry:

- How do we evaluate whether AI systems are robust, secure, and difficult to misuse?
- How do we build better tools for understanding training behavior, distributed ML systems, and retrieval workflows?
- How can machine learning and numerical methods support scientific and biomedical problems?

That combination is the work I want this site to reflect: not only models, but the systems, measurements, and constraints that make models useful.
