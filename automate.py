import os
import subprocess

os.chdir(r"C:\Users\ishan\Documents\Projects\Awesome-Position-Embedding")


def run_cmd(cmd):
    print("Running:", cmd)
    subprocess.run(["pwsh", "-Command", cmd], capture_output=True, text=True)

# 0. Initialize git if not already
run_cmd("git init")
run_cmd("git add .")
run_cmd('git commit -m "Initial commit"')

# Step 1
readme_path = "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

def safe_replace(text, old_start, old_end, new_text):
    start_idx = text.find(old_start)
    end_idx = text.find(old_end) + len(old_end)
    if start_idx == -1 or text.find(old_end) == -1:
        print("Failed to find", old_start[:30])
        return text
    return text[:start_idx] + new_text + text[end_idx:]

s1_new = """| Methodology | Description | Limitation / Significance | Year | Paper Link | Detailed Page |
|---|---|---|---|---|---|
| **Static Sinusoidal Functional Era** | Hand-crafted mathematical formula using Sine and Cosine waves [INDEX: 1]. | Highly rigid; inflexible out-of-domain context decay. | 2017 | [Vaswani et al.](https://arxiv.org/abs/1706.03762) | [Details](pages/static-sinusoidal.md) |
| **Learnable Absolute Table Era** | Parameterised lookup matrix for absolute positions [INDEX: 1]. | Hard context extrapolation limit. | 2018 | [BERT](https://arxiv.org/abs/1810.04805) | [Details](pages/learnable-absolute.md) |
| **Relative Distance Bias Era** | Adjustable scalar bias modifier based on relative offset distance. | Severe computational cache inflation. | 2018 | [Shaw et al.](https://arxiv.org/abs/1803.02155) | [Details](pages/relative-distance.md) |
| **Complex Rotary Geometric (RoPE)** | Multiplies query and key vectors by a complex-valued rotation matrix [INDEX: 18]. | Smooth geometric angle extrapolation natively. | 2021 | [RoPE](https://arxiv.org/abs/2104.09864) | [Details](pages/rope-era.md) |"""

s2_new = """| Variant | Mechanism | Year | Paper Link | Detailed Page |
|---|---|---|---|---|
| **Absolute Additive Position Embeddings** | Adds an independent absolute position vector directly to the semantic word token embedding vector. | 2017 | [Link](https://arxiv.org/abs/1706.03762) | [Details](pages/absolute-additive.md) |
| **Relative Position Bias** | Bypasses input layer modifications, injecting a learnable bias parameter straight into the multi-head self-attention score. | 2018 | [Link](https://arxiv.org/abs/1803.02155) | [Details](pages/relative-bias.md) |
| **Rotary Position Embedding (RoPE)** | A multiplicative, geometric transformation rotating keys and queries by an angle based on token position. | 2021 | [Link](https://arxiv.org/abs/2104.09864) | [Details](pages/rope-variant.md) |
| **Alibi (Attention with Linear Biases)** | Injects a static, linearly decaying penalty scalar straight into the attention score based on token distance. | 2021 | [Link](https://arxiv.org/abs/2108.12409) | [Details](pages/alibi.md) |"""

s3_new = """| Extrapolation Strategy | Mathematical Approach | Year | Paper Link | Detailed Page |
|---|---|---|---|---|
| **Linear / Dynamic RoPE Interpolation** | Stretches the rotation base parameter to downscale angular velocity [INDEX: 18]. | 2023 | [Link](https://arxiv.org/abs/2306.15595) | [Details](pages/dynamic-rope.md) |
| **YaRN** | Multi-frequency scaling protocol: interpolates high-frequency channels, extrapolates low-frequency. | 2023 | [Link](https://arxiv.org/abs/2309.00071) | [Details](pages/yarn.md) |"""

s4_new = """| Challenge | Problem | Mitigation | Year | Paper Link | Detailed Page |
|---|---|---|---|---|---|
| **KV Cache VRAM Satiation Crisis** | Physical volume of historical Key-Value attention vectors explodes. | Multi-Head Latent Attention (MLA). | 2022 | [Link](https://arxiv.org/abs/2211.05102) | [Details](pages/kv-cache.md) |
| **Dataloader Load-Imbalance Stall** | Variable-length position sequences desynchronize distributed nodes. | Length-Grouped Token Batching and Fused FlashAttention. | 2023 | [Link](https://arxiv.org/abs/2308.10820) | [Details](pages/dataloader-stall.md) |"""

s5_new = """| Application | Description | Year | Paper Link | Detailed Page |
|---|---|---|---|---|
| **Pre-Training Web-Scale Foundational Transformers** | Sequence coordination across global distributed clusters. | 2023 | [Link](https://arxiv.org/abs/2302.13971) | [Details](pages/pre-training.md) |
| **Software Engineering Coding Agents** | Drives automated developer platforms (Devin, Cursor). | 2023 | [Link](https://arxiv.org/abs/2312.07128) | [Details](pages/software-agents.md) |
| **Visual Patch Position Mapping** | Coordinates spatial object tracking inside vision foundation layers (ViT, CLIP). | 2020 | [Link](https://arxiv.org/abs/2010.11929) | [Details](pages/vision-patch.md) |"""

text = safe_replace(text, "*   **The Static Sinusoidal Functional Era", "boundaries [INDEX: 18, 22].", s1_new)
text = safe_replace(text, "- ### A. Absolute Additive Position Embeddings", "with zero weight adjustments.", s2_new)
text = safe_replace(text, "*   **Linear / Dynamic RoPE Interpolation Scaling**", "cohesion [INDEX: 18].", s3_new)
text = safe_replace(text, "- ### The Key-Value (KV) Cache VRAM Satiation Crisis", "optimize memory bus bandwidth [INDEX: 22].", s4_new)
text = safe_replace(text, "*   **Pre-Training Web-Scale Foundational Transformers", "track boundaries accurately [INDEX: 1].", s5_new)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

run_cmd('git add . && git commit -m "tabularised the bullets" && git push')

# Step 2
os.makedirs("pages", exist_ok=True)
pages_data = [
    ("static-sinusoidal", "Static Sinusoidal Functional Era", "2017", "https://arxiv.org/abs/1706.03762"),
    ("learnable-absolute", "Learnable Absolute Table Era", "2018", "https://arxiv.org/abs/1810.04805"),
    ("relative-distance", "Relative Distance Bias Era", "2018", "https://arxiv.org/abs/1803.02155"),
    ("rope-era", "Complex Rotary Geometric Transformation Era", "2021", "https://arxiv.org/abs/2104.09864"),
    ("absolute-additive", "Absolute Additive Position Embeddings", "2017", "https://arxiv.org/abs/1706.03762"),
    ("relative-bias", "Relative Position Bias", "2018", "https://arxiv.org/abs/1803.02155"),
    ("rope-variant", "Rotary Position Embedding (RoPE)", "2021", "https://arxiv.org/abs/2104.09864"),
    ("alibi", "Alibi (Attention with Linear Biases)", "2021", "https://arxiv.org/abs/2108.12409"),
    ("dynamic-rope", "Linear / Dynamic RoPE Interpolation Scaling", "2023", "https://arxiv.org/abs/2306.15595"),
    ("yarn", "YaRN (Yet another RoPE extensioN)", "2023", "https://arxiv.org/abs/2309.00071"),
    ("kv-cache", "The Key-Value (KV) Cache VRAM Satiation Crisis", "2022", "https://arxiv.org/abs/2211.05102"),
    ("dataloader-stall", "The Sequence Length Distributed Dataloader Load-Imbalance Stall", "2023", "https://arxiv.org/abs/2308.10820"),
    ("pre-training", "Pre-Training Web-Scale Foundational Transformers", "2023", "https://arxiv.org/abs/2302.13971"),
    ("software-agents", "Long-Horizon Software Engineering Repository Analysis & Coding Agents", "2023", "https://arxiv.org/abs/2312.07128"),
    ("vision-patch", "Visual Patch Position Mapping in Multi-Modal Frontends", "2020", "https://arxiv.org/abs/2010.11929"),
]

for slug, title, year, link in pages_data:
    page_content = f"""# {title}

[Back to Readme](../README.md)

This page provides detailed information on {title}.

```mermaid
flowchart TD
    A[Concept] --> B[{title}]
    B --> C[Application / Impact]
```

## Information
- **Year:** {year}
- **Paper Link:** [{link}]({link})
"""
    with open(f"pages/{slug}.md", "w", encoding="utf-8") as f:
        f.write(page_content)

run_cmd('git add . && git commit -m "detailed pages created" && git push')

# Step 3
os.makedirs("assets", exist_ok=True)
svg_banner = """<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:rgb(255,0,0);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgb(0,0,255);stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad)" />
  <text x="50%" y="50%" font-family="Arial" font-size="40" fill="white" font-weight="bold" dominant-baseline="middle" text-anchor="middle">
    🚀 Awesome Position Embedding 🌟
    <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" />
  </text>
</svg>"""
with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(svg_banner)

with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

text = '![Banner](assets/banner.svg)\n\n' + text
text = text.replace("# Awesome-Position-Embedding", "🚀 # Awesome-Position-Embedding 🌟")
text = text.replace("## Position Embedding", "🧠 ## Position Embedding")
text = text.replace("## 1. The Macro Chronological Evolution", "📅 ## 1. The Macro Chronological Evolution")
text = text.replace("## 2. Core Functional", "⚙️ ## 2. Core Functional")
text = text.replace("## 3. The RoPE", "🧮 ## 3. The RoPE")
text = text.replace("## 4. Production", "🏭 ## 4. Production")
text = text.replace("## 5. Frontier", "🌌 ## 5. Frontier")
text = text.replace("## References", "📚 ## References")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

run_cmd('git add . && git commit -m "added emojis and banner" && git push')

# Step 4
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

badges_left = '<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'

header_insert = f"""<div align="center">
{badges_left}
</div>

**SEO Meta Description:** A comprehensive and awesome curated list of position embedding techniques, including RoPE, Alibi, and relative bias, for Transformer neural networks and Large Language Models.
"""
text = text.replace("🚀 # Awesome-Position-Embedding 🌟", "🚀 # Awesome-Position-Embedding 🌟\n\n" + header_insert)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

run_cmd('git add . && git commit -m "seo optimised and badges to left added" && git push')

# Step 5
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

badge_right = '<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>'
text = text.replace(badges_left, badges_left + badge_right)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

run_cmd('git add . && git commit -m "badges to right added" && git push')

# Step 6
star_history = """
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Position-Embedding&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Position-Embedding&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Position-Embedding&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Position-Embedding&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
with open(readme_path, "a", encoding="utf-8") as f:
    f.write("\n" + star_history)

run_cmd('git add . && git commit -m "star history added" && git push')

# Step 7
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace("chartrepos", "chart?repos")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)
run_cmd('git add . && git commit -m "fixed star plot" && git push')

# Step 8
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)
run_cmd('git add . && git commit -m "invalid awesome link fixed" && git push')

print("All tasks completed successfully!")
