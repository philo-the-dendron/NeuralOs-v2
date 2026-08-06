# NeuralOS v2 — Competitive & Related-Work Landscape (2026-08-05)

> StandardResearch, 3 agents cross-checked. Topic: who else is building what
> NeuralOS is building, so we don't reinvent wheels and know our peers.
> Every URL in this doc was fetched this session.

## TL;DR — six discoveries that change positioning

1. **Intel Lava (the i16 fixed-point SNN framework) was ARCHIVED 2026-05-13.** It used the same i16 design axis as NeuralOS. **There is now a vacuum** — NeuralOS can position as Lava's spiritual successor and inherit its community.

2. **The "Neuromorphic OS" thesis was published independently** (arXiv:2603.26722, Cheng/Buyya March 2026) — calls explicitly for "a standardized Neuromorphic OS as the foundational layer for a ubiquitous, energy-autonomous Green Cognitive Substrate." **NeuralOS is the implementation of an academic call that exists.** Cite this paper for positioning.

3. **No public Huawei ternary work exists** (5 arXiv queries → 0 hits; Noah's Ark GitHub 23 repos → none ternary). The publicly visible Chinese ternary leader is **Tencent's AngelSlim team** (Sherry 1.25-bit, Tequila ternary). Principal's "Huawei" claim remains private intelligence — do not cite without his acknowledgement of the public gap.

4. **Ternary SNN × LLM bridge is empty.** 24 ternary-SNN papers and 32+ ternary-LLM papers exist, but they **don't cite each other and share no code**. Nobody unifies ternary weights {-1,0,+1} with ternary *spiking* neurons in one runtime. **NeuralOS can be that bridge.**

5. **The "no_std SNN + cipherpunk + RISC-V" combination does not exist anywhere** — confirmed by absence across the sovereign-AI GitHub topic (510 repos), the spiking-neural-network topic (117 repos, ~60% Python), and the Rust SNN ecosystem. **Genuinely unoccupied territory.**

6. **Kraken SoC** (ETH Zurich Benini group, 22nm RISC-V) ships a hardware TNN engine on ColibriUAV drones — **direct precedent for what NeuralOS targets**. ETH Zurich is the natural collaboration target.

## Cross-checked validations [HIGH]

| Claim | Status |
|---|---|
| Rust no_std fixed-point SNN gap | ✅ confirmed (only 2 Rust SNN repos total; zero no_std) |
| Microsoft BitNet is the public ternary leader | ✅ confirmed (b1.58 → v2 → BitDistill → Sparse-BitNet → BITEMBED) |
| Intel China AI Lab (Anbang Yao) is runner-up | ✅ confirmed (CAT-Q ICML 2026 oral, IntelChina-AI/BitTern repo) |
| Prism ML Bonsai is the production deploy leader | ✅ confirmed (48k/mo downloads on 1.7B alone) |
| Sovereign-AI is a real movement | ✅ confirmed (510 repos under `sovereign-ai` topic) |
| All research-summarization products are SaaS | ✅ confirmed (Elicit/Consensus/SciSpace/Scholarcy all closed) |
| Cardano/Haskell rigor in ML is a real gap | ✅ confirmed (Fiat Crypto is methodology peer, not ML) |

## Conflicts

**None material.** All three agents agreed on every material point.

## SNN + neuromorphic landscape — top peers

| Project | Lang | Status | Relevance to NeuralOS |
|---|---|---|---|
| **Intel Lava** | Python | **ARCHIVED 2026-05-13** | Was the i16 fixed-point SNN reference — **NeuralOS inherits the slot** |
| **snnTorch** (UCSC, Eshraghian) | Python | Active, 2.0k★ | Has NIR export — natural interop target |
| **SpikingJelly** (Peking U) | Python | Active V2, 2.1k★ | Most active PyTorch SNN framework; supports NIR |
| **NEST** | C++ | Very active (22k commits) | HPC only — different scale |
| **Brian2** | Python | Active | Computational neuroscience research |
| **Innatera Pulsar** (NL) | Talamo SDK | **Production silicon, RISC-V core** | Closest commercial competitor — but closed SDK tied to proprietary chip |

**NIR (Neuromorphic Intermediate Representation)** is the de-facto interchange format between snnTorch/SpikingJelly/etc. — NeuralOS should produce/consume NIR for instant ecosystem fit.

## Neuromorphic hardware (real, active)

| Chip | Org | Status |
|---|---|---|
| **Kraken SoC** | ETH Zurich Benini | 22nm RISC-V + SNN + TNN engines, deployed on ColibriUAV |
| **Innatera Pulsar** | Innatera NL | **Mass-market neuromorphic MCU**, RISC-V + spiking fabric, CES 2026 launch |
| **BrainChip Akida** | BrainChip AU | Production, IP licensable |
| **SynSense Xylo/Speck** | SynSense CH/CN | Production, backed by Huawei/Samsung/Merck |
| **Intel Loihi 2** | Intel | Pivoting — Lava archived, next-gen SDK promised |
| **BitROM / Platinum ASICs** | Keio / Duke | Ternary-CiROM silicon for BitNet b1.58 |

**Open-source neuromorphic hardware**: Kraken, ReckOn, SENECA, Wenquxing-22A, FeNN — mostly RISC-V based. **9 papers** in 2025-26 combine RISC-V with SNN acceleration.

## Ternary NN landscape (key teams)

**Public leaders:**
- **Microsoft BitNet team** (Ma/Wang/Wei/Huang/Dong) — b1.58 → v2 (4-bit act) → BitDistill → Sparse-BitNet → BITEMBED. Reference model: `BitNet-b1.58-2B-4T`.
- **Intel China AI Lab** (Anbang Yao) — CAT-Q at ICML 2026 oral, `IntelChina-AI/BitTern` repo.
- **Tencent AngelSlim team** (Hong Huang, Dapeng Wu) — Sherry (1.25-bit), Tequila. **The publicly visible Chinese ternary leader.**
- **Prism ML** — production deploy leader (Bonsai-1.7B-27B GGUF/MLX).
- **Academic**: TWN (SJTU 2016), Ternary Spike (Guo AAAI 2024), EdgeRazor (Nanjing/Zhou), BWTA (Tsinghua/Lu), FTerViT (ETH/Magno, on ESP32-S3).

**Huawei specifically:** public record silent. See TL;DR #3.

## Cipherpunk AI / sovereign-AI peers

| Project | Cipherpunk fit | Why relevant |
|---|---|---|
| **LocalAI** (Go, MIT) | 5/5 | p2p/distributed, `formal-verification/` dir, privacy-first |
| **mistral.rs** (Rust, MIT) | 5/5 | Candle-based Rust local LLM server — closest in stack |
| **Llamafile** (Mozilla) | 5/5 | Single-file distribution genius |
| **row-bot** | 5/5 | "Personal AI Sovereignty" — closest peer in ethos |
| **eullm** (Rust+Python) | 5/5 | Sovereign EU-compliant LLMs |
| **Khoj** (AGPL) | 5/5 | Open-source second-brain; research-summarization model |
| **Ollama** | 4/5 | Owned runtime but leaning into cloud ecosystem |
| **Open WebUI** | 3/5 | **Re-licensed away from OSS** — anti-pattern warning |

**Anti-patterns:** Open WebUI (license drift), LM Studio (closed binary), GPT4All (aging without engine), Elicit/Consensus/SciSpace/Scholarcy (SaaS dressed as research tools).

## NeuralOS v2's defensible niche

What nobody else does (cross-verified by absence):

1. **Ternary SNN × hardware-native unified runtime** — bridges two non-citing literatures
2. **Multi-format ternary compat at library layer** — BitNet `Round()`, TWN threshold, Prism Q1_0/Q2_0_g128, NativeTernary, CAT-Q/BitTern, AngelSlim — every team has its own
3. **Verification-first ternary** — nobody is exploiting bit-exact verification (Cardano-grade rigor applied where it's actually tractable)
4. **`no_std` Rust + cipherpunk + RISC-V** — genuinely unoccupied
5. **Offline research-summarizer with owned LLM** — every dedicated product is SaaS; the open path requires assembling Khoj+PrivateGPT+AnythingLLM

## Strategic recommendations

1. **Position as Lava's spiritual successor** in any public communication — instant credibility, fills a documented vacuum.
2. **Cite arXiv:2603.26722 (Cheng/Buyya "Neuromorphic OS")** in v2 README — academic validation of the framing.
3. **Support NIR import/export** from day one — instant interop with snnTorch + SpikingJelly communities.
4. **Reach out to ETH Zurich Benini / Rutishauser** — they're building the RISC-V SNN hardware (Kraken, ColibriUAV, ReckOn) that NeuralOS could target. Natural collaboration.
5. **Multi-format ternary spec** — adopt NativeTernary (2.000 bpw, 2026 paper) as the v2 wire format; track Prism Q1_0/Q2_0_g128 and BitNet `Round()` as import paths.
6. **Verification-first ternary** as the unique research angle — pair with Fiat Crypto methodology for formally-verified ternary ops.
7. **The app's wedge** is "offline, owned, cipherpunk research summarizer" — beat Elicit/Consensus by being the only non-SaaS option.

## Format compat targets (prioritized)

| Format | Origin | Priority |
|---|---|---|
| **BitNet `Round()` ternarization** | Microsoft b1.58 | Must — de facto reference |
| **TWN threshold (Euclidean-min)** | SJTU 2016 | Must — original |
| **GGUF Q1_0 (g128)** — 1-bit | Prism ML Bonsai | Must — largest deploy base |
| **GGUF Q2_0_g128 / Ternary-Bonsai** | Prism ML | Must — production ternary format |
| **NativeTernary (2.000 bpw)** | Savdhariya 2026 | Should — best-in-class wire format |
| **CAT-Q / BitTern PTQ format** | Intel China | Should — ICML-2026 blessed |
| **AngelSlim / Tequila / Sherry** (3:4 sparse 1.25-bit) | Tencent | Should — Chinese deploy stack |
| **MLX 1-bit / 2-bit g128** | Prism ML (Apple) | Nice-to-have — macOS/iOS reach |

## Honest gaps / caveats

- **Huawei public silence is real** — not a search failure. The principal's direct knowledge stands unverified publicly; do not contradict him, present the gap.
- **Prism ML team identity private** — 13 members, no public roster verified.
- **IBM NorthPole 2026 status** — IBM blog URLs 404'd; chip exists, current state unverifiable this session.
- **SenseTime** not searched (budget).
- **Mythic** (analog AI) — current status unverified this session.

## Full agent reports

The three full agent reports live in `docs/landscape/` for reference:
- `01_snn_neuromorphic_landscape.md`
- `02_ternary_teams_quantization.md`
- `03_cipherpunk_ai_sovereignty.md`
