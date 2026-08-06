# NeuralOS v2 — Architecture

> Une stack de souveraineté, pas un "AI OS". Quatre cibles de déploiement
> partagent un cœur neural `no_std`.

## Vue d'ensemble

```
                    neuralos-snn  (library #1, no_std, i16 fixed-point)
                          │
           ┌──────────────┼──────────────┐
           │              │              │
      🤖 RISC-V brain  🖥 Debian     📚 App de recherche
       (bare-metal)     (tower PC)    (DistilBERT + fetch/parse/summarize)
```

Library au centre. Trois cibles de déploiement. Library = #1, distro = #2,
microkernel + app = #3 tied. Ottawa RISC-V est un track parallèle, pas le driver.

## Les quatre cibles

### 1. `neuralos-snn` — library (PRIORITÉ #1)

**Ce que c'est:** Spiking Neural Network library en Rust, `no_std`, `i16` fixed-point,
capacité fixe. LIF + STDP + topologies + (Phase 4) ternary weights.

**Pourquoi no_std:** permet le déploiement bare-metal RISC-V (ESP32-C3, HiFive, QEMU)
sans dépendances OS. C'est aussi une contrainte qui force la discipline : pas d'alloc,
pas de panic-handler Magique, pas de unwrap. Prévisible, testable.

**Source:** port propre depuis `NeuralOS/libneuralos/src/core/neural_processing/`
(v0.1) + `libneuralos_before_bridge_removal/src/core/lock_free_*.rs` +
`simd_vectorization.rs`. Voir `AUDIT_PORT_TABLE.md`.

### 2. Distro Debian — "Prime AI" OS (PRIORITÉ #2)

**Ce que c'est:** Debian Testing + meta-package `neuralos-prime` (+ option Live ISO via
`live-build`). Remplace Windows 10 sur la tour. Pas un derivative complet (Kali-tier =
multi-année), juste Debian + nos outils.

**Pourquoi meta-package et pas derivative:** le détail difficile d'une distro, c'est
l'infra (build farm, miroirs, security updates). On hérite de tout ça en restant sur
Debian. On ajoute juste notre couche.

**Scope:**
- Live ISO bootable avec app pré-installée
- APT meta-package qui convertit stock Debian → NeuralOS Prime
- Theming/session custom ("prime AI" identity)
- Outils dev pré-installés (Rust toolchain, candle + modèles cache, Docker, etc.)

### 3. Microkernel RISC-V — robot brain (PRIORITÉ #3/4)

**Ce que c'est:** runtime bare-metal sur RISC-V qui embarque `neuralos-snn` comme
couche de décision. QEMU d'abord (gratuit, aujourd'hui), hardware réel ensuite
(ESP32-C3 ~$5-15, ou SiFive/StarFive si budget).

**Pourquoi pas from-scratch microkernel:** leçon v0.1. Le `neuralos-microkernel/`
original a 82K LOC dont seulement ~1K réelles (le SNN). Le reste : GDT/IDT/MMU vides,
boucle de NULL-deref, drivers fantômes. Écrire un microkernel from scratch est un
tar pit de 10 ans.

**Approche:**
- **QEMU RISC-V (`riscv64gc`)** d'abord — gratuit, tourne sur la tour
- **`esp-rs` + `esp-idf`** pour ESP32-C3 — drivers fournis (GPIO, I²C, Wi-Fi, BLE),
  tous open-source, tous auditables. C'est ça, "own my software" — pas réinventer les
  drivers, utiliser des drivers ouverts que tu peux lire et modifier.
- **OpenSBI + U-Boot** pour boards RISC-V plus grandes

**Anti-scope:** pas de x86 (le `neuralos-microkernel/` v0.1 est x86 + inutile, on drop).

### 4. App de recherche — fetch/parse/summarize (PRIORITÉ #3/4)

**Ce que c'est:** Tauri v2 desktop app. Fetch arXiv/PubMed/RSS → parse → summarize
avec DistilBERT (ou Phi-3-mini / Qwen-2.5-1.5B pour meilleure qualité) via candle.
100% local, pas d'API cloud. Le SNN y entre comme "neuromorphic mode" sidebar
(classification relevance/spam via SNN plutôt que LLM).

**Pourquoi Tauri v2:** un codebase, build Linux + Windows + macOS. Shell léger
(~10MB), pas d'Electron. Backend Rust natif, frontend Svelte/React/Vue.

**Cible utilisateurs:** le principal lui-même, pour le travail universitaire
(résumé de recherche difficile — ex: diabetes research).

## Phases

| Phase | Ships | Effort |
|---|---|---|
| **0** | Scaffold + LIF neuron port avec tests property | cette session + 1 |
| **1** | SNN complet (STDP, synapse, topologies, lock-free, SIMD) | 2-3 sessions |
| **2** | App de recherche v1 (Tauri + DistilBERT + arXiv fetcher) | 4-6 sessions |
| **3** | Debian packaging (`.deb` + Live ISO) | 2-3 sessions |
| **4** | Crypto owned (ChaCha20-Poly1305 + X25519 via crypto-bigint) | 2-3 sessions |
| **5** | Ternary quantization (trit weights + format spec) | 2 sessions |
| **6** | RISC-V target (QEMU first, hardware quand budget) | open-ended |

## Décisions du principal

- **Repo mechanics:** un seul repo nommé `NeuralOS` pour v0.1 (archive),
  `NeuralOs-v2` pour v2. Pas de rename, pas de version dans le nom.
- **Backups:** les 3 backups filesystem locaux (`/home/student/projets/NeuralOS-backup-*`)
  restent intouchés — filet de sécurité.
- **License:** AGPL-3.0-or-later.
- **Langue de communication:** français (préférence du principal), documentation
  technique bilingue.
- **Cardano/Haskell affinity:** signifie rigueur, formal-verification-friendly,
  property tests > unit tests quand possible.
- **Cipherpunk stance:** "own my hardware, own my software, real crypto."
- **Anti-closed-AI:** local models only, jamais d'API OpenAI/Anthropic.

## Ottawa RISC-V (August 2026) — track parallèle

Le principal va à un événement RISC-V à Ottawa ce mois d'août. Pour Ottawa,
le demo potentiel est: **ternary SNN on owned RISC-V hardware with owned crypto.**
Trois threads chauds en un demo. Mais Ottawa ne drive pas la séquence principale —
la library et l'app d'abord. Si Ottawa devient un objectif ferme, on détache
Phases 1+4+5+6 en sub-plan August séparé.

Détails à confirmer par le principal:
- Événement exact (RISC-V Summit? atelier? autre?) + dates
- Talk slot vs hallway track
- Budget hardware (ESP32-C3 ~$5-15 vs SiFive/StarFive ~$50-200)
