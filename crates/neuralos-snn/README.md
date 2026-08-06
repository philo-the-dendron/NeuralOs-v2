# neuralos-snn

> Spiking Neural Network library — `no_std`, `i16` fixed-point, capacity-boundée.
> Cœur de la sovereignty stack NeuralOS.

## Scope

### Ce que la crate fait (planned, Phase 0+)

- **LIF neurons** — Leaky-Integrate-and-Fire en math entières `i16` (millivolts) / `u32` (microsecondes). Pas de floating-point dans le hot path.
- **STDP learning** — Spike-Timing-Dependent Plasticity, pair-based, avec historique de spikes par neurone.
- **Synapse matrix** — CSR sparse format, capacité fixe, itération O(1) par neurone pré-synaptique.
- **Topology builders** — Random, Small-World (Watts-Strogatz), Feedforward, Balanced (80/20 E/I).

### Ce que la crate NE fait PAS (anti-scope)

- Pas de hardware acceleration. La library fait du compute pur. L'accélération (DMA, SIMD, NPU) vit dans d'autres crates si nécessaire.
- Pas de I/O réseau. Pas de persistence. Pas de UI.
- Pas de drivers. Pas de "OS". La library est une library.
- Pas de crypto. La crypto vit dans `neuralos-crypto` (à venir).
- Pas de modèles LLM. Le ML "conventionnel" (BERT, `DistilBERT`) vit dans `neuralos-ml` (à venir).

## Design constraints

| Constraint | Pourquoi |
|---|---|
| `no_std` par défaut | Permet le déploiement bare-metal RISC-V (ESP32-C3, `HiFive`, QEMU) |
| `i16` fixed-point | Pas de FPU requis — tourne sur microcontrôleurs sans unité flottante |
| Capacité fixe (pas d'alloc) | Prévisible en mémoire — critique pour embedded |
| Une seule impl par concept | Leçon v0.1: 3 copies parallèles du SNN dans le repo original |

## Status

**Scaffold.** Le code réel arrive Phase 0 (prochaine session):

1. Port du LIF neuron depuis `NeuralOS/libneuralos/src/core/neural_processing/lif_neuron.rs` (v0.1)
2. Tests property-based (proptest)
3. Fix du bug `current_time()` identifié dans l'audit (lif_neuron.rs:278-281)

## License

AGPL-3.0-or-later. Voir `LICENSE` à la racine du workspace.
