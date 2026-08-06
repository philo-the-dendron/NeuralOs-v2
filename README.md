# NeuralOS v2

> **Sovereignty stack** — open ISA, owned code, real crypto, local AI.
> Bibliothèque SNN au centre, distro Debian, microkernel RISC-V, app de résumé de recherche.

## Ce que c'est

NeuralOS v2 est une sovereignty stack en Rust. Pas un "AI OS" — une pile logicielle
ouverte dont chaque couche t'appartient : code lisible, crypto auditable, modèles qui
tournent en local, ISA matérielle ouverte (RISC-V).

Le projet original (NeuralOS v0.1, archivé au tag `v0.1-broken-baseline` dans
[`Caramoussin/NeuralOs`](https://gitea.com/Caramoussin/NeuralOs)) a accumulé 411K
LOC dont seulement ~6K réelles. Le reste était du théâtre : fonctions vides,
`thread::sleep` déguisé en DMA, hardware hardcoded. v2 ne répète pas ces erreurs.

Voir [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) pour les 10 patterns de
théâtre identifiés à ne jamais répéter.

## Priorité (de la tête du principal)

| # | Composant | Status |
|---|---|---|
| **1** | `neuralos-snn` — bibliothèque SNN `no_std` | Scaffold. Phase 0: port LIF neuron. |
| 2 | Distro Debian (custom Live ISO via `live-build`) | Pas commencé. |
| 3/4 | Microkernel RISC-V (QEMU first) | Pas commencé. |
| 3/4 | App de résumé de recherche (Tauri v2 + DistilBERT) | Pas commencé. |

## Principes (Cardano-grade rigor)

1. **Chaque ligne ship avec un test ou un test vector.** Pas de `Ok(())` avec "// In a real implementation."
2. **CI gating dès le jour 1.** `cargo check` doit passer au niveau workspace.
3. **Une seule source de vérité.** Pas de copies parallèles (leçon v0.1).
4. **Structure mérite son existence.** On ajoute une crate quand elle a du code réel — pas de scaffolding de 6 crates vides.
5. **Hexagonal/DDD au service du code, pas l'inverse.**
6. **Noms honnêtes.** Si une fonction s'appelle `init_mmu`, elle initialise le MMU.
7. **`no_std` par défaut** pour la library principale — contrainte qui force la discipline et permet le déploiement RISC-V.
8. **Pas de dépendance cloud.** Local AI only. Pas d'API OpenAI/Anthropic.

## Workspace

```
NeuralOs-v2/
├── Cargo.toml                # workspace, un seul member pour l'instant
├── crates/
│   └── neuralos-snn/         # bibliothèque SNN (scaffold)
└── docs/
    ├── ARCHITECTURE.md       # vision multi-target
    ├── AUDIT_PORT_TABLE.md   # ce qu'on porte depuis v0.1 (avec file:line)
    └── LESSONS_LEARNED.md    # les 10 patterns de théâtre
```

## Quickstart

```bash
git clone git@gitea.com:Caramoussin/NeuralOs-v2.git
cd NeuralOs-v2
cargo check
cargo test
```

## License

AGPL-3.0-or-later. La full text sera ajoutée au workspace avant le premier tag stable.

## Relation avec NeuralOS v0.1

[`Caramoussin/NeuralOs`](https://gitea.com/Caramoussin/NeuralOs) est l'archive.
Tag [`v0.1-broken-baseline`](https://gitea.com/Caramoussin/NeuralOs/releases/tag/v0.1-broken-baseline)
fige l'état cassé. Les gems à porter (LIF neuron, STDP, lock-free primitives, SIMD
kernel, event bus, candle BERT wrapper, AES-GCM key, SQLite repos) sont listés dans
[`docs/AUDIT_PORT_TABLE.md`](docs/AUDIT_PORT_TABLE.md) avec leurs `file:line`.
