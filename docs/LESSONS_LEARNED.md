# NeuralOS v0.1 — Lessons Learned

> Retour d'expérience sur l'état cassé du repo NeuralOS avant le pivot vers v2.
> Documenté après l'audit structurel du 2026-08-05 : 411 675 lignes de Rust analysées,
> ~6 000 LOC de valeur réelle identifiées. Ce document fige l'état pour qu'on ne le répète pas.

## TL;DR

Le dépôt NeuralOS v0.1 contient ~411K lignes de Rust dont ~6K seulement implémentent réellement quelque chose. Le reste est du théâtre : fichiers à noms impressionnants avec corps vides, `thread::sleep` déguisé en DMA, hardware hardcoded, modules qui s'invoquent sans compiler. Cette archive capture l'état cassé pour qu'on ne répète pas les mêmes erreurs en v2.

## Les 10 patterns de théâtre identifiés (à ne jamais répéter)

1. **Noms impressionnants, corps vides.** `init_idt()`, `init_pic()`, `init_mmu()` — fonctions vides dans `neuralos-microkernel/src/bare_metal.rs:103-125`. Le nom dit "kernel", le contenu dit "rien".

2. **`thread::sleep` déguisé en DMA/SIMD/NPU.** `kernel/src/neural/hardware_accelerated.rs:554,575,592` — `sleep(30ns)` pour "simuler" un transfert NPU. Le matériel n'est jamais touché.

3. **Hardware hardcoded.** `detect_hardware()` retourne `"Intel Core i9-14900K, 16GB, Loihi 2"` dans `kernel/src/hardware.rs:23-34`. Aucune détection réelle.

4. **"Quantum" qui n'est pas quantique.** `crates/neuralos-core/src/quantum_effect.rs` — réimplémentation de TypeScript Effect-TS, n'a rien à voir avec l'informatique quantique.

5. **Appels fantômes inter-modules.** `kernel/src/neural_service_integration.rs:9-12` importe `NeuralRequest`, `NeuralResponse`, `RoutePriority` — aucun n'existe. Le module ne compile pas.

6. **Théâtre de test.** `kernel/src/validation_test.rs` — `let system_functional = true; println!("All 10 tests passed!")`. Ne valide rien.

7. **Pas de CI.** Le workspace est cassé (`multiple workspace roots` dans `libneuralos/Cargo.toml:17`), donc aucun check n'aurait passé même s'il y en avait eu.

8. **Multi-copies parallèles.** 4 copies de `libneuralos/` : la courante, `libneuralos_before_bridge_removal/`, `libneuralos_backup_20250926_230200/` (byte-identique à la précédente), `libneuralos_clean/` (rewrite abandonné). Trois mortes.

9. **Décharge de work-product.** 87 fichiers `.md` en MAJUSCULES (`ACCOMPLISHMENTS.md`, `CELEBRATION.md`, `MISSION_ACCOMPLISHED.md`) — output de sessions IA, pas de la documentation.

10. **Stub-theater à grande échelle.** `libneuralos/src/ai_threat_detection.rs` (2 391 LOC, 14 retours `Ok(Vec::new())`), `libneuralos/src/p2p_foundation.rs` (clés cryptographiques = `vec![1, 2, 3, 4]`), `libneuralos/src/security_audit.rs` (findings pré-écrits hardcoded). Chaque fichier a un nom impressionnant et un corps qui ne fait rien.

## Les ~6K LOC réelles identifiées (à porter en v2)

| Source | LOC | Pourquoi réel |
|---|---|---|
| `libneuralos/src/core/neural_processing/` | ~3 500 | LIF neuron i16 fixed-point, STDP plasticity, sparse synapse matrix (CSR), 4 topologies (random/small-world/feedforward/balanced). Math vraie, tests vrais. |
| `libneuralos_before_bridge_removal/src/core/lock_free_*.rs` + `simd_vectorization.rs` | ~2 500 | AtomicF32 via AtomicU32+bits, CAS-based spike generation (pas de double-spike), 101 intrinsics AVX2 vectorisant le LIF. Perdu dans la coupe du bridge — à récupérer. |
| `crates/neuralos-core/src/foundation/events/event_bus.rs` | ~430 | Pub/sub `no_std` propre (`heapless::Deque` + `spin::Mutex`). Le morceau le mieux ingénié du dépôt. |
| `crates/neuralos-core/src/model_architecture.rs` | ~280 | Wrapper candle BERT réel — charge safetensors, appelle `forward()`, extrait embeddings. Le seul endroit qui fait vraiment du ML. |
| `crates/neuralos-core/src/privacy_encryption.rs::HardwareBoundKey` | ~90 | AES-256-GCM via `ring`, nonce-prepend, round-trip testé. (Sera remplacé en v2 par ChaCha20-Poly1305 owned, voir cipherpunk stance.) |
| `src/infrastructure/repositories/sqlite/` | ~809 | Persistance sqlx réelle — CREATE TABLE / INSERT / SELECT / DELETE. L'unique couche qui fait ce qu'elle dit. |

## Leçons pour NeuralOS v2

1. **Chaque ligne ship avec un test ou un test vector.** Pas de `Ok(())` avec "// In a real implementation".
2. **CI gating dès le jour 1.** Avant tout code, `cargo check` doit passer au niveau workspace.
3. **Un seul nom, une seule source de vérité.** Pas de copies parallèles, pas de backups dans le repo (les backups restent sur le filesystem local, hors git).
4. **La structure mérite son existence.** Pas de scaffolding de 6 crates vides — on ajoute une crate quand elle a du code réel à contenir.
5. **La documentation décrit le but et l'API, pas l'historique des sessions.** Pas de ALL_CAPS work-product dumps.
6. **Noms honnêtes.** Si une fonction s'appelle `init_mmu`, elle initialise le MMU. Sinon, elle s'appelle `todo_mmu` ou elle n'existe pas.
7. **Hexagonal/DDD au service du code, pas l'inverse.** La structure existe parce qu'il y a du code à structurer — pas l'inverse.
8. **Pas de "recopie pour explorer".** Les 4 copies parallèles de libneuralos sont nées d'un refactor abandonné. Refactor in-place, commit souvent, rollback via git pas via duplicates.

## Scope v2 (post-archive)

| Phase | Ships | Effort |
|---|---|---|
| **0** | Crate unique `neuralos-snn`, port du LIF neuron avec tests property-based | 1 session |
| **1** | Port complet SNN (STDP, synapse, topologies, lock-free, SIMD depuis le backup) | 2-3 sessions |
| **2+** | Crypto owned, RISC-V target (QEMU first), app de résumé de recherche, distro Debian | ouverte |

L'ordre des priorités suivant la tête du principal : **library = #1, distro = #2, microkernel+app tied = #3.**

Ottawa RISC-V (August 2026) reste un track parallèle potentiel, pas le driver.

## Référence

- Audit structurel : `NEURALOS_AUDIT_NOTES.md`
- Rapports par-crate (dispatchés en parallèle le 2026-08-05) : libneuralos, kernel, neuralos-core, neuralos-microkernel, src/packages — chaque verdict cite `file:line` pour toute claim.
- Backups filesystem locaux (intacts, ne pas toucher) : `/home/student/projets/NeuralOS-backup-20250915_*` (×3)
