# Audit Port Table — ce qu'on porte depuis NeuralOS v0.1

> Source: audit structurel du 2026-08-05 (411K LOC analysées, ~6K réelles identifiées).
> Chaque entrée cite `file:line` dans le repo v0.1 archivé au tag `v0.1-broken-baseline`.

## Phase 0-1: `neuralos-snn` library

| Source v0.1 (path:lines) | LOC | Vers v2 | Notes |
|---|---|---|---|
| `libneuralos/src/core/neural_processing/lif_neuron.rs` | 465 | `crates/neuralos-snn/src/lif_neuron.rs` | **Gem #1.** i16 fixed-point, refractory, adaptation, LFSR noise, builder. Bug à corriger: `current_time()` returns `last_spike_time_us` (:278-281), casse le firing-rate filter. LFSR seed = id seulement (:248) → deterministic, pas noise. |
| `libneuralos/src/core/neural_processing/stdp_plasticity.rs` | 699 | `crates/neuralos-snn/src/stdp.rs` | **Gem #2.** HashMap spike history, fixed-point exp-decay, homeostatic scaling. Bug: inner `calculate_ltp_change(post_neuron_id, pre_spike_time_us)` — le 2ème arg est `current_time_us` du call site (:295-311, 344-365), filter `if post_spike_time > pre_spike_time_us` toujours false. LTP path inert. |
| `libneuralos/src/core/neural_processing/network.rs` | 996 | `crates/neuralos-snn/src/network.rs` | **Gem #4.** CSR `SparseSynapseMatrix`, 4 topologies (random/small-world/feedforward/balanced), LFSR Fisher-Yates, SmallVec batching. 95% byte-identique à `spiking_neural_network/network.rs` — pick UNE. Bug: `update_plasticity_optimized` (:502-525) même sign-bug que STDP. |
| `libneuralos/src/core/spiking_neural_network/synapse.rs` | 378 | `crates/neuralos-snn/src/synapse.rs` | **Gem #3.** Synapse avec conductance, eligibility trace, activity counter. `STDPRule::calculate_weight_change(dt_us)` (:263-283) = vraie exp decay fixed-point. **Meilleure** impl que `stdp_plasticity.rs` — pick celle-ci. |
| `libneuralos/src/core/spiking_neural_network/lif_neuron.rs` | 206 | DROP | Doublon de `neural_processing/lif_neuron.rs` avec types incompatibles (`id: usize` vs `u16`). Confusion de collision. |
| `libneuralos/src/core/spiking_neural_network/network.rs` | 997 | DROP | Doublon byte-identique de `neural_processing/network.rs`. |
| `libneuralos/src/utils/mod.rs` | 94 | inline dans les modules qui en ont besoin | Helpers triviaux (sigmoid/tanh/relu/now_us). Réimplementer en 30 LOC. |
| `libneuralos/src/lib.rs` | 47 | rewrite | Module wiring + error enum. Reconstruire propre. |

**Total Phase 0-1 SNN port:** ~2 500 LOC clean (après collapse des doublons + fix des bugs).

## Phase 1 (extension): lock-free + SIMD depuis le backup

Ces sources sont dans `libneuralos_before_bridge_removal/` (le backup pré-refactor,
77K LOC, perdues dans la coupe du bridge — à récupérer).

| Source v0.1 (path) | LOC | Vers v2 | Notes |
|---|---|---|---|
| `libneuralos_before_bridge_removal/src/core/lock_free_neural_processing.rs` | 596 | `crates/neuralos-snn/src/concurrent/atomic_lif.rs` | **Top find.** AtomicF32 via AtomicU32+bits (:100-149). CAS-based spike generation (:257-290) — `compare_exchange` sur membrane potential prévient double-spiking cross-thread. Vraie concurrence. |
| `libneuralos_before_bridge_removal/src/core/lock_free_neural_network.rs` | 710 | `crates/neuralos-snn/src/concurrent/pool.rs` | Thread-pool spike processors (`thread::spawn` loop :463-477). `LockFreeSpikeBuffer` MPMC queue. |
| `libneuralos_before_bridge_removal/src/core/simd_vectorization.rs` | 989 | `crates/neuralos-snn/src/simd/avx2_lif.rs` | **Top find.** 101 intrinsics `_mm256_*`/`_mm_*` AVX2. Vectorise LIF integration (:230-271): load 16 i16 neurons, widen i32, compute, clamp, pack back. À révectoriser pour RISC-V vector extension (Phase 6). |
| `libneuralos_before_bridge_removal/src/core/neural_optimization.rs` | 1 460 | `crates/neuralos-snn/src/pruning.rs` (partial) | Magnitude pruning (:400-435) + structured pruning (L2-norm, top-k :438-483) = 2 stratégies réelles sur 7. Les 5 autres = TODO stubs. |

**Total Phase 1 extension:** ~2 400 LOC de valeur réelle (concurrent + SIMD + pruning).

## Phase 2: app de recherche — sources hors-scaffold

| Source v0.1 (path) | LOC | Vers v2 | Notes |
|---|---|---|---|
| `crates/neuralos-core/src/model_architecture.rs` | 279 | `crates/neuralos-ml/src/model.rs` | Wrapper candle BERT réel. Charge safetensors, `forward()`, extrait embeddings. Ajouter DistilBERT (manuel — code actuel est BERT seulement). |
| `src/infrastructure/repositories/sqlite/` | 809 | `crates/neuralos-persistence/src/sqlite/` | Real sqlx persistence (Model/Task/NPU/Batch). |
| `src/domain/` | 592 | `crates/neuralos-domain/src/` | DDD entities + repository/service traits. |
| `src/infrastructure/repositories/memory/` | 281 | `crates/neuralos-persistence/src/memory/` | In-memory doubles pour tests. |

## Phase 4: crypto owned

| Source v0.1 (path) | LOC | Vers v2 | Notes |
|---|---|---|---|
| `crates/neuralos-core/src/privacy_encryption.rs::HardwareBoundKey` | ~90 | `crates/neuralos-crypto/src/aead.rs` (référence) | Référence pour l'API. **On remplace l'impl ring par from-scratch ChaCha20-Poly1305** (cipherpunk stance). ring = bien mais pas "owned." |

## Sources explicites à NE JAMAIS porter

| Path | Pourquoi |
|---|---|
| `libneuralos/src/ai_threat_detection.rs` | 14 retours `Ok(Vec::new())`. 200 structs impressionnantes, 0 logique. |
| `libneuralos/src/p2p_foundation.rs` | Keys = `vec![1, 2, 3, 4]`. Network ops = `println!() + Ok(())`. |
| `libneuralos/src/security_audit.rs` | Findings pré-écrits hardcoded. Contient `transmute` UB pour forger un `&'static mut`. |
| `libneuralos/src/trait_fixes.rs` | Utilise `crossbeam::queue::ArrayQueue` — crossbeam PAS dans Cargo.toml. Ne compile pas. |
| `libneuralos/src/ports.rs` | Orphelin (pas dans mod tree). Utilise `heapless` — pas dans Cargo.toml. |
| `libneuralos/src/api/error.rs` | Orphelin. Référence `crate::core::CoreError` (n'existe pas) et `NeuralError::Bridge` (commenté). Vapor. |
| `libneuralos/src/core/neural_processing/liquid_neural.rs` | 60% type defs, 30% trivial learning (`delta = lr * strength * 0.01`), pattern extraction hardcode 2 patterns. |
| `kernel/src/neural/wait_free_core.rs::WaitFreeSpikeProcessor` | Ring buffer racy (`head.store` sans CAS :237, :291). "Wait-free" menteur (DashMap = lock-based sharded). |
| `kernel/src/neural/lock_free_scheduler.rs::LockFreeNeuralScheduler` | Bug shutdown (AtomicBool différent du thread :179 vs :205). `execute_task` = `sleep(10ms)` (:318). |
| `kernel/src/neural/hardware_accelerated.rs` | DMA/SIMD/NPU = `thread::sleep` (:554, :575, :592). `discover_npu_devices` retourne 2 devices hardcoded. |
| `kernel/src/cipherpunk_interface.rs` | Retourne strings ASCII. `Vec<u8, 1024>` (heapless syntax sur std types — compile pas). |
| `kernel/src/service_layer.rs` | Récursion infinie `fn status(&self) { self.status() }` (:567, :571). |
| `crates/neuralos-core/src/quantum_effect.rs` | **Pas quantique.** Réimplémentation de TypeScript Effect-TS. |
| `crates/neuralos-core/src/dashboard/*` | 2 455 LOC de théâtre. `web_interface::start()` = 4 printlns. |
| `crates/neuralos-core/src/agent.rs` | `process_request` retourne `format!("Analyzed system resources for task: {}", request.task)` avec `confidence: 0.95` hardcoded. |
| `neuralos-microkernel/src/ai_threat_detection.rs` | Syntaxe invalide : `**markdown bold**` au lieu de `// comments`. Compile pas. |
| `neuralos-microkernel/src/enterprise_access_control.rs` | Importe `crate::neural_encryption_bridge` — module inexistant. |
| `neuralos-microkernel/src/dual_kernel_coordination.rs` | 148 LOC de `println!`. Pas de logique de coordination. |
| `neuralos-microkernel/src/bare_metal.rs` | `init_idt/init_pic/init_mmu` empty bodies (:103-125). `run_hardware_loop` = NULL deref (:75-79). |
| `src-tauri/` | Tauri v1/v2 schema mismatch. `kernel_bridge.rs` module manquant. Security module émet `"*_placeholder"` strings. |
| `packages/neuralos-application-layer/` | Byte-identique stale 2025-09-20 snapshot de `src/`. Pas dans workspace. |
| `packages/neuralos-desktop/` | Cargo.toml only, pas de src/ dir. Vapor. |

## Bug list à fixer pendant le port

À documenter au fur et à mesure. Les bugs identifiés par l'audit (au-dessus) sont
pointés vers leur `file:line` dans v0.1; le port v2 doit les corriger, pas les reproduire.
