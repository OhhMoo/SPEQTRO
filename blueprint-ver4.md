# SPEQTRO Blueprint v4 — Open-Source Multimodal Model Replacement

**Objective:** Replace the Claude API dependency in SPEQTRO with a locally-hosted,
fine-tuned open-source model that retains full tool-use, chemical reasoning, and
multimodal (spectral image) capabilities.

---

## 1. Why Replace Claude

| Concern | Current State | Target State |
|---|---|---|
| API cost | Per-token billing, scales with usage | Zero inference cost after deployment |
| Data privacy | Spectra/structures leave user environment | Fully local, no external calls |
| Offline use | Requires internet + API key | Works air-gapped in lab environments |
| Customisation | Prompt engineering only | Full weight-level specialisation |
| Latency | Round-trip network + queue | Local GPU, ~200–500 ms/response |

---

## 2. Capability Requirements

The replacement model must be able to do everything the current Claude integration does:

### 2.1 Core capabilities

| Capability | Detail |
|---|---|
| **Structured tool calling** | Emit valid JSON tool calls matching SPEQTRO's MCP schema; parse tool results back into reasoning |
| **Multi-turn agentic loop** | Maintain context across multiple tool calls in one session (verify pipeline: up to 5 sequential calls) |
| **Chemical language** | Parse and generate SMILES, InChI, IUPAC names, peak tables, molecular formulas |
| **Spectroscopic reasoning** | Interpret ¹H/¹³C shift tables, IR band assignments, MS fragmentation trees |
| **Multimodal input** | Read IR spectrum images, NMR printouts, TLC photos, structure drawings |
| **Structured output** | Return verdict JSON (`CONFIRMED / LIKELY / UNCERTAIN / UNLIKELY`) with breakdown fields |
| **Uncertainty expression** | Distinguish "this peak is consistent" from "this peak is diagnostic" |

### 2.2 Minimum model spec

| Dimension | Minimum | Preferred |
|---|---|---|
| Parameters | 7B | 14–32B |
| Context window | 16k tokens | 32k+ tokens |
| Vision encoder | Yes (for spectral images) | Yes |
| Tool / function calling | Native or fine-tuned | Native |
| License | Apache 2.0 or MIT | Apache 2.0 |

---

## 3. Base Model Candidates

Three models are evaluated as starting points. All are open-weight, multimodal,
and support structured outputs with fine-tuning.

### 3.1 Qwen2.5-VL-7B / 72B *(primary recommendation)*

- **Why:** Best-in-class vision encoder for document/plot understanding; native
  tool-call schema; strong chemistry performance on ChemBench; Apache 2.0 license.
- **Vision:** Supports variable-resolution images via NaViT-style dynamic tiling —
  handles narrow spectral plots and wide NMR printouts equally well.
- **Tool use:** Ships with a function-calling template compatible with SPEQTRO's
  existing MCP JSON schema with minimal adapter work.
- **Tradeoffs:** 72B requires multi-GPU inference (2× A100 or 4× A40);
  7B fits on a single 24 GB GPU with 4-bit quantisation.

### 3.2 Llama-3.2-Vision-11B / 90B *(strong alternative)*

- **Why:** Meta's open instruction-tuned model with built-in vision cross-attention;
  well-supported ecosystem (llama.cpp, vLLM, Ollama).
- **Vision:** Cross-attention vision adapter; handles spectra well but slightly
  weaker than Qwen2.5-VL on dense plot reading.
- **Tool use:** Requires tool-use fine-tuning; no native function-calling template
  in the base model — this is a significant fine-tuning requirement.
- **Tradeoffs:** Larger community, but tool-call alignment needs more SFT data.

### 3.3 InternVL2-8B / 26B *(lightweight option)*

- **Why:** Smallest footprint with competitive multimodal performance; InternLM2
  backbone has strong chemistry benchmark scores.
- **Vision:** Dynamic high-resolution strategy; handles both chemical structure
  images and spectrum plots.
- **Tool use:** Requires fine-tuning; weaker baseline on agentic tasks.
- **Tradeoffs:** Less community support; licensing is more restrictive at larger sizes.

### Recommendation

**Start with Qwen2.5-VL-7B.** Develop and validate the full pipeline at 7B scale,
then scale to 72B if evaluation shows significant accuracy gaps.

---

## 4. Dataset Strategy

Fine-tuning requires three distinct dataset types, assembled in layers.

### 4.1 Layer 1 — Chemistry domain knowledge (continual pretraining, optional)

Purpose: Ensure the model understands chemical notation and spectroscopy before
any task-specific training begins.

| Source | Content | Size estimate |
|---|---|---|
| PubChem compound corpus | SMILES, IUPAC names, CID descriptions | ~100M records, subsample 5M |
| ChemDataExtractor (Cambridge) | Chemical entities + properties from papers | ~2M sentences |
| NMRDB / SDBS text | Spectrum descriptions, compound assignments | ~300k records |
| Reaxys / SciFinder abstracts *(if licensed)* | Reaction conditions, yield, NMR data in text | negotiated access |
| PubMed chemistry subset | Spectroscopic characterisation paragraphs | ~500k abstracts |

> **Note:** Layer 1 is optional if starting from Qwen2.5-VL-7B-Instruct — the
> base model already has substantial chemistry knowledge from pretraining.
> Only needed if starting from a general-purpose or coding base.

### 4.2 Layer 2 — Spectroscopy instruction tuning (SFT)

Purpose: Teach the model to reason correctly about spectroscopic evidence.

#### 4.2.1 NMR interpretation

| Dataset | Format | Source |
|---|---|---|
| NMRshiftDB2 spectrum + structure pairs | `{smiles, nucleus, shifts[]} → assignment` | nmrshiftdb.org (CC-BY-SA) |
| BMRB biomolecular NMR | `{sequence, shifts} → structural context` | bmrb.io (public) |
| Synthetic SFT: predict-then-verify | `{smiles} → predicted shifts → match reasoning` | Generated from CASCADE + NMRshiftDB2 predictions |
| Synthetic SFT: peak table QA | `"What functional group causes the 170 ppm 13C peak?"` | Template-generated from functional_group_shifts.json |

Target: **50,000 – 150,000 examples** after deduplication and quality filtering.

#### 4.2.2 IR interpretation

| Dataset | Format | Source |
|---|---|---|
| SDBS IR + structure pairs | `{molblock, wavenumber_peaks} → functional group assignments` | sdbs.db.aist.go.jp (public) |
| NIST WebBook IR spectra | `{smiles, ir_image} → band table` | webbook.nist.gov |
| SSIN training data (if accessible) | `{spectrum_vector} → functional_group_label` | Internal / from SSIN paper |
| Synthetic SFT: SMARTS-to-IR | `{smiles} → expected IR bands + rationale` | Generated from ir.py predictions |

Target: **20,000 – 60,000 examples**.

#### 4.2.3 Mass spectrometry

| Dataset | Format | Source |
|---|---|---|
| GNPS spectral library | `{smiles, precursor_mz, fragments[]} → compound assignment` | gnps.ucsd.gov (CC0) |
| MassBank Europe | `{smiles, ms2_spectrum} → structure + neutral losses` | massbank.eu (CC-BY) |
| Synthetic SFT: fragmentation reasoning | `{smiles, adduct} → predicted neutral losses + rationale` | Generated from ms.py / ICEBERG |

Target: **40,000 – 100,000 examples**.

#### 4.2.4 Multimodal spectroscopy (vision)

| Dataset | Format | Source |
|---|---|---|
| SDBS spectral images + assignments | `{ir_image, nmr_image} → peak_table + smiles` | SDBS scrape (for research) |
| Synthetic: render spectrum → caption | Render predicted spectrum as matplotlib image → description | Generated pipeline (see §5.1) |
| NMR printout OCR + reasoning | `{nmr_printout_image} → peak list + interpretation` | Manual annotation + augmentation |

Target: **15,000 – 40,000 image-text pairs**.

### 4.3 Layer 3 — Tool-use agentic trajectories (most critical)

Purpose: Teach the model to use SPEQTRO's tool registry correctly in multi-turn
agentic loops, not just answer spectroscopy questions in isolation.

#### 4.3.1 Structure of a training trajectory

Each trajectory is a full conversation thread:

```
[system]   <SPEQTRO system prompt>
[user]     "Verify this product: SMILES=CCO, 1H peaks: 3.69, 2.61, 1.18"
[assistant] <tool_call: pipeline.verify_product(smiles="CCO", ...)>
[tool]     <tool_result: {verdict: "LIKELY", confidence: 72.3, ...}>
[assistant] "The structure is LIKELY correct (72.3% confidence). The 3.69 ppm
             peak matches the CH2 group predicted by CASCADE 1.0..."
```

#### 4.3.2 Trajectory sources

| Source | Volume | Method |
|---|---|---|
| **Claude-generated silver traces** | 10,000–50,000 | Run current SPEQTRO agent on a curated molecule set; log full trajectories; filter for quality |
| **Hand-authored gold traces** | 500–2,000 | Chemist-annotated, highest quality; used for DPO preference pairs |
| **Synthetic correct/incorrect pairs** | 20,000 | Template tool-call generation with correct vs. wrong tool selection for DPO |
| **Error recovery traces** | 5,000 | Trajectories where a tool fails and the model tries a fallback correctly |

#### 4.3.3 Silver trace generation pipeline

```
molecule_set (PubChem subset, ~10k compounds)
    │
    ▼
Generate observed peaks using CASCADE + NMRshiftDB2
    │
    ▼
Run current SPEQTRO + Claude agent → log full trajectory JSON
    │
    ▼
Quality filter:
  - verdict field present and non-null
  - tool calls use correct parameter schema
  - final response is chemically coherent
  - no hallucinated SMILES or impossible shifts
    │
    ▼
Cleaned trajectory dataset (~60–70% pass rate expected)
```

### 4.4 Dataset sizing summary

| Layer | Examples | Format |
|---|---|---|
| Chemistry SFT (NMR + IR + MS) | ~200,000 | Instruction–response pairs |
| Multimodal SFT | ~30,000 | Image + instruction → response |
| Agentic tool-use trajectories | ~50,000 | Multi-turn conversations with tool calls |
| DPO preference pairs | ~10,000 | (chosen, rejected) pairs |
| **Total** | **~290,000** | Mixed |

---

## 5. Synthetic Data Generation Pipelines

Several dataset components must be generated programmatically from SPEQTRO's
own tools. This is a key advantage — the existing tool suite acts as a labelling oracle.

### 5.1 Spectrum rendering pipeline (for vision data)

```python
# Pseudocode — actual implementation as a standalone script

for smiles in molecule_set:
    # 1H
    h1_result   = predict_h1_cascade(smiles)        # CASCADE 1.0
    h1_shifts   = [p["shift_ppm"] for p in h1_result["predictions"]]
    h1_img      = render_nmr_spectrum(h1_shifts, nucleus="1H")   # matplotlib lorentzian
    h1_caption  = describe_spectrum(h1_shifts, smiles, nucleus="1H")  # template

    # 13C
    c13_result  = predict_c13_cascade(smiles)       # CASCADE 2.0
    c13_shifts  = [p["shift_ppm"] for p in c13_result["predictions"][smiles]]
    c13_img     = render_nmr_spectrum(c13_shifts, nucleus="13C")
    c13_caption = describe_spectrum(c13_shifts, smiles, nucleus="13C")

    # IR
    ir_result   = predict_absorptions(smiles)
    ir_img      = render_ir_spectrum(ir_result["bands"])
    ir_caption  = describe_ir_bands(ir_result["bands"])

    save(smiles, h1_img, c13_img, ir_img, h1_caption, c13_caption, ir_caption)
```

Augmentation: vary noise level, baseline distortion, label density, and
colour scheme to prevent the model from overfitting to a single rendering style.

### 5.2 Peak-matching QA generation

For each compound in the training set, generate 3–5 QA variants:

- **Direct match:** `"Does the 170.5 ppm peak support a carbonyl group in {smiles}?"` → answer with reasoning
- **Error detection:** Insert a wrong peak; ask if the spectrum is consistent
- **Multiplicity reasoning:** `"The triplet at 3.69 ppm integrates for 2H. What does this suggest?"`
- **Solvent disambiguation:** Ask the model to identify a solvent residual peak

### 5.3 Agentic trajectory augmentation

After collecting silver traces, augment by:

- **Paraphrase user queries** (5 rephrasings per trace) with a T5-paraphrase model
- **Swap equivalent molecules** (stereoisomers, isotopologues)
- **Inject tool failures** and label the correct recovery action

---

## 6. Training Protocol

### 6.1 Phase 1 — Supervised Fine-Tuning (SFT)

**Goal:** Teach domain knowledge, spectroscopic reasoning, and correct tool-call format.

| Hyperparameter | Value |
|---|---|
| Base model | Qwen2.5-VL-7B-Instruct |
| Training framework | LLaMA-Factory or Axolotl |
| Quantisation during training | QLoRA (4-bit NF4, compute in bfloat16) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate | 2e-4 (cosine schedule, 3% warmup) |
| Batch size | 128 (gradient accumulation × 16) |
| Epochs | 3 (early stopping on validation loss) |
| Max sequence length | 8,192 tokens |
| Gradient clipping | 1.0 |
| Hardware | 4× A100 80 GB (or 8× A40 48 GB) |
| Estimated wall time | 24–48 hours |

**Training order (curriculum):**
1. Chemistry SFT only (NMR, IR, MS Q&A) — 1 epoch
2. Add multimodal SFT — 1 epoch
3. Add agentic tool-use trajectories — 1 epoch

Mixing all at once risks the model forgetting spectrum-image reading while learning
tool-call syntax. Curriculum order avoids this.

### 6.2 Phase 2 — Direct Preference Optimisation (DPO)

**Goal:** Align tool-selection decisions and response quality using chemist-labelled
preference pairs.

| Hyperparameter | Value |
|---|---|
| Method | DPO (Rafailov et al. 2023) |
| β (KL penalty) | 0.1 |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Epochs | 1–2 |
| Reference model | SFT checkpoint from Phase 1 |
| Dataset | 10,000 (chosen, rejected) pairs |

**Preference pair categories:**

| Category | Chosen | Rejected |
|---|---|---|
| Tool selection | Calls `pipeline.verify_product` for a verification task | Calls `nmr.predict_c13_cascade` (partial, not pipeline) |
| Confidence reasoning | Cites specific MAD values and predictor names | Vague "the shifts look consistent" |
| Error handling | Recovers gracefully when CASCADE is unavailable | Crashes or returns empty response |
| Hallucination avoidance | "NMRshiftDB2 is unavailable (Java not found)" | Invents plausible-sounding shift values |
| Purity reporting | Reports contaminant integrals explicitly | Ignores non-product peaks |

### 6.3 Phase 3 — Quantisation and Serving Optimisation

After DPO, prepare the model for production inference:

| Step | Tool | Output |
|---|---|---|
| Merge LoRA → base | `peft.merge_and_unload()` | Full-weight model |
| GPTQ 4-bit quantisation | AutoGPTQ | 4-bit GPTQ checkpoint (~3.5 GB for 7B) |
| AWQ quantisation (alternative) | llm-awq | 4-bit AWQ checkpoint (~3.5 GB for 7B) |
| GGUF conversion | llama.cpp `convert.py` | GGUF for local deployment via Ollama |
| vLLM packaging | vLLM + OpenAI-compatible server | HTTP inference endpoint |

Target: **~3.5 GB VRAM** at 4-bit for 7B model; runs on a single RTX 4090 or 3090.

---

## 7. Integration Architecture

### 7.1 Replacing the Claude client in SPEQTRO

The Claude Agent SDK call in `agent/runner.py` must be swapped for a local
inference client. The tool-call protocol stays identical because the fine-tuned
model is trained to emit the same MCP JSON schema.

```
Current:                         Target:

AgentRunner                      AgentRunner
    │                                │
    ▼                                ▼
anthropic.Anthropic()            openai.OpenAI(
    │                              base_url="http://localhost:8000/v1",
    ▼                              api_key="local"
Claude claude-sonnet-4-6         )
    │                                │
    ▼                                ▼
tool_use blocks                  tool_calls (OpenAI format)
    │                                │
    ▼                                ▼
MCP tool dispatch                MCP tool dispatch (unchanged)
```

**Key change:** vLLM and Ollama both expose an OpenAI-compatible `/v1/chat/completions`
endpoint. Swapping the Anthropic client for an OpenAI client pointed at localhost
requires changes in `runner.py` and `config.py` only — the tool registry, MCP
server, and all tool modules are untouched.

### 7.2 Config changes

New config keys in `~/.speqtro/config.json`:

```json
{
  "model.provider": "local",
  "model.base_url": "http://localhost:8000/v1",
  "model.name": "speqtro-qwen2.5-vl-7b",
  "model.api_key": "local",
  "model.max_tokens": 4096,
  "model.temperature": 0.1
}
```

When `model.provider` is `"anthropic"` (default), behaviour is unchanged.
When `"local"`, `runner.py` uses the OpenAI client instead.

### 7.3 Inference server options

| Option | Pros | Cons |
|---|---|---|
| **vLLM** | Fastest throughput, OpenAI API compatible, tool call support, multi-GPU | Requires Linux/CUDA; large install |
| **Ollama** | Simplest setup, cross-platform, auto GGUF download | No native vision streaming; tool calls via template |
| **LM Studio** | GUI, Windows-native, easy model management | No headless/server mode for CI |
| **llama.cpp server** | Minimal footprint, GGUF, CPU fallback | Slower, manual model management |

**Recommended for development:** Ollama (local testing, Windows-compatible)
**Recommended for production/lab:** vLLM on Linux server

---

## 8. Evaluation Framework

### 8.1 Benchmark suite

| Benchmark | What it tests | Target metric |
|---|---|---|
| **VerifyBench** | SPEQTRO verify pipeline — correct verdict on 200 known compounds with curated spectra | Accuracy ≥ 85% vs. Claude |
| **ShiftMAE** | ¹H and ¹³C shift reasoning — given a spectrum table and structure, identify inconsistencies | F1 ≥ 0.80 |
| **ToolCallAcc** | Tool call schema correctness — valid JSON, correct tool name, no hallucinated params | ≥ 98% valid calls |
| **MultimodalIR** | IR image → functional group identification | F1 ≥ 0.75 on SDBS test set |
| **HallucinationRate** | Fraction of responses containing fabricated shift values or SMILES | < 3% |
| **ChemBench subset** | General chemistry knowledge (40 chemistry reasoning tasks) | ≥ 70% |

### 8.2 Regression tests

Existing `speqtro verify` unit tests must all pass unchanged — the model is
a drop-in replacement; tool outputs are deterministic (the tools themselves
don't change), and the model's verdict should match within ±1 verdict tier.

### 8.3 A/B comparison protocol

For each evaluation molecule:
1. Run Claude claude-sonnet-4-6 → record `{verdict, confidence, tool_calls, response}`
2. Run fine-tuned local model → record same
3. Compute agreement rate on `verdict`, Pearson r on `confidence_percent`, and
   tool-call schema validity

Accept the local model when:
- Verdict agreement ≥ 85% with Claude baseline
- Tool-call validity ≥ 98%
- Hallucination rate < 3%

---

## 9. Roadmap

### Phase 0 — Infrastructure (weeks 1–2)

- [ ] Set up GPU training environment (4× A100 or cloud equivalent)
- [ ] Install LLaMA-Factory + Axolotl + vLLM
- [ ] Pull Qwen2.5-VL-7B-Instruct weights from HuggingFace
- [ ] Confirm Qwen2.5-VL tool-call template works with SPEQTRO MCP schema (dry run)
- [ ] Build spectrum rendering pipeline (§5.1) as a standalone script
- [ ] Write VerifyBench evaluation harness

### Phase 1 — Dataset assembly (weeks 3–6)

- [ ] Download and clean NMRshiftDB2 full dataset (structure + spectrum pairs)
- [ ] Download GNPS + MassBank MS2 libraries
- [ ] Download SDBS IR data (structure + spectrum)
- [ ] Generate 50,000 silver agentic trajectories using current SPEQTRO + Claude
- [ ] Render 30,000 synthetic spectrum images (¹H, ¹³C, IR) with captions
- [ ] Generate 200,000 chemistry SFT instruction–response pairs
- [ ] Annotate 500 gold trajectories + 2,000 DPO preference pairs (human chemists)
- [ ] Run deduplication, length filtering, and quality scoring on all datasets

### Phase 2 — SFT training (weeks 7–9)

- [ ] Run chemistry SFT curriculum (epoch 1: text-only, epoch 2: multimodal, epoch 3: agentic)
- [ ] Evaluate on VerifyBench, ShiftMAE, ToolCallAcc checkpoints after each epoch
- [ ] Hyperparameter sweep on LoRA rank (32 / 64 / 128) using 10% dataset subset
- [ ] Save best SFT checkpoint

### Phase 3 — DPO alignment (weeks 10–11)

- [ ] Expand DPO dataset to 10,000 pairs (augment gold with synthetic)
- [ ] Run DPO on SFT checkpoint
- [ ] Evaluate hallucination rate and tool-call validity
- [ ] Save best DPO checkpoint

### Phase 4 — Quantisation and integration (weeks 12–13)

- [ ] Merge LoRA → full weights
- [ ] GPTQ 4-bit quantisation + perplexity check vs. bfloat16 baseline
- [ ] Package as GGUF for Ollama distribution
- [ ] Implement `model.provider = "local"` switch in `runner.py` and `config.py`
- [ ] Add `speqtro config set model.provider local` command
- [ ] Update `speqtro doctor` to check local inference server health

### Phase 5 — Evaluation and iteration (weeks 14–16)

- [ ] Full VerifyBench run: local model vs. Claude baseline
- [ ] Run all existing SPEQTRO unit/integration tests
- [ ] Identify failure modes → add targeted SFT examples → re-run DPO if needed
- [ ] Stress test: 500-molecule batch, measure throughput (tokens/sec) and stability

### Phase 6 — Scale-up (weeks 17–20, conditional on Phase 5 results)

- [ ] Evaluate Qwen2.5-VL-72B (4-bit AWQ) if 7B accuracy falls below threshold
- [ ] Fine-tune 72B with same dataset using full-precision LoRA on 8× A100
- [ ] Benchmark 72B vs. 7B vs. Claude on VerifyBench
- [ ] Decision: ship 7B, 72B, or maintain Claude as default with local as opt-in

### Phase 7 — Release (week 21+)

- [ ] Add `[local-model]` optional extra to `pyproject.toml` (installs vLLM / Ollama client)
- [ ] Publish fine-tuned weights to HuggingFace (`speqtro-qwen2.5-vl-7b`)
- [ ] Add `speqtro fetch-weights --model speqtro-llm` command
- [ ] Write migration guide for users switching from Claude API
- [ ] Update README: "Run fully locally — no API key required"

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 7B model insufficient for complex elucidation tasks | Medium | High | Phase 6 scale-up to 72B; keep Claude as fallback provider |
| Tool-call hallucination (wrong param names, invented tools) | Medium | High | Strict DPO pairs targeting schema violations; ToolCallAcc gate in CI |
| Vision encoder struggles with low-resolution spectral images | Low | Medium | Augment training images with noise/downscale; test SDBS at 72 dpi |
| Dataset contamination (train compounds appear in benchmark) | Low | High | Bloom filter deduplication between train and VerifyBench |
| GPTQ quantisation degrades chemistry reasoning | Low | Medium | Perplexity check + ShiftMAE regression; fall back to 8-bit if needed |
| vLLM incompatibility with Qwen2.5-VL tool-call format | Low | Medium | Pin vLLM version; maintain Ollama as backup serving path |
| Training cost overrun | Medium | Medium | Phase 0 dry run on 10% dataset; cloud spot instances for A100 hours |

---

## 11. Hardware Requirements

### Minimum (development, 7B model)

| Component | Spec |
|---|---|
| GPU | 1× RTX 4090 (24 GB) or 1× A10G (24 GB) |
| RAM | 64 GB |
| Storage | 500 GB NVMe (models + datasets) |
| Inference | Ollama or llama.cpp on same machine |

### Recommended (training, 7B model)

| Component | Spec |
|---|---|
| GPU | 4× A100 80 GB or 8× A40 48 GB |
| RAM | 256 GB |
| Storage | 2 TB NVMe RAID |
| Interconnect | NVLink or InfiniBand for multi-GPU |

### Production inference (7B, 4-bit)

| Component | Spec |
|---|---|
| GPU | 1× RTX 4090 or 1× A10G |
| VRAM required | ~4 GB (4-bit GPTQ) |
| Throughput | ~80–150 tokens/sec (7B, single A10G) |
| Latency (first token) | ~150–300 ms |

---

## 12. Key Dependencies

| Library | Purpose | Version |
|---|---|---|
| `transformers` | Model loading, tokeniser | ≥ 4.45 |
| `peft` | LoRA fine-tuning, merge | ≥ 0.13 |
| `trl` | DPO trainer | ≥ 0.11 |
| `axolotl` or `llama-factory` | Training orchestration | latest |
| `vllm` | Production inference server | ≥ 0.6 |
| `auto-gptq` | GPTQ quantisation | ≥ 0.7 |
| `llm-awq` | AWQ quantisation | ≥ 0.2 |
| `openai` | Client for local vLLM endpoint | ≥ 1.0 |
| `datasets` | HuggingFace dataset handling | ≥ 2.20 |
| `wandb` | Training metrics and run tracking | ≥ 0.17 |

---

## 13. Success Criteria

The local model is considered production-ready when all of the following hold:

1. **VerifyBench verdict agreement** ≥ 85% vs. Claude claude-sonnet-4-6
2. **Tool-call schema validity** ≥ 98% on 1,000-call stress test
3. **Hallucination rate** < 3% (no fabricated shifts, SMILES, or tool names)
4. **Existing unit tests** pass 100% (tools are unchanged; agent outputs are equivalent)
5. **Inference latency** ≤ 500 ms first-token on target hardware
6. **Model footprint** ≤ 6 GB on disk (4-bit quantised)

---

*Blueprint version 4 — authored against SPEQTRO CLI codebase state March 2026.*
*Supersedes blueprint-ver3. To be updated after Phase 2 evaluation results.*
