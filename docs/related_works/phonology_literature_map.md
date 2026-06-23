# Literature Map: Phonological Operations in Transformer LMs
### All Sources, Annotated

---

## 1. Core Phonology-in-LLMs Papers

---

### Liao & Shi (2026) — "How Tokenization Limits Phonological Knowledge Representation in Language Models"
**arXiv:2604.17105 | ACL 2026 | Also at ICML 2025 (earlier version)**

The most directly relevant predecessor. Studies BERT, GPT-2, GPT-neo, Mistral-7B, Llama3/3.1-8B on three phonological tasks: rhyming awareness, grapheme-to-phoneme (G2P) conversion, and syllable counting. Key findings:

- Phonological knowledge lives in **middle layers** (20–60% of total model depth), not final layers — suggesting it's an intermediate-level representation, not a surface output statistic.
- Introduces **STAD (Syllabification-Tokenization Alignment Distance)**: normalized Hamming distance between BPE tokenization boundary vector and CMU syllable boundary vector for a word. Formula: STAD(v_tok, v_syl; w) = Σ|b_i − c_i| / n. Lower STAD = better alignment = better phonological task performance.
- Words with STAD > 0.25 consistently yield worse model performance on G2P and syllable counting across all models tested.
- Loanwords and cognates (identified via CogNet database) have systematically higher STAD — historical linguistic factors corrupt tokenization.
- Fine-tuning Llama3.1-8B with IPA-augmented data (wrapping random words with IPA transcriptions during training) improves phonological performance with only 1.1% GSM8K drop and 0.9% MMLU drop.
- **Character-level delimiter intervention** (writing "b/o/y" instead of "boy") significantly improves rhyming awareness — shows the bottleneck is at tokenization granularity for local phonological features.

**Citation uses for your paper:**
- Use STAD as per-item metadata in your dataset (the key new metadata field Liao & Shi introduce)
- Their negative result on character delimiters for global phonological tasks (G2P) vs. positive for local (rhyming) motivates your prediction that Op 4 (oronyms, requiring cross-boundary reassembly) won't benefit from delimiter intervention even though Op 2 (clipping, local) will
- Cite as primary behavioral predecessor your benchmark extends
- The probing methodology (linear classifier on hidden states) is their main tool — you go further by doing causal circuit analysis

**Full URL:** https://arxiv.org/abs/2604.17105

---

### Suvarna, Khandelwal & Peng (2024) — "PhonologyBench: Evaluating Phonological Skills of Large Language Models"
**arXiv:2404.02456 | GitHub: github.com/asuvarna31/llm_phonology**

Introduces PhonologyBench, a benchmark of three tasks: G2P conversion, syllable counting, rhyme word generation. Uses ~3000 words from SIGMORPHON 2021 G2P shared task. Key results:

- 17% accuracy gap on rhyme generation vs. human baseline (~90% human, ~73% GPT-4)
- 45% gap on syllable counting
- GPT-4 substantially outperforms GPT-3.5 on all tasks
- Open-source models (LLaMA) significantly underperform GPT-4

**Citation uses:**
- Primary behavioral benchmark predecessor
- Your benchmark extends PhonologyBench in three ways: (a) adds oronyms and opaque hypocorisms as new operation types; (b) uses BLiMP minimal-pair format (log-probability, not generation), enabling per-item circuit analysis; (c) includes circuit-level metadata (patching effects, STAD, BPE token audit)
- Their rhyme generation numbers establish the behavioral baseline your minimal-pair format should replicate

---

### OpenReview 2025 — "Syllable Tokenization Does Not Improve Phonological Representations"

Pretrains language models with standard BPE tokenization vs. syllable-aware tokenization and finds **no significant improvement** in phonological tasks from the syllable tokenizer. This is an important negative result.

**Citation uses:**
- Directly motivates your circuit-level approach: if fixing the tokenizer doesn't fix phonological performance, the bottleneck must be representational/computational, not purely tokenization
- Sets up the argument: "it's not about the tokenizer; it's about what circuits the model develops to operate on whatever tokenized representations it has"

**Full URL:** https://openreview.net/pdf?id=cDADe0cpxS

---

### Phun-Bench (2026) — "Evaluating LLMs on Phonological Understanding in Chinese"

Chinese phonological benchmark covering three dimensions: homophony, rhyme, and phonetic similarity. Evaluates contemporary Chinese-capable LLMs.

**Citation uses:**
- Shows the phonological bottleneck generalizes cross-linguistically
- Cite in introduction to frame the problem as architectural (not English-specific)
- If your paper focuses on English only, use this as a "future work / generalization" pointer

**Full URL:** https://chatpaper.com/paper/296222

---

### PACUTE (OpenReview 2026) — "Phonology-, Affix-, and Character-level Understandings in Transformer Evaluations"

Benchmark specifically designed to test what LLMs miss due to BPE tokenization obscuring character-level and morphological structure.

**Citation uses:**
- Related benchmark in the "tokenization obscures structure" space
- Cite alongside PhonologyBench and Liao & Shi as the cluster of papers your work builds on

**Full URL:** https://openreview.net/forum?id=TXpISo5qmn

---

## 2. Mechanistic Interpretability — Circuit Analysis

---

### García-Carrasco, Maté & Trujillo (2024) — "How does GPT-2 Predict Acronyms? Extracting and Understanding a Circuit via Mechanistic Interpretability"
**arXiv:2405.04156 | AISTATS 2024 | GitHub: github.com/jgcarrasco/acronyms_paper**

Identifies a circuit of **8 attention heads (~5% of total heads)** in GPT-2 Small for predicting three-letter acronyms (e.g., given "Artificial Intelligence: A_I_", predict "I"). Three functional groups: heads that attend to the target letter positions, heads that propagate positional information via the causal mask, and heads that move letter identities to the output position. Notably, this is the first MI paper to study a multi-token prediction task rather than single-token.

**Citation uses:**
- Closest MI predecessor to your work
- Your Op 3 (initialisms: B→Bee, J→Jay) is structurally adjacent — both require letter-to-letter-name lookup. **Core experiment**: do the 8 acronym heads also activate for initialism prediction?
- Their methodology (logit lens + activation patching to identify heads, OV circuit visualization to interpret function) is your methodological template for the circuit analysis section
- Their functional categories (letter-position heads, propagation heads, letter-mover heads) give you vocabulary to describe your own circuit components

---

### Wang et al. (2023) — "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small"
**ICLR 2023 | arXiv:2301.05217**

The canonical large-circuit MI paper. Identifies 28 attention heads in 6 functional roles (duplicate token detection, previous token tracking, induction, S-inhibition, name mover, negative name mover) implementing indirect object identification (IOI: "John gave Mary the book. She gave it to ___" → "John"). Faithfulness of circuit is 87% using mean ablation.

**Citation uses:**
- Methodological template for circuit extraction, activation patching, logit attribution
- Your causal scrubbing and faithfulness/completeness/minimality evaluation follows their protocol
- Their 87% faithfulness figure (mean ablation) with known degradation under other ablation methods is the baseline you compare to (see Miller et al. 2024 for the method-conditionality problem)

---

### Olsson et al. (2022) — "In-Context Learning and Induction Heads"
**Transformer Circuits Thread | arXiv:2209.11895**

Identifies the two-head induction circuit: previous-token heads (Layer 0) copy the token immediately before a repeated sequence; induction heads (Layer 1) use this signal to predict what follows the repeated token. The canonical simple, interpretable circuit.

**Citation uses:**
- Op 4 (oronyms) requires matching a phonological pattern across a word boundary — structurally similar to induction (pattern matching across a sequence). Check whether induction heads (specifically Head 5.1 in GPT-2 Small) activate on oronym items
- Your circuit discovery methodology (head-level attribution, then composition analysis) follows the induction head methodology

---

### Nanda et al. (2023) — "Progress Measures for Grokking via Mechanistic Interpretability"
**ICLR 2023 | arXiv:2301.05217**

Fully reverse-engineers the grokking circuit on modular addition. Discovers Fourier features. The gold standard for circuit completeness ("Triangulated" in MECHVAL).

**Citation uses:**
- Sets the bar your paper explicitly acknowledges it doesn't reach: "our circuit claims are Causally Suggestive to Mechanistically Supported; full Triangulation as in Nanda et al. requires mathematical completeness that phonological circuits are unlikely to achieve"
- The Fourier feature analogy: if the model learns phonological operations cleanly, there may be an analogous compact representation (IPA phoneme directions in residual stream?) that is the "Fourier mode" of phonology

---

### Miller, Chughtai & Saunders (2024) — "Transformer Circuit Faithfulness Metrics Are Not Robust"
**arXiv:2407.08734**

Shows that the IOI circuit's 87% faithfulness figure (Wang et al.) drops below 50% under ablation methods other than mean ablation. Method-conditionality is the central issue.

**Citation uses:**
- Critical caveat for your faithfulness numbers: "we report faithfulness under mean ablation; following Miller et al. (2024), we also report under [resample ablation / zero ablation] and note the range"
- Your MECHVAL E1 criterion (intervention reach) requires you to do this

---

### Verb Conjugation Circuit in GPT-2 (arXiv:2506.22105, 2025)

Isolates subject-verb agreement circuit in GPT-2 Small using activation patching + direct path patching.

**Citation uses:**
- Shows your exact methodology on GPT-2 Small is active and publishable in 2025–2026
- Establishes that non-semantic, structural linguistic tasks have discoverable circuits in GPT-2 Small — motivates that phonological operations should too

---

### Chan et al. (2022) — "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses"
**Alignment Forum 2022**

Formal method for testing whether a proposed circuit explanation is complete: replace all activations outside the proposed circuit with corrupted-run activations and verify that circuit alone achieves specified faithfulness threshold.

**Citation uses:**
- Protocol for E14 in your experiment suite (causal scrubbing on top-k circuit)
- Cite alongside Wang et al. (2023) for circuit validation methodology

---

### A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models (arXiv:2407.02646, 2025)

Comprehensive survey of 337+ MI papers organized by task type.

**Citation uses:**
- Background/related work paragraph: "for a comprehensive survey of MI methods, see [cite]"
- Check their taxonomy to see if phonological operations appear anywhere — if not, explicitly note that phonological/sound-based tasks are absent from current MI research (motivating your contribution)

---

## 3. Benchmarks and Datasets

---

### BLiMP: The Benchmark of Linguistic Minimal Pairs for English (Warstadt et al. 2020)
**TACL 2020 | arXiv:1912.00582**

67 sub-datasets × 1000 minimal pairs each, testing syntax, morphology, and semantics. The gold standard for minimal-pair LM evaluation. Format: {sentence_good, sentence_bad} pairs where the model scores the acceptable sentence higher. Extended to morphology via BLiMP-morph, etc.

**Citation uses:**
- Your benchmark format: BLiMP {good, bad, target_token, foil_token} directly adapts their methodology to phonological operations
- "To our knowledge, BLiMP and its derivatives contain no phonological minimal pairs — all 67 sub-datasets test syntax, morphology, or semantics" (motivating statement for your paper)
- Their finding that "models struggle with subtle semantic and syntactic phenomena such as negative polarity items" is the syntactic analogue to your finding about which phonological operations are hard

---

### MultiBLiMP 1.0 (2026) — Massively Multilingual Benchmark of Linguistic Minimal Pairs
**arXiv, published March 2026 | 101 languages**

Extends BLiMP to 101 languages.

**Citation uses:**
- Shows the BLiMP framework is actively expanding; your PhonoBench-MI fills the missing phonological dimension for English
- Future work: "extending our benchmark to other languages following MultiBLiMP's multilingual protocol"

---

### CxMP: A Linguistic Minimal-Pair Benchmark for Evaluating Construction Grammar (arXiv:2602.21978)

Construction-based minimal pairs — tests whether models understand form-meaning pairings in constructions.

**Citation uses:**
- Related benchmark in the "BLiMP extensions" space; cite alongside MultiBLiMP as evidence that the minimal-pair methodology is actively expanding to new linguistic domains

---

### JOKER Shared Task — Onomastic Wordplay (CLEF 2025/2026)
**CLEF 2025: ceur-ws.org/Vol-4038/ | CLEF 2026: joker-project.com/2025/**

Humor detection, search, and translation shared task with specific Track 3: onomastic wordplay. 2,333 English onomastic wordplays (funny names) with professional French reference translations. Types: portmanteau, pun/homophone, neologism, assonance/alliteration, anagram.

**Citation uses:**
- **Primary data source for Op 4 (oronyms)**: scrape Track 3 items, filter for (a) phonetically transparent, (b) target word is single BPE token, (c) both parts look like plausible English names
- Cite as data source in your dataset construction section
- Their typology of onomastic wordplay types maps imperfectly but usefully onto your operation taxonomy

---

### KOWIT-24: A Richly Annotated Dataset of Wordplay in News Headlines (arXiv:2503.01510)

98 phonetic similarity items, 190 polysemy items, 26 homonymy items, covering cross-word phonetic smearing.

**Citation uses:**
- Secondary data source for Op 4; filter for items where the target word is a single BPE token and the "funny reading" involves cross-word-boundary fusion
- Cite alongside JOKER as a wordplay corpus you drew Op 4 items from

---

### Word Segmentation as a Phonological Probing Task (ACL 2025 CoNLL)
**aclanthology.org/2025.conll-1.34**

Uses word segmentation (identifying word boundaries in a phoneme sequence) as a probing task to study phonological representations learned by phoneme-level models.

**Citation uses:**
- Related probing approach; your work extends probing to circuit-level analysis
- Their word segmentation framing is exactly the reverse of your Op 4 oronym task: they ask "where are the word boundaries?" you ask "can the model dissolve word boundaries to find the target word?"

---

## 4. Theoretical Background

---

### Saxe, McClelland & Ganguli (2013/2019) — "A Mathematical Theory of Semantic Development in Deep Neural Networks"
**PNAS 2019 | arXiv:1810.10531**

Proves that learning dynamics in deep linear networks are governed by the SVD of the input-output correlation matrix. Each singular mode is learned independently with timescale O(1/s_α) where s_α is the singular value.

**Citation uses:**
- Provides theoretical basis for predicting operation difficulty ordering: operations with high singular values (frequent, consistent phonological patterns — Op 3, Op 2) are learned early and have cleaner circuits; operations with low singular values (rare, inconsistent — Op 1b, Op 7) are learned late and have messier circuits
- Predicts: if Op 1b (opaque hypocorisms) is memorized rather than rule-derived, its "circuit" will be a high-layer retrieval mechanism with a flat, high-rank contribution rather than a structured low-rank circuit
- This is theoretical grounding for why different operation types should have qualitatively different circuit structures

---

### Warstadt et al. (2020) — BLiMP (listed above under Benchmarks)

---

### Mechval-v2 (your own paper)

**Citation uses:**
- Self-cite for the validity framework applied in Tier 6 experiments
- Key criteria to apply: C1 (falsifiability), C4 (discriminant validity — can you distinguish the "phonological assembly" circuit from the "letter lookup" circuit?), I1 (necessity via ablation), I4 (double dissociation between Op 3 circuit and Op 1b circuit), E1 (intervention reach — multiple ablation methods), V3 (alternative level — could the circuit be explained at the token frequency level rather than phonological level?)
- Applying your own framework to your own paper is a strong methodological move; cite explicitly and note which verdict tier each circuit claim achieves

---

## 5. Related / Background

---

### "The Self-Hating Attention Head: A Deep Dive in GPT-2" (Alignment Forum, 2025)

GPT-2 Small Head L1H5 computes semantic similarity and suppresses self-attention.

**Citation uses:**
- Background for GPT-2 Small head-level anatomy; your circuit analysis operates in the same model
- Reference point: "phonological heads should be distinguishable from the semantic similarity heads identified by [cite] because..."

---

### Speech Codec Probing from Semantic and Phonetic Perspectives (ISI/USC 2026)

Probes speech tokenizers for phonetic vs. semantic content using layer-wise analysis and CKA. Finds current speech tokenizers primarily capture phonetic rather than lexical-semantic structure.

**Citation uses:**
- Cross-modal bridge: speech tokenizers are the audio analogue of text tokenizers; their finding that speech tokenizers encode phonetics but not semantics is the flip side of your finding that text tokenizers encode semantics but not phonetics
- Cite in discussion: "our results complement [cite], who show speech tokenizers fail in the opposite direction"

---

### Extracting Rule-Based Descriptions of Attention Features in Transformers (arXiv:2510.18148, 2025)

Rewrites attention head computation as weighted sum of promotion/suppression terms; extracts interpretable rule descriptions.

**Citation uses:**
- Related methodology for attention head interpretation; if you want to go beyond OV circuit visualization to natural-language descriptions of your identified heads, cite this as your interpretability tool

---

### LBPE: Long-token-first Tokenization (arXiv:2411.05504)

Alternative tokenization scheme that prioritizes longer tokens.

**Citation uses:**
- Tokenization intervention comparison; cite in context of "alternative tokenizers that might improve phonological performance"

---

## 6. Summary Table

| Paper | Year | Venue | Your use |
|-------|------|-------|----------|
| Liao & Shi | 2026 | ACL | STAD metric; probing baseline; primary predecessor |
| Suvarna et al. (PhonologyBench) | 2024 | — | Behavioral benchmark predecessor |
| García-Carrasco et al. | 2024 | AISTATS | Acronym circuit; Op 3 overlap experiment |
| Wang et al. (IOI) | 2023 | ICLR | Circuit methodology template |
| Olsson et al. (Induction Heads) | 2022 | TCT | Op 4 mechanism hypothesis |
| Nanda et al. (Grokking) | 2023 | ICLR | Triangulation bar / circuit completeness |
| Miller et al. | 2024 | — | Faithfulness metric robustness caveat |
| Chan et al. (Causal scrubbing) | 2022 | AF | E14 circuit validation |
| Warstadt et al. (BLiMP) | 2020 | TACL | Minimal-pair format |
| MultiBLiMP | 2026 | — | BLiMP extension context |
| CxMP | 2026 | — | BLiMP extension context |
| JOKER 2025/2026 | 2025 | CLEF | Op 4 data source |
| KOWIT-24 | 2024 | — | Op 4 secondary data source |
| Syllable tokenization negative | 2025 | OR | Motivates circuit-level approach |
| Phun-Bench | 2026 | — | Cross-linguistic generalization |
| PACUTE | 2026 | OR | Related benchmark |
| Saxe et al. | 2019 | PNAS | Theoretical difficulty ordering |
| Mechval-v2 | 2026 | — | Self-audit validity framework |
| Verb circuit GPT-2 | 2025 | arXiv | Methodology precedent |
| Self-hating head | 2025 | AF | GPT-2 Small anatomy background |
| Speech codec probing | 2026 | ISI | Cross-modal phonetics complement |
| Word segmentation probing | 2025 | CoNLL | Related probing approach |
| MI practical review | 2025 | arXiv | Survey / background |

