# Qualitative error analysis — every model, SWELLS and real languages

Generated with `code/analysis/error_analysis.py` (full dump with more examples: `results/scratch/error_analysis_raw_dump.txt`).
Error types are assigned automatically from gold vs prediction word sets: *tense/inflection* (same stems, different endings/auxiliaries),
*person/pronoun*, *number*, *word order* (same words, other order), *lexical (1–2 words)*, *lexical/garbled* (≥ half the words wrong).
"Reasoning tail" = the last ~450 characters of the model's chain of thought before the answer.

**Provenance note.** Per-item predictions for the 8B SFT are from the thinking-on W+G re-evaluation `results/raw/sft8b_think_wg`
(42.4 / 27.4 exact, chrF 78.0 / 78.2). The headline 8B SFT row in the tables (50.3 / 33.9, chrF 80.7 / 78.5) comes from the
2 Oct evaluation whose per-item file was not kept locally (summary only). Same model; the partition profile is the same
(1–2 ≈ 95, 6–7 high, Latin ≈ 0), so the qualitative findings hold, but the per-partition numbers quoted below for SFT are from the 42.4 run.

---

## Part A — SWELLS, artificial → English, dictionary + grammar (W+G), 1,400 items, 14 partitions

Partitions: 1 word lookup · 2m/2f noun gender · 3 verb tense · 4 compound past · 5m/5f adjective agreement + tense ·
6 relative clauses · 7m/7f relative clause + compound past + agreement · 8a/8g/9a/9g Latin-style case languages (held out from training).

### Summary table (exact accuracy per partition)

| model | 1 | 2m | 2f | 3 | 4 | 5m | 5f | 6 | 7m | 7f | 8a | 8g | 9a | 9g | avg | fails | no tag | median chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4B base (untrained) | 37 | 16 | 32 | 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 6.4 | 1310 | 11 | 60 |
| 4B curriculum SFT | 77 | 91 | 82 | 33 | 28 | 20 | 22 | 72 | 68 | 68 | 8 | 3 | 3 | 2 | 41.2 | 823 | 1 | 14,391 |
| 4B GRPO-from-base, 3 ep, no-CoT | 100 | 97 | 91 | 30 | 15 | 16 | 15 | 76 | 51 | 62 | 9 | 6 | 0 | 12 | 41.4 | 820 | 0 | 55 |
| 8B base (untrained) | 56 | 64 | 63 | 13 | 7 | 13 | 7 | 3 | 4 | 2 | 12 | 9 | 7 | 9 | 19.2 | 1131 | 307 | 1,014 |
| 8B CoT-GRPO course-7 ckpt, think off | 99 | 85 | 64 | 58 | 27 | 27 | 24 | 10 | 7 | 3 | 46 | 8 | 18 | 5 | 34.4 | 919 | 596 | 10,485 |
| 8B SFT, think on (42.4 run) | 96 | 96 | 89 | 45 | 35 | 53 | 35 | 53 | 33 | 43 | 9 | 3 | 2 | 2 | 42.4 | 806 | 10 | 4,003 |
| 8B CoT-GRPO epoch 1, think on | 97 | 94 | 92 | 65 | 36 | 38 | 32 | 15 | 18 | 17 | 45 | 14 | 30 | 15 | 43.4 | 792 | 48 | 11,110 |
| **8B CoT-GRPO final (2 ep), think on** | 96 | 99 | 98 | 71 | 61 | 58 | 42 | 35 | 31 | 34 | 47 | 11 | 40 | 13 | **52.6** | 664 | 6 | 3,071 |

Reference: Marmonier et al. fine-tune 61.4 (Latin ≈ 0), their base 32.4; o4-mini 75.0; Gemini 3.1 Pro 96.7.

### Error-type profile (counts over all failures; groups 1–5 / 6–7 / Latin)

| model | lexical/garbled | tense/infl. | tense/aux | 1–2 words | person | order | dominant failure in 1–5 | in 6–7 | in Latin |
|---|---|---|---|---|---|---|---|---|---|
| 4B base | 1073 | 133 | 18 | 57 | 10 | 7 | garbled | garbled | garbled |
| 4B SFT | 519 | 168 | 26 | 42 | 28 | 17 | garbled 178 / tense 108 | garbled 46, 1–2 words 19, person 15 | garbled 295 |
| 4B GRPO no-CoT | 394 | 177 | 38 | 68 | 76 | 49 | garbled 148 / tense 87 / order 43 | **person 45**, 1–2 words 38 | garbled 228, tense 87 |
| 8B base | 812 | 207 | 58 | 33 | 5 | 4 | garbled | garbled | garbled |
| 8B GRPO c7 think-off | 705 | 112 | 62 | 19 | 8 | 10 | garbled 204 | garbled 235 (truncation) | garbled 266 |
| 8B SFT | 419 | 185 | 66 | 82 | 32 | 12 | **tense 108** / garbled 75 | garbled 60, 1–2 words 39 | garbled 284 |
| 8B GRPO ep1 | 409 | 219 | 56 | 76 | 13 | 13 | garbled 109 / tense 75 | garbled 170 | garbled 130 / **tense 129** |
| 8B GRPO final | 321 | 164 | 32 | 110 | 19 | 13 | garbled 69 / tense 60 | garbled 97 / **1–2 words 82** | garbled 155 / tense 100 |

### A1. Qwen3-4B base (untrained) — 6.4

- **Positive**: only trivial items. `gxèvzrèvèqàtr àfx` → "He was watching." (gold "he was watching"). Partition 1–2 hits are dictionary copies.
- **Negative**: 1,310 failures, 82 % garbled. Typical: copies the source back verbatim (`l'pevu peqbàvupàcu skuké reskudké` → same string), or produces a loosely related sentence from the glossed words ("the mice that we chose" → "Choose the mouse."; "the generous admirals that I chose" → "He chose the generous admiral."). Median output 60 chars: it writes almost no reasoning at all despite the CoT instruction.
- **Interpretation**: the untrained 4B neither reasons nor uses the grammar; it treats the task as dictionary lookup + guess. Format is fine (11 tag misses), content is missing — the baseline's problem is not the prompt.

### A2. Qwen3-4B curriculum SFT — 41.2

- **Positive**: textbook reproductions of the training CoT: `fdè gdèpungdadègnlw …` → "I would watch the young nannies" with the reasoning "…the sentence structure in Sagdèomç typically places adjectives after their nouns unless they belong to the specific category…". The 7f example ("the small mice that we observed") correctly applies the complex agreement rule ("we therefore expected to find the morpheme ajà- added to this verb form").
- **Negative**: tense is the main 1–5 failure (108): "I chose the bruises" → "I had chosen the bruises"; "she would choose the small toys" → "she chose the small toys"; the reasoning *states* the right rule and then writes the wrong English tense. In 6–7 the errors are swapped participants ("that we watched" → "that he watched"). Latin is a disaster (295 garbled): `gcgoîûn âmoêmd moèpûnàbâê` → "the sailor will hear" (drops "the nations"); "the tall masters will pity the small statues" → "the small statues pity the tall masters" — it applies the trained word-order template to a case language.
- **Interpretation**: SFT learned the *procedure* of a SWELLS solution (sentence templates, rule names) and executes it mechanically. Where the test rule set equals the training rule set (6–7) this works; where it doesn't (Latin) the template produces confident nonsense. Output is very long (median 14 k chars) — the model recites the whole template every time.

### A3. Qwen3-4B GRPO-from-base, 3 final epochs, no-CoT — 41.4

- **Positive**: direct answers, 55 chars, zero tag misses: "I watched the abandoned houses", "the good nannies that we watched", Latin "the son sees the sailor". Partition 1 is 100 %.
- **Negative**: the profile is different from SFT — **person/pronoun** (45 in 6–7) and **word order** (43 in 1–5) dominate: "the outdated festivals that she chose" → "…that he chose"; "I watched the unfriendly jackals" → "I watched the jackals unfriendly"; "I was admiring the kisses" → "we admired the kisses"; Latin: "the oxen will see the boys" → "the oxes will see the boys", "the famous tribes will pity…" → "…pity…" (future dropped).
- **Interpretation**: without CoT the model has learned surface statistics of the gold outputs (which pronoun/order is most frequent) rather than reading the grammar; it gets exactly the items whose answer is the majority pattern. chrF reward made it very clean (zero junk) but cannot distinguish he/she or future/present (each ≈ 90+ chrF), so these are precisely the residual errors. Note it does *not* collapse on Latin as badly as SFT (9/6/0/12 vs 8/3/3/2) but is still ≈ base.

### A4. Qwen3-8B instruct, untrained — 19.2 (think off, 4k)

- **Positive**: "the famous man fears the good tribe" (9a), "the son sees the sailor" — when it bothers to reason about case, it works.
- **Negative**: 307/1,400 never close the tag; 596 in the course-7 checkpoint. Degenerate loops ("…by the nannies by the nannies by the nannies…", or the source repeated 40 times). Semantic inversions: "she observed the kisses" → "The kiss is observed by her"; "he observed the red magnets" → "The red magnet and they observe."
- **Interpretation**: the instruct model has the capability (Latin 12/9/7/9 is already above 4B SFT) but no discipline: it rambles, loops, and invents passive constructions. This is the starting point for the 8B GRPO runs; what RL has to teach first is "stop and answer".

### A5. Qwen3-8B CoT-GRPO, course-7 checkpoint, think off — 34.4

- **Positive**: same strengths as the base, cleaner: "I was observing the new wheels", "the gray crosses that she watched", Latin "the master hears the poets".
- **Negative**: 596 failures are *truncations* — the 4k budget ends mid-reasoning ("I think I need to make an educated guess here. Given that rç'd is the auxiliary…" is the extracted "answer"). Of the completed ones, tense/aux ("she observed the kisses" → "She has observed the kisses") and order errors.
- **Interpretation**: an intermediate checkpoint: it already reasons more systematically than the base but has not yet learned to finish inside the budget; with thinking on and 16k it scores 37.8. Not a result to report on its own, only as a trajectory point.

### A6. Qwen3-8B curriculum SFT, thinking on — 50.3 official (42.4 in the per-item run analysed here)

- **Positive**: clean, short, template-like CoT: "we will observe", "I will bury the old mice", "the gray crosses that she watched"; Latin hits are the few sentences that happen to follow trained word order ("the nations know the horses").
- **Negative**: 1–5 failures are mostly **tense** (108 of ~240): "she would admire the kits" → "she admired the kits"; "I chose the generous thugs" → "I will choose the generous thug". In 6–7 it drops the relative structure ("the generous thugs that we chose" → "We chose the generous thugs") or swaps gender ("that she chose" → "that he chose"). Latin: 384/400 wrong — "the ghosts will see the magistrates" → "the ghosts of the magistrates will see"; "the tall ox will pity the harsh masters" → "I pity the harsh oxen of the master". The reasoning on Latin items literally quotes rules of the *training* languages ("the subject pronoun always follows the verb…").
- **Interpretation**: same failure mode as 4B SFT at higher accuracy: it memorised how a solution reads and applies it even where it does not fit. It is confident and wrong; it never says "this language has cases". On the shared rule set it is excellent (6–7 ≈ 63–78 in the official run), which is exactly where GRPO is still weak.

### A7. Qwen3-8B CoT-GRPO, final stage epoch 1, thinking on — 43.4

- **Positive**: Latin already at 45/14/30/15 (above everything local): "the oxen will see the boys", "the harsh horse praises the bad poet" — the reasoning checks the accusative ending and the future morpheme explicitly.
- **Negative**: median output 11 k chars, 48 no-tag; many 6–7 answers are a half-finished monologue ("Starting with the first word: qxyqxyxébe. The dictionary says it's a 2nd group verb…"). Relative clauses are collapsed ("the magnets that we admired" → "The magnets admired the magnet"); Latin failures are 50 % **tense** ("the tall horses pity…" → "The tall horse will pity…").
- **Interpretation**: after one epoch the model has learned to *use* the grammar but not to *finish*: long, exploratory reasoning that often runs out of budget on the stacked-rule items. The tense-heavy Latin errors are the chrF blind spot (future vs present ≈ 90 chrF).

### A8. Qwen3-8B CoT-GRPO, final (2 epochs), thinking on — 52.6 ← headline

- **Positive**: rules 3–5 now 71/61/58: "I would choose", "I chose the small houses"; Latin "the pig hears the son", "the harsh father sees the tall son". Reasoning is grammar-driven and self-correcting ("Wait, the word order is OVS…", "the accusative plural for second declension is -xt, so ézànñèxt is the object"). Median output fell to 3.1 k chars, only 6 no-tag — epoch 2 taught it to stop.
- **Negative**: 664 failures. In 6–7, 82 are "1–2 words off": relative clause flattened ("the terminals that I buried" → "I buried the terminals"), head noun replaced by the source token ("gqcé watched the beautiful stained glass windows"), or tense dropped ("the lice that I chose" → "…that I choose"). In 1–5: participant swaps ("I chose the kisses" → "the kisses chose him"; "I would admire the new tires" → "He admired the new tires"). Latin: 100 of 263 failures are **tense/number** only ("the ghost will send the tribes" → "…the tribe"; "the tall pirates forget" → "forgot"); the other 155 are case/voice confusions ("the ghosts forget the poet" → "The ghost is forgotten by the poets" — the reasoning notices the ending does not match the ablative and still chooses passive).
- **Interpretation**: GRPO learned the dictionary + the morphology rules and to apply them to unseen rule systems; it did not learn to resolve *stacked* constraints (relative pronoun + compound past + agreement) and it is systematically careless about single-morpheme features (tense, number, gender) — the features chrF barely penalises. Versus SFT item-by-item (on the 42.4 run): GRPO fixes 103 of SFT's 384 Latin failures, SFT fixes 8 of GRPO's; in 6–7 the two models fail on different items (oracle ≈ 70).

### Cross-model conclusions (SWELLS)

1. **Format is never the bottleneck for trained models** (≤ 10 tag misses out of 1,400 for every SFT/final-GRPO model). For the untrained 8B and intermediate GRPO checkpoints it is: truncation/loops explain 300–600 failures.
2. **SFT = procedure memorisation.** Both SFT models fail Latin in the same way (template applied to the wrong language) and both make tense errors while *stating* the correct rule.
3. **No-CoT GRPO = output statistics.** It is clean and good on majority patterns but blind to person/number/tense — chrF's blind spots.
4. **CoT GRPO = rule use that transfers,** weakest on stacked rules; tense/number is its residual error class in every partition group. A reward term that is sensitive to single-morpheme errors (word-exact or exact match mixed with chrF) targets exactly this and is the natural next ablation.
5. The two trained 8B models are complementary (different failure items), which is itself evidence they learned different things.

---

## Part B — Real languages (Hus / Back-to-School, 18 languages, chrF, book+dict context)

Trained = Mizo, Ilokano, Kimbundu (5,982 training sentences); unseen = the other 15. ek = English→X, ke = X→English.
"Too long" = hypothesis > 3× reference length; "copied" = hypothesis equals the source.

| model | trained ek | trained ke | unseen ek | unseen ke | unseen ek too long / copied (of 1,500) | unseen ke too long / copied |
|---|---|---|---|---|---|---|
| Qwen3-4B base, Hus prompt | 12.7 | 29.0 | 13.4 | 25.1 | 345 / 43 | 197 / 64 |
| Qwen3-4B base, SWELLS-format prompt | 12.7 | 27.9 | 15.3 | 27.3 | 281 / 322 | 108 / 149 |
| SWELLS-trained 4B SFT, SWELLS prompt | 15.4 | 26.8 | 13.6 | 24.6 | 9 / 0 | 17 / 0 |
| Qwen3-4B real-language SFT | 35.4 | 34.2 | 15.1 | 30.2 | 412 / 3 | 14 / 1 |
| Qwen3-4B GRPO-from-base (sub-2k) | 30.6 | 31.2 | 22.2 | 30.4 | 119 / 1 | 59 / 5 |
| Qwen3-4B GRPO-from-base (full 5,982) | 32.3 | 37.4 | 23.3 | 34.0 | 39 / 0 | 31 / 11 |

Note (8 Oct): the GRPO-from-base rows above use the first-line extractor; 573/10,002 of its outputs start with a bare `<translation>` line and were under-scored. Tag-aware rescoring (`HUS_RESCORED_TAGFIX.tsv`, snapshot §7.2d) gives corpus chrF 31.7 / 36.1 trained, 23.5 / 34.8 unseen (book+dict); the qualitative conclusions below are unchanged.
| Qwen3-4B GRPO-from-SFT | 35.8 | 38.2 | 16.4 | 33.0 | 673 / 0 | 137 / 5 |
| Llama-3.2-3B real-language SFT | 35.6 | 32.9 | 14.2 | 27.4 | 328 / 1 | 21 / 1 |
| Gemini 3.1 Pro (reference) | 45.7 | 53.9 | 40.4 | 53.8 | 2 / 0 | 2 / 5 |

(Numbers here are per-sentence means over book+dict rows and differ by ≤ 1 chrF from the language-averaged tables in the snapshot.)

### B1. Qwen3-4B base, Hus prompt — 13 / 25–29

- **Negative**: continuation-mode junk: "1. 2. 3. 4. 5. …" for a Minangkabau sentence; counting words from the dictionary ("10 9 five, 10 seven skunk caribou nine" for Gitksan; "6 fish died, 7 friends saw 4 fish" for Kimbundu). 345/1,500 unseen en→X outputs are > 3× too long.
- **Positive**: X→en on languages with transparent loanwords/numbers: Mizo "The Hershey-Chase experiment showed that DNA is the genetic material…" (chrF 54) — proper nouns and English technical words carry the score.
- **Interpretation**: the base model does not know where to stop and often does not even attempt translation; its X→en score is mostly named-entity copying. This is the "discipline" gap that any training closes.

### B2. Qwen3-4B base, SWELLS-format prompt (CoT + tags) — 13 / 27, unseen 15 / 27

- **Negative**: degenerate repetition inside the tag ("Түркестән үзәк үзәк үзәк…", "Chin chya kachin chya kachin…"); 322/1,500 unseen en→X outputs are the English source copied back; X→en often returns meta-text ("The translation of the given Chokwe text segment into English is as follows:").
- **Positive**: Kabuverdianu/Gitksan en→X where the target is close to Portuguese/English (chrF 33–41, partly by leaving English in place).
- **Interpretation**: the explicit CoT prompt gives the base model more room to drift — more copying and repetition, tag closed only 68 %/91 %. Confirms that the baseline's weakness is not a missing format.

### B3. SWELLS-trained 4B SFT on real languages (SWELLS prompt) — 15 / 27, unseen 14 / 25

- **Negative**: fluent-looking but unrelated English: Dinka "Atoms are so incredibly tiny…" → "ten, one, seven, and five eat"; Chuvash → "What is the character encoding of the always-evermore kaki persimmon wheat turkani?"; Mizo → "travel widely regardless of death, the change tonight in large numbers". en→X outputs are short pseudo-words ("sadek, 'l-ŋâi sai ta, 0 or 1, du sadek sadek rai u ga").
- **Positive**: perfectly formatted (0 copies, ≤ 17 too long); Dogri X→en "The famous Canal District (Dutch: Grachtenstad) Amsterdam…" chrF 62 when the sentence is mostly named entities.
- **Interpretation**: artificial-language SFT transferred the *interface* (think, then one tagged answer) and the habit of gluing dictionary glosses into an English sentence, but not real-language competence: it builds grammatical English from the wrong glosses. Transfer artificial → real is weak, as the aggregate numbers say.

### B4. Qwen3-4B real-language SFT — trained 35 / 34, unseen 15 / 30

- **Negative (unseen en→X)**: 412/1,500 are degenerate repetitions ("O Amazona tam tam kota kota kota…", "kwa kudibhinga kwa kudibhinga…"); it also writes in the *wrong trained language* (Waray target produced in Ilokano: "No kasapulan a komersial daytoy, masapul nga adda daytoy…").
- **Negative (X→en)**: fluent hallucination unrelated to the source: Ilokano "Sharks do exist…" → "Birds do not get lonely, but they do get bored sitting in a tree…"; Gitksan "And they jailed them." → "It was a very peaceful night."
- **Positive**: trained languages are genuinely good: Kimbundu "In 2002 Goma was destroyed by the eruption of the Nyiragongo volcano…" (55), Ilokano "The trick to getting by is the concept of the niche…" (44).
- **Interpretation**: SFT learned the three training languages and English fluency, and when faced with an unseen language it either emits a trained language or confabulates a plausible English sentence. It does *not* consult the dictionary for new languages — the same "memorised procedure" behaviour as on SWELLS, now as language memorisation.

### B5. Qwen3-4B GRPO-from-base (sub-2k and full) — trained 31–32 / 31–37, unseen 22–23 / 30–34

- **Negative**: unseen en→X still contains source-language leakage and invented morphology ("Dhonguan thiaar, people yiaar thong thong messages thiaar thiaar computer…"), and the X→en failures are generic filler ("There are many types of systems that need to be maintained and checked every day"). Outputs are longer than the reference (median 155–180 vs 110 chars) — it over-explains.
- **Positive**: on unseen languages it actually uses the dictionary: Gitksan "Ii na yuxwhl train ii yee'y g̲oohl Stockholm." → "…on the train, and you are going to Stockholm…" (41); Kabuverdianu "In Japan, the first few flower festivals of summer were organized for the emperor…" (59); en→X Dinka/Kachin produce target-script, dictionary-sourced words (34, 30). Full-data run reduces the too-long count 119 → 39.
- **Interpretation**: GRPO never saw a target it had to copy, so it did not memorise languages; it learned "look words up, write one sentence, stop". That is why it is the only arm that improves unseen en→X (13 → 23) while SFT stays at 15. Its ceiling is lexical: grammar is not used (book ≈ dict), so word order and function words are English-like.

### B6. Qwen3-4B GRPO-from-SFT — trained 36 / 38, unseen 16 / 33

- **Negative**: inherits SFT's unseen en→X pathology and amplifies it: 673/1,500 unseen en→X outputs are > 3× too long (median 334 chars vs 110), e.g. Waray target produced in Ilokano, Dinka target full of Kimbundu-looking morphology ("…usona dya videolink tunde mu kibhite kya USAF base mu kididi kyang kyang kya mu ithangana…"). X→en still hallucinates ("Sharks do exist…" → "Young children, in general, are not very afraid of strangers…").
- **Positive**: best trained-language scores of the 4B family (35.8 / 38.2) and good unseen X→en (33.0): Wolof "Examples of such activities include things like hiking, various water activities, photography, wildlife…" (53).
- **Interpretation**: RL on top of SFT optimises what the SFT already does (trained languages, English fluency) and cannot undo the language memorisation; on unseen targets chrF rewards long outputs that happen to share n-grams, so it rambles. GRPO-from-SFT is the wrong order for unseen-language transfer.

### B7. Llama-3.2-3B real-language SFT — trained 36 / 33, unseen 14 / 27

- **Negative**: same as Qwen SFT: Dogri en→X at chrF 0–1 (wrong script/degenerate), 328 too-long unseen en→X; X→en hallucinations ("Why do transportation systems engender such complaints…" → "How many people do you think would be required to perform the job?"; "Okay well." → "He said, 'I am worried about the potential impact this may have on the economy.'").
- **Positive**: trained languages on par with Qwen SFT (Ilokano pyramid sentence 52); Minangkabau/Malay-like targets are decent (54–61) because of Indonesian pre-training knowledge.
- **Interpretation**: the SFT failure mode is model-family independent — memorise trained languages, confabulate elsewhere. Supports reporting SFT-doesn't-transfer as a general finding.

### B8. Gemini 3.1 Pro (reference) — trained 46 / 54, unseen 40 / 54

- **Negative**: the hard cases are genuinely hard: Natugu (chrF 13–15 both directions, a language with almost no web presence), Guarani/Wolof paraphrases that are correct in meaning but lexically far from the reference.
- **Positive**: Kabuverdianu en→X 65, Kalamang en→X 56 ("Owa fiber eir mambon, kon tama Dian taraun=kin? Kerkapun."), Dogri X→en 71 — it actually produces target morphology.
- **Interpretation**: the frontier model uses the resources (and prior knowledge) for both directions; our 4B models are 20–25 chrF below on unseen languages, and the gap is mostly en→X.

### B9. Does Gemini read the grammar book? (resource ablation, 18 languages, `b2s-eval/results_gemini31.tsv`)

Gemini answers directly (no reasoning text), so grammar use can only be read off the book+dict / dict / none ablation (same 10,002 items, 0 API errors).

| chrF | book+dict | dict | none |
|---|---|---|---|
| X→en, 18-lang mean | 53.9 | 53.4 | 52.8 |
| en→X, 18-lang mean | 41.9 | 40.9 | 40.6 |

- **16 of 18 languages: book ≈ dict ≈ none** (book−dict between −1.9 and +1.4; removing all resources costs ≤ 3). Gemini already knows Mizo, Ilokano, Kimbundu, Wolof, Latgalian, Kabuverdianu, … from pretraining; the resources are irrelevant to it there.
- **The two languages it does not know are where the book is used**: Kalamang none 17 → dict 39 → book 43 (X→en; en→X 18 → 35 → 40; book +4 both directions) and Natugu X→en 35 → 31 → 45, en→X 13.5 → 12.8 → 22 (book +13.5 / +9.2 over dict).
- **Positive (book needed)**: Kalamang "Sor wa me mais" → ref "This fish is spoiled": book exact (100), dict "This Sor is spoiled" (58), none "The canoe will come tomorrow" (8). "Tumun se bo teimun miarip" → ref "The kid has become this big": book "The child has become that big" (58), dict "Tumun cuscus until east that many" (10), none hallucination (9). Natugu "X rpibo bade kx trtingr" → ref "And I told him no": book "And I told him the truth" (67), dict "The black dog swims to meet them" (11), none "He went to the village" (9); the long Natugu breadfruit-branch sentence goes 20 (dict) → 87 (book).
- **Negative (book hurts)**: Kalamang "Kier tamanggara botkin?" → ref "Where are you two going?": dict "Where are you going?" (70) but book "Where did Kier go?" (20) — it misparses the dual pronoun *kier* as a name after reading the book; "I'm just sitting" → dict "ap nakmeleluo" (52), book "an doka" (7). Natugu "Nzvz-kzng badr doa lr Gwalekana…" → none 70, dict 46, book 32: the book drags a sentence it already translated well into an over-literal reparse. So the book is a double-edged resource even for a frontier model: it helps when the model has nothing else, and adds noise when it does.
- **Interpretation**: grammar use is real but conditional — a frontier model consults the book only for languages absent from pretraining, where it is worth +4–13 chrF; for known languages prior knowledge dominates and the book is noise. SWELLS, where nobody knows the languages, is the "Kalamang case" for every item — which is why the grammar is decisive there for every model (W-only collapses SFT 50 → 9, Gemini 96 → 6.5 eng→art) and ≈ 0 on Hus for our 4B models, which cannot exploit a long real book even when they have nothing else. This is consistent with Hus's own finding that books help only for extremely low-resource languages.

### Cross-model conclusions (real languages)

1. **Untrained base = stopping problem**; any training fixes it (too-long counts drop by 3–10×), which is the "discipline" part of the gain.
2. **SFT (Qwen and Llama) = language memorisation**: strong on the three trained languages, emits a trained language or a fluent hallucination on unseen ones. This mirrors the SWELLS SFT behaviour (procedure memorisation, Latin collapse).
3. **GRPO-from-base = dictionary use that transfers**: the only arm improving unseen en→X; still lexical, no grammar use, English-like syntax.
4. **GRPO-from-SFT inherits SFT's biases** and adds length pathology on unseen targets; order matters.
5. **Grammar book is unused by every 4B model** (book ≈ dict in all cells); what they learn to use is the dictionary. Gemini uses the book, but only for the two languages it does not already know (Kalamang +4, Natugu +9–13 chrF; §B9) — grammar use is conditional on having no prior knowledge, which is exactly the SWELLS situation.
