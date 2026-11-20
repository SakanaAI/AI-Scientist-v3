# Writing Style
Write for a reader who will skim the abstract, look at the figures, and read the results once.

## Abstract
1. Report at most two headline numbers. Express the main effect in interpretable terms (relative change, multiple of the measured noise, or position relative to a known reference) rather than a list of raw full-precision values. When a paper strategy is provided, one of them is the quantity named in its `central_thesis`.
2. Keep per-ablation deltas, statistical procedure (z-scores, pooled standard deviations, intervals, degrees of freedom), and internal configuration names out of the abstract. A qualitative sentence such as "most of the gain depends on component X" is enough.
3. Define or avoid subfield jargon so a researcher from a neighboring area can follow it.
4. End on the takeaway or implication of the work. Do not end on a list of null, borderline, or unresolved findings; one limiting clause is enough. A failed final qualification is a finding, not a limiting clause: state it plainly in the abstract.
5. Stay within the abstract word target. Give background and setting in one or two sentences, then spend the remaining words on the method, the main finding, and the takeaway. Closing sentences that place the results in a broader context are optional and count toward the target.

## Results
1. Open each paragraph with its finding stated as a claim, then point to the evidence (table or figure), then give the caveat. Write the claim as an ordinary sentence; do not bold or otherwise typeset it as a pull quote.
2. Tables and figures carry the numbers; prose interprets them. Quote only the numbers the argument needs and never restate every table cell in text.
3. Report each effect once, against one clearly named reference. Put alternative references or robustness variants of the same effect in Supplementary Information.
4. Write results as paragraphs, not bullet lists. Use bullets only for genuine enumerations such as compared configurations or limitations.
5. Keep secondary statistical machinery (attribution games, bounds, worst-case pairings, extra significance variants) in Supplementary Information and summarize its conclusion in one sentence in the main text.
6. State an unverified mechanism in at most one sentence, explicitly labeled as a hypothesis, or omit it. Do not narrate work that was not done ("we have not verified", "we did not profile", "was not inspected") in Results; state each such gap once, in Limitations.
7. Summarize unreplicated exploratory runs in one or two sentences in the main text; give their details in Supplementary Information.
8. Report measured outcomes. Do not compare results against numeric pre-run expectations or specification estimates; these are internal planning artifacts, not findings. For an ablation with a recorded hypothesis, state in one clause whether the measured outcome supports it.
9. Keep the search process out of results prose and captions: search-time scores of selected configurations, counts of runs explored or unscored, and which configuration each exploratory run started from belong in the experimental protocol and Supplementary Information. Two exceptions apply:
   - When no qualifying winner was found, search coverage from `study.*` (scored and unscored counts, search scores) is the evidence; report it in Results, kept separate from final results.
   - A figure whose subject is the search itself (run progression, sample efficiency) may name search-time scores in its caption so the reader can read the axes.

## Consistency
1. Match numeric precision to the measured uncertainty. In prose, round differences to about one digit beyond the noise scale (for example 0.009 rather than 0.0095 when the run-to-run spread is about 0.001); full precision belongs in tables.
2. Keep claim strength identical across abstract, introduction, results, and conclusion. Use words like "strong" or "robust" only where the evidence supports them everywhere they appear.
3. Use one name per concept, method, and baseline throughout, defined once in the Method section.
4. Do not repeat the same headline numbers in every section. The conclusion synthesizes and states implications; it does not recap the numbers.

## Prose
1. Write connected sentences with transitions that carry the argument. Avoid long runs of short single-fact sentences and verbless fragments.
2. Captions describe what is shown and how to read it; the main text states the conclusions.
