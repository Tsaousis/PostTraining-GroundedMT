# Hus prompt format 2x2 (18 languages, chrF)

Question: does rendering the Hus prompts in the SWELLS format (CoT instruction + `<translation>` tags) make the untrained model follow the task better?

Answer: no. Base Qwen3-4B drops from 13.2/25.8 (Hus prompt) to 7.9/19.9 (SWELLS prompt, en->X / X->en, book+dict); it closes the tag only 68% / 91% of the time and rambles up to the 4096-token budget. The SWELLS-trained SFT (never saw a real language) recovers the format (98%/96% tags) and lands at 14.8/26.3, roughly the untrained model's level on the plain Hus prompt, below the real-language SFT (18.1/31.9). "trained*" for the SWELLS SFT means the MIK languages, which that model never trained on. Raw outputs: raw_swellsfmt/{qwen3-4b-base,swells-sft4b} (6,668 rows each), scoring: ../../code/real_language/aggregate_swellsfmt.py. Grammar book: book+dict <= dict for the SWELLS-format rows too.
