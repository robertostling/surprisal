import surprisal
from pathlib import Path
import sys

model_name = sys.argv[1]

with open(Path.home() / "hf-read-gated.key") as f:
    hf_token = f.read().strip()


g = surprisal.AutoHuggingFaceModel.from_pretrained(
        model_name,
        model_class='causal',
        token=hf_token)
# b = surprisal.AutoHuggingFaceModel.from_pretrained(model_id="bert-base-uncased")


stims = [
    "Titta, en enkel liten mening!",
]

surps = [*g.surprise(stims), *g.surprise(stims, use_bos_token=False)]

for surp in surps:
    print(surp)

*_, surp = surps
print(f"tokens: {surp}")

for wslc in [0, 1, slice(0, 1)]:
    print(f"span of interest (word index): {wslc}")
    print(f"recovered surprisal: {surp[wslc, 'word']}")
    print("=" * 32)

