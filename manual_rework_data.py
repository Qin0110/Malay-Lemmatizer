import argparse
import json

from Preprocess.hybrid_lemmatizer import HybridLemmatizer

parser = argparse.ArgumentParser(
    description='Rewrite SFT assistant targets as lemmatized sentences.'
)
parser.add_argument('--input', default='unsloth/data/train_sft.jsonl')
parser.add_argument('--output', default='unsloth/data/training.jsonl')
# Loading malaya's KenLM + HuggingFace models is slow and unnecessary when the
# targets only need lemmatizing, so normalization stays opt-in.
parser.add_argument('--normalize', action='store_true',
                    help='run the malaya normalizer before lemmatizing')
args = parser.parse_args()

records = []
with open(args.input, 'r', encoding='utf-8') as file:
    for line in file:
        if not line.strip():
            continue
        records.append(json.loads(line))

normalizer = None
if args.normalize:
    from Preprocess.normalizer import Normalizer
    normalizer = Normalizer()

# Routes each token to WordNet (English) or MALINDO (Malay); "renovate" and
# "friends" are lemmatized as English instead of being passed through.
lemmatizer = HybridLemmatizer(normalizer=normalizer)

total = len(records)
failed = 0
changed = 0
print(f'[start] {total} records loaded', flush=True)

with open(args.output, 'w', encoding='utf-8') as out_file:
    for n, data in enumerate(records, 1):
        string = data["messages"][2]["content"]

        try:
            # MALINDO preserves entry casing ("Aku", "NAM"); train_sft.jsonl
            # targets are lowercase, so match that convention.
            final_sentence = lemmatizer.lemmatize(string)
        except Exception as exc:
            # Keep the original text so one bad row cannot abort the whole run.
            failed += 1
            final_sentence = string
            print(f'[warn] record {n} failed ({exc}); kept original', flush=True)

        if final_sentence != string:
            changed += 1

        data["messages"][2]["content"] = final_sentence
        out_file.write(json.dumps(data, ensure_ascii=False) + '\n')
        out_file.flush()

        if n % 50 == 0 or n == total:
            print(f'[progress] {n}/{total}', flush=True)

print(f'[done] wrote {total} records, {changed} changed, {failed} fallback(s)', flush=True)
