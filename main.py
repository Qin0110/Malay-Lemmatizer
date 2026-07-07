from malindo_morph.dictionary import Dictionary

malindo = Dictionary()

words = [
    "menggunting",
    "gunting",
    "membaca",
    "dibaca",
    "kebertanggungjawabannya",
]

for word in words:
    results = malindo.lookup(word)

    print("=" * 50)
    print("WORD:", word)

    if not results:
        print("Not found in MALINDO")
        continue

    for r in results[:5]:
        print({
            "id": r["id"],
            "root": r["root"],
            "surface": r["surface"],
            "prefix": r["prefix"],
            "suffix": r["suffix"],
            "circumfix": r["circumfix"],
            "reduplication": r["reduplication"],
            "source": r["source"],
            "stem": r["stem"],
            "lemma": r["lemma"],
        })