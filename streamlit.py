import re
import streamlit as st

from malindo_morph.dictionary import Dictionary
from Preprocess.normalizer import Normalizer
from Preprocess.tokenizer import Tokenizer

@st.cache_resource
def load_dictionary():
    return Dictionary()


# def tokenize_malay(text):
#     return re.findall(
#         r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*|[0-9]+|[^\w\s]",
#         text
#     )


# def simple_lemmatize_token(token, dictionary):
#     cleaned = token.lower().strip(".,!?;:\"'()[]{}“”‘’")

#     if not cleaned:
#         return {
#             "token": token,
#             "root": token,
#             "lemma": token,
#             "method": "punctuation"
#         }

#     results = dictionary.lookup(cleaned)

#     if not results:
#         return {
#             "token": token,
#             "root": cleaned,
#             "lemma": cleaned,
#             "method": "not_found"
#         }

#     row = results[0]

#     lemma = row["lemma"]

#     # If MALINDO gives multiple lemma options, choose first for now
#     if "@" in lemma:
#         lemma = lemma.split("@")[0]

#     return {
#         "token": token,
#         "root": row["root"],
#         "stem": row["stem"],
#         "lemma": lemma,
#         "prefix": row["prefix"],
#         "suffix": row["suffix"],
#         "circumfix": row["circumfix"],
#         "method": "malindo_lookup"
#     }


st.title("Malay Lemmatizer")
st.write("Enter a Malay sentence and the system will return the root and lemma using MALINDO Morph.")

dictionary = load_dictionary()

user_input = st.text_area(
    "Enter Malay text:",
    placeholder="Contoh: Dia sedang menggunting kertas dengan gunting yang tajam."
)

if st.button("Lemmatize"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        normalizer = Normalizer(dictionary)
        tokenizer = Tokenizer()

        user_input = normalizer.normalize(user_input)
        tokens = tokenizer.tokenize(user_input)

        results = []

        for token in tokens:
            result = dictionary.morphological_analysis(token)
            results.append(result)

        st.subheader("Result")

        st.dataframe(results, use_container_width=True)

        lemmatized_sentence = " ".join([item["Lemma"] for item in results])

        st.subheader("Lemmatized Sentence")
        st.write(lemmatized_sentence)