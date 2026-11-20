from langid.evaluate import snippets
from langid.profiles import LanguageIdentifier, build_profile, ngram_counts, out_of_place, tokenize


def test_tokenize_keeps_letters_marks_and_inner_apostrophes():
    assert tokenize("Don't stop 42 times!") == ["don't", "stop", "times"]
    assert tokenize("हिन्दी भाषा") == ["हिन्दी", "भाषा"]      # vowel signs stay inside words


def test_padded_ngrams_match_the_text_example():
    grams = set(ngram_counts(["text"], ns=(2,)))
    assert grams == {"_t", "te", "ex", "xt", "t_"}


def test_profile_is_ranked_by_frequency():
    assert build_profile("aaa ab", ns=(1,), top_n=2) == ["a", "b"]


def test_out_of_place_distance():
    assert out_of_place(["a", "b"], ["a", "b"]) == 0
    assert out_of_place(["b", "a"], ["a", "b"]) == 2
    assert out_of_place(["z"], ["a", "b"]) == 2        # missing n-gram costs the profile length


def test_identifies_toy_languages_and_round_trips(tmp_path):
    model = LanguageIdentifier.train({
        "en": "the cat and the dog are in the house with the other animals " * 20,
        "de": "die katze und der hund sind im haus mit den anderen tieren " * 20,
    })
    assert model.identify("the dog is in the house") == "en"
    assert model.identify("der hund ist im haus") == "de"
    path = tmp_path / "profiles.json"
    model.save(path)
    assert LanguageIdentifier.load(path).identify("die katze") == "de"


def test_snippets():
    assert snippets("a b c d e", 2) == ["a b", "c d"]
