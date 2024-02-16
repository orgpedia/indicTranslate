from indictranslate import Translator
from pathlib import Path

import tempfile

GlossaryStr = """
translations:
  - 'eng_Latn': 'The fox jumped over the lazy dog.'
    'hin_Deva': 'लोमड़ी आलसी बिल्ली के ऊपर से कूद पड़ी।'

  - 'eng_Latn': 'The fox jumped over the lazy horse.'
    'hin_Deva': 'लोमड़ी आलसी गधे के ऊपर से कूद पड़ी।'
"""


def test_sentence():
    translator = Translator()
    s = translator.translate_sentences(["My name is Anthony Gonsavles."])
    assert s == ["मेरा नाम एंथनी गोंसावल्स है।"]


def test_number():
    translator = Translator(enable_numeric=True)
    s = translator.translate_sentences(["24-345"])
    assert s == ["२४-३४५"]


def test_glossary():
    temp_file = Path(tempfile.mkstemp(suffix=".yml")[1])
    temp_file.write_text(GlossaryStr.lstrip())
    translator = Translator(glossary_path=temp_file)

    s = translator.translate_sentences(["The fox jumped over the lazy dog."])
    assert s == ["लोमड़ी आलसी बिल्ली के ऊपर से कूद पड़ी।"]


def test_sentences():
    sentences = [
        "The fox jumped over the lazy dog.",
        "The lion jumped over the lazy tiger.",
        "The panther jumped over the lazy rabbit.",
        " -234-567 ",
        "The fox jumped over the lazy horse.",
        "The sparrow jumped over the crow",
    ]

    temp_file = Path(tempfile.mkstemp(suffix=".yml")[1])
    temp_file.write_text(GlossaryStr.lstrip())

    translator = Translator(glossary_path=temp_file, enable_numeric=True)

    s = translator.translate_sentences(sentences)
    ans = [
        "लोमड़ी आलसी बिल्ली के ऊपर से कूद पड़ी।",
        "शेर ने आलसी बाघ के ऊपर से छलांग लगा दी।",
        "तेंदुआ आलसी खरगोश के ऊपर से कूद गया।",
        " -२३४-५६७ ",
        "लोमड़ी आलसी गधे के ऊपर से कूद पड़ी।",
        "गौरैया कौवे के ऊपर से कूद गई",
    ]
    assert s == ans
