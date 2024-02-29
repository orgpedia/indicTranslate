import json
import sys
from more_itertools import pairwise
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split
import yaml

import pysbd
import resource

DefaultModelEn2Indic = "ai4bharat/indictrans2-en-indic-dist-200M"
DefaultModelIndic2En = "ai4bharat/indictrans2-indic-en-dist-200M"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
TODO - WHAT IS NEEDED ? NOT HOW?
3. Add way to auto detect the languge and corresponding set src and target languages.
4. Add a way to save intermediate translations, needed for large scale translations.
5. Batching: 1) saving intermediate results and manage memory 2) maximize the GPU loading.

Notes:
1. You can use unicode characters to detect the script but how about language ?
2. Input files are three types 1) .txt-> only sentences 2) .json -> only sentences 3) .json->sents+paras

"""

FloresIsoCodes = {
    "asm_Beng": "asm",
    "ben_Beng": "ben",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "eng_Latn": "eng",
    "gom_Deva": "kok",
    "guj_Gujr": "guj",
    "hin_Deva": "hin",
    "kan_Knda": "kan",
    "kas_Arab": "kas",
    "kas_Deva": "kas",
    "mai_Deva": "mai",
    "mal_Mlym": "mal",
    "mar_Deva": "mar",
    "mni_Beng": "mni",
    "mni_Mtei": "mni",
    "npi_Deva": "nep",
    "ory_Orya": "ori",
    "pan_Guru": "pan",
    "san_Deva": "san",
    "sat_Olck": "sat",
    "snd_Arab": "snd",
    "snd_Deva": "snd",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "urd_Arab": "urd",
}


NumbersDict = {
    "asm_Beng": "০১২৩৪৫৬৭৮৯",
    "ben_Beng": "০১২৩৪৫৬৭৮৯",
    "brx_Deva": "०१२३४५६७८९",
    "doi_Deva": "०१२३४५६७८९",
    "eng_Latn": "0123456789",
    "gom_Deva": "०१२३४५६७८९",
    "guj_Gujr": "૦૧૨૩૪૫૬૭૮૯",
    "hin_Deva": "०१२३४५६७८९",
    "kan_Knda": "೦೧೨೩೪೫೬೭೮೯",
    "kas_Arab": "۰۱۲۳۴۵۶۷۸۹",
    "mai_Deva": "०१२३४५६७८९",
    "mal_Mlym": "൦൧൨൩൪൫൬൭൮൯",
    "mni_Mtei": "꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹",
    "mar_Deva": "०१२३४५६७८९",
    "npi_Deva": "०१२३४५६७८९",
    "ory_Orya": "୦୧୨୩୪୫୬୭୮୯",
    "pan_Guru": "੦੧੨੩੪੫੬੭੮੯",
    "san_Deva": "०१२३४५६७८९",
    "sat_Olck": "᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙",
    "snd_Deva": "०१२३४५६७८९",
    "tam_Taml": "௦௧௨௩௪௫௬௭௮௯",
    "tel_Telu": "౦౧౨౩౪౫౬౭౮౯",
    "urd_Arab": "۰۱۲۳۴۵۶۷۸۹",
}


def memory_usage():
    mem_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return f"{mem_MB:,.0f} MB"


class Translator:
    def __init__(
        self,
        model_name_or_path="default",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
        glossary_path=None,
        enable_numeric=False,
        numeric_passthrough="()/-. \t\n",
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if self.src_lang != "eng_Latn" and self.tgt_lang != "eng_Latn":
            raise ValueError("One of the languages (src/tgt) has to be English")

        if self.src_lang not in FloresIsoCodes:
            raise ValueError("Unknown language {self.src_lang}")

        if self.tgt_lang not in FloresIsoCodes:
            raise ValueError("Unknown language {self.tgt_lang}")

        self.processor = IndicProcessor(inference=True)
        if glossary_path:
            self.glossary = self.load_glossary(glossary_path)
        else:
            self.glossary = None
        if model_name_or_path == "default":
            if self.src_lang == "eng_Latn":
                self.tokenizer = IndicTransTokenizer(direction="en-indic")
                model_name_or_path = DefaultModelEn2Indic
            else:
                self.tokenizer = IndicTransTokenizer(direction="indic-en")
                model_name_or_path = DefaultModelIndic2En

        self.max_batch_words = 1024 if self.src_lang == 'eng_Latn' else 256

        print(f"src_lang: {self.src_lang} tgt_lang: {self.tgt_lang} {model_name_or_path}")

        # will initialize it JIT
        self.model = None
        self.model_name_or_path = model_name_or_path

        # self.model = AutoModelForSeq2SeqLM.from_pretrained(
        #     model_name_or_path, trust_remote_code=True
        # )

        if enable_numeric:
            src_str = numeric_passthrough + NumbersDict[src_lang]
            tgt_str = numeric_passthrough + NumbersDict[tgt_lang]

            self.num_chars_set = set(c for c in src_str)
            self.trans_dict = str.maketrans(src_str, tgt_str)
        else:
            self.num_chars_set = None
            self.trans_dict = None

    def load_model(self):
        if not self.model:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        return self.model

    def load_glossary(self, glossary_path):
        translations = yaml.load(glossary_path.read_text(), Loader=yaml.FullLoader)["translations"]
        sl, tl = self.src_lang, self.tgt_lang
        return {trans[sl]: trans[tl] for trans in translations if sl in trans and tl in trans}

    def is_number(self, text):
        return all(c in self.num_chars_set for c in text) if self.num_chars_set else False

    def translate_number(self, text):
        assert self.trans_dict
        return text.translate(self.trans_dict)

    def _translate_cpu_sentences(self, sentences):
        if not sentences:
            return []

        sentences = self.processor.preprocess_batch(
            sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang
        )
        # Tokenize the batch and generate input encodings
        inputs = self.tokenizer(
            sentences,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        num_tokens = len(inputs["input_ids"]) * len(inputs["input_ids"][0])
        print(f"Before generate: {memory_usage()} num_tokens: {num_tokens}")

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, num_beams=3, num_return_sequences=1, max_length=256
            )

        print(f"After generate: {memory_usage()}")

        outputs = self.tokenizer.batch_decode(outputs, src=False)
        outputs = self.processor.postprocess_batch(outputs, lang=self.tgt_lang)
        return outputs

    def _translate_sentences(self, sentences):
        def sentence_len(pair_sent):
            return len(pair_sent[1])

        if not sentences:
            return []

        # bring short sentences first, this reduces the tensor sizes
        sorted_pair_sentences = sorted(enumerate(sentences), key=sentence_len)

        # delayed loading of the model, only if it is needed
        self.load_model()

        batch_idxs, batch_sentences, batch_words, batch_idx = [], [], 0, 0
        output_sentences = [""] * len(sentences)
        for idx, sentence in sorted_pair_sentences:
            batch_words += len(sentence.split())
            batch_idxs.append(idx)
            batch_sentences.append(sentence)

            # Batch the number of sentences based on the words, rough proxy for tokens
            # no point keeping ready tensors in memory
            if batch_words > self.max_batch_words:
                print(f"\t[{batch_idx}]  #words: {batch_words} #sentences: {len(batch_sentences)}")
                output_batch = self._translate_cpu_sentences(batch_sentences)
                for idx, output_batch_sentence in zip(batch_idxs, output_batch):
                    output_sentences[idx] = output_batch_sentence
                batch_idxs.clear()
                batch_sentences.clear()
                batch_words = 0
                batch_idx += 1

        if batch_sentences:
            print(f"\t[{batch_idx}]  #words: {batch_words} #sentences: {len(batch_sentences)}")
            output_batch = self._translate_cpu_sentences(batch_sentences)
            for idx, output_batch_sentence in zip(batch_idxs, output_batch):
                output_sentences[idx] = output_batch_sentence

        return output_sentences

    def translate_sentences(self, sentences):
        def short_translate_sentence(sentence):
            result = self.glossary.get(sentence, None) if self.glossary else None
            if result is None and self.num_chars_set and self.is_number(sentence):
                result = self.translate_number(sentence)
            return result

        if not self.num_chars_set and not self.glossary:
            return self._translate_sentences(sentences)

        output_sentences = [short_translate_sentence(s) for s in sentences]

        assert len(sentences) == len(output_sentences)

        input_short_sentences = [s for (s, o) in zip(sentences, output_sentences) if o is None]
        output_short_sentences = self._translate_sentences(input_short_sentences)

        assert len(input_short_sentences) == len(output_short_sentences)

        short_idx = 0
        for idx in range(len(sentences)):
            if output_sentences[idx] is None:
                output_sentences[idx] = output_short_sentences[short_idx]
                short_idx += 1

        return output_sentences

    def split_paragraph(self, paragraph):
        if self.src_lang == "eng_Latn":
            seg = pysbd.Segmenter(language="en", clean=False)
            return seg.segment(paragraph)
        else:
            iso_lang = FloresIsoCodes[self.src_lang]
            return sentence_split(paragraph, lang=iso_lang, delim_pat=DELIM_PAT_NO_DANDA)

    def translate_paragraphs(self, paragraphs):
        sentences, partitions = [], [0]
        for paragraph in paragraphs:
            sentences.extend(self.split_paragraph(paragraph))
            partitions.append(len(sentences))

        output_sentences = self.translate_sentences(sentences)

        output_paragraphs = [" ".join(output_sentences[s:e]) for (s, e) in pairwise(partitions)]

        return output_paragraphs

    def translate_file(self, input_file, output_file):
        """Handles 3 kinds of files
        txt: one sentence per line
        json: list of sentences json: list of 'sentences' and 'paragraphs'.
        """

        if isinstance(input_file, (str, Path)):
            input_file = Path(input_file)
        else:
            raise ValueError("Unknown type of input_file: {type(input_file)}")

        if isinstance(output_file, (str, Path)):
            output_file = Path(output_file)
        else:
            raise ValueError("Unknown type of input_file: {type(input_file)}")

        sentences, paragraphs = [], []
        input_format = ""
        if input_file.suffix.lower() == "json":
            inputs = json.loads(input_file.read_text())
            if type(inputs, list):
                sentences = inputs
                input_format = "json_sent"

            else:
                sentences, paragraphs = inputs["sentences"], inputs["paragraphs"]
                input_format = "json_both"
        else:
            sentences = input_file.read_text().strip().split("\n")
            input_format = "text"

        output_paragraphs = self.translate_paragraphs(paragraphs)
        output_sentences = self.translate_sentences(sentences)

        if input_format == "text":
            assert not output_paragraphs
            output_file.write_text("\n".join(output_sentences))
        elif input_format == "json_both":
            output_file.write_text(
                json.dumps({"sentences": output_sentences, "paragraphs": output_paragraphs})
            )
        else:
            output_file.write_text(json.dumps(output_sentences))


def main():
    input = sys.argv[1]

    translator = Translator()
    if Path(input).exists():
        input_file = Path(input)
        output_file = input_file.parent / f"{input_file.stem}.trans{input_file.suffix}"
        translator.translate_file(input_file, output_file)
        # print(output_file.read_text())
    else:
        input_sentence = input
        output_sentences = translator.translate_paragraphs([input_sentence])
        print(output_sentences)


if __name__ == "__main__":
    main()


"""
import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
batch = tokenizer(batch, src=True, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

outputs = tokenizer.batch_decode(outputs, src=False)
outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
print(outputs)

>>> ['यह एक परीक्षण वाक्य है।', 'यह एक और लंबा अलग परीक्षण वाक्य है।', 'कृपया 9876543210 पर एक एस. एम. एस. भेजें और 15 अक्टूबर, 2023 तक newemail123@xyz.com पर एक ईमेल भेजें।']

"""
