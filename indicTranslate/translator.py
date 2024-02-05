import json
import os
import sys
from itertools import tee, zip_longest
from more_itertools import pairwise
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split

import pysbd

DefaultModel = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Translator:
    def __init__(self, model_name_or_path=DefaultModel, src_lang="eng_Latn", tgt_lang="hin_Deva"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = IndicTransTokenizer(direction="en-indic")
        self.processor = IndicProcessor(inference=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

    def translate_sentences(self, sentences):
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

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, num_beams=5, num_return_sequences=1, max_length=256
            )

        outputs = self.tokenizer.batch_decode(outputs, src=False)
        outputs = self.processor.postprocess_batch(outputs, lang=self.tgt_lang)

        return outputs

    def split_paragraph(self, paragraph):
        if self.src_lang == "eng_Latn":
            seg = pysbd.Segmenter(language="en", clean=False)
            return seg.segment(paragraph)
        else:
            return sentence_split(paragraph, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)

    def translate_paragraphs(self, paragraphs):
        sentences, partitions = [], [0]
        for paragraph in paragraphs:
            sentences.extend(self.split_paragraph(paragraph))
            partitions.append(len(sentences))

        output_sentences = self.translate_sentences(sentences)

        output_paragraphs = [" ".join(output_sentences[s:e]) for (s, e) in pairwise(partitions)]

        return output_paragraphs

    def translate_file(self, input_file, output_file):
        if isinstance(input_file, (str, Path)):
            input_file = Path(input_file)
        else:
            raise ValueError('Unknown type of input_file: {type(input_file)}')

        if isinstance(output_file, (str, Path)):
            output_file = Path(output_file)
        else:
            raise ValueError('Unknown type of input_file: {type(input_file)}')

        sentences, paragraphs = [], []

        input_format = ''
        if input_file.suffix.lower() == 'json':
            inputs = json.loads(input_file.read_text())
            if type(inputs, list):
                setences = inputs
                input_format = 'json_sent'
                                
            else:
                sentences, paragraphs = inputs['sentences'], inputs['paragraphs']
                input_format = 'json_both'                
        else:
            sentences = input_file.read_text().strip().split('\n')
            input_format = 'text'                            

        output_paragraphs = self.translate_paragraphs(paragraphs)                
        output_sentences = self.translate_sentences(sentences)

        if input_format == 'text':
            assert not output_paragraphs 
            output_file.write_text('\n'.join(output_sentences))
        elif input_format == 'json_both':
            output_file.write_text(json.dumps({'sentences': output_sentences,
                                               'paragraphs': output_paragraphs}))
        else:
            output_file.write_text(json.dumps(output_sentences))

        
def main():
    input = sys.argv[1]

    translator = Translator()    
    if Path(input).exists():
        input_file = Path(input)
        output_file = input_file.parent / f'{input_file.stem}.trans{input_file.suffix}'
        translator.translate_file(input_file, output_file)
        print(output_file.read_text())
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
