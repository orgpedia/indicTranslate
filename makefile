.DEFAULT_GOAL := help

.PHONY: help install test test_install test_run


help:
	$(info Please use 'make <target>', where <target> is one of)
	$(info )
	$(info   install     install packages and prepare software environment)
	$(info )
	$(info   test        test the translator)
	$(info )
	$(info Check the makefile to know exactly what each target is doing.)
	@echo # dummy command


install: pyproject.toml
	poetry install

format:
	poetry run ruff format indicTranslate/*.py tests/*.py

lint:
	poetry run ruff indicTranslate/*.py tests/*.py


test_install:
	poetry install --only=test

test: test_install format lint
	poetry run pytest


test_run:
	poetry run python indicTranslate/translator.py  tests/eng_sentences.txt 
	diff tests/eng_sentences.trans.txt tests/eng_sentences.trans.ref.txt && rm -f tests/eng_sentences.trans.txt
