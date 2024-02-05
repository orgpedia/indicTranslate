.DEFAULT_GOAL := help

.PHONY: help install test


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

test:
	poetry run python indicTranslate/translator.py  tests/eng_sentences.txt 
