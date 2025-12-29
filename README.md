# Turkish Phishing Email Detection

This project implements and evaluates multiple approaches for detecting phishing emails in Turkish using cross-lingual transfer learning. It focuses on scenarios where no native Turkish phishing dataset is available.

## Overview

The pipeline uses a large English phishing dataset and explores three main strategies:

- Machine translation from English to Turkish followed by training a Turkish-specific model (BERTurk)
- Multilingual transfer learning with XLM-RoBERTa using zero-shot and few-shot setups
- A traditional TF-IDF + Logistic Regression baseline

All experiments are evaluated on machine-translated Turkish data.

## Features

- End-to-end training and evaluation pipeline
- URL-aware machine translation with MarianMT
- Transformer-based models using Hugging Face and PyTorch
- TF-IDF baseline for comparison
- Support for zero-shot, few-shot, and mixed-language training
- Automatic metric reporting and result saving

## Models Used

- TF-IDF + Logistic Regression
- BERTurk (dbmdz/bert-base-turkish-cased)
- XLM-RoBERTa (xlm-roberta-base)

## Dataset

- CEAS 2008 phishing email dataset (English) with approximately 39,000 labeled emails
