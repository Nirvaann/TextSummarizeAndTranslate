from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import logging
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from concurrent.futures import ThreadPoolExecutor
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Setup rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Supported models for different target languages
TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "en": "Helsinki-NLP/opus-mt-mul-en",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    # Add more as needed
}

def get_translator_and_tokenizer(target_lang):
    model_name = TRANSLATION_MODELS.get(target_lang, TRANSLATION_MODELS["en"])
    translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return translator, tokenizer

# Setup summarizer
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device, framework="tf")

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

@lru_cache(maxsize=100)
def summarize_and_translate(text, target_lang="en", max_length=150, min_length=30):
    try:
        detected_lang = detect(text)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        if target_lang and detected_lang != target_lang:
            translator, tokenizer = get_translator_and_tokenizer(target_lang)
            inputs = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)
            translated = translator.generate(**inputs)
            summary = tokenizer.decode(translated[0], skip_special_tokens=True)
        return summary, detected_lang
    except Exception as e:
        logging.error(f"Error in summarize_and_translate: {str(e)}")
        return None, None

@lru_cache(maxsize=100)
def translate_text(text, target_lang="en"):
    try:
        detected_lang = detect(text)
        logging.info(f"Translating from {detected_lang} to {target_lang}. Input: {text}")
        # Only translate if target_lang is not 'en'
        if target_lang != "en":
            translator, tokenizer = get_translator_and_tokenizer(target_lang)
            logging.info(f"Using model for {target_lang}")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = translator.generate(**inputs)
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
            logging.info(f"Translation result: {translation}")
        else:
            translation = text
            logging.info("No translation performed (target_lang == 'en')")
        return translation, detected_lang
    except Exception as e:
        logging.error(f"Error in translate_text: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    detected_language = None

    if request.method == 'POST':
        text = request.form.get('text')
        min_length = int(request.form.get('min_length', 30))
        max_length = int(request.form.get('max_length', 150))
        target_lang = request.form.get('target_lang', 'en')
        if text:
            summary, detected_language = summarize_and_translate(text, target_lang, max_length, min_length)

    return render_template(
        'index.html',
        summary=summary,
        detected_language=detected_language
    )

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    translation = None
    translation_detected_language = None

    if request.method == 'POST':
        translate_text_input = request.form.get('translate_text')
        translate_target_lang = request.form.get('translate_target_lang', 'en')
        if translate_text_input:
            translation, translation_detected_language = translate_text(translate_text_input, translate_target_lang)

    return render_template(
        'translate.html',
        translation=translation,
        translation_detected_language=translation_detected_language
    )

@app.route('/summarize', methods=['POST'])
@limiter.limit("10 per minute")
def summarize():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 30)
        target_lang = data.get('target_lang', 'en')

        summary, detected_lang = summarize_and_translate(text, target_lang, max_length, min_length)
        if summary is None:
            return jsonify({'error': 'Failed to process text'}), 500

        return jsonify({
            'summary': summary,
            'detected_language': detected_lang
        })
    except Exception as e:
        logging.error(f"Error in summarize route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)