"""
Turkish Phishing Email Detection Project
Streamlined Implementation with Transformer Models
"""

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EmailData:
    """Structure to hold email information"""
    subject: str
    body: str
    sender: str
    headers: Dict[str, str]
    urls: List[str]
    label: int  # 0: legitimate, 1: phishing


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Handles loading and splitting phishing email datasets"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_english_dataset(self) -> List[EmailData]:
        """Load English phishing dataset - implement based on your data format"""
        # TODO: Implement dataset loading
        # Example: df = pd.read_csv(self.data_path)
        return []
    
    def split_data(self, emails: List[EmailData], test_size: float = 0.15, 
                   val_size: float = 0.15, random_state: int = 42) -> Tuple[List, List, List]:
        """Split data into train, validation, and test sets"""
        train_val, test = train_test_split(emails, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
        return train, val, test


# ============================================================================
# MACHINE TRANSLATION
# ============================================================================

class MachineTranslator:
    """Handles translation from English to Turkish using Marian NMT"""
    
    def __init__(self):
        model_name = "Helsinki-NLP/opus-mt-en-tr"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate_email(self, email: EmailData) -> EmailData:
        """Translate email while preserving URLs and metadata"""
        translated_subject = self._translate_text(email.subject)
        translated_body = self._translate_with_urls(email.body, email.urls)
        
        return EmailData(
            subject=translated_subject,
            body=translated_body,
            sender=email.sender,
            headers=email.headers,
            urls=email.urls,
            label=email.label
        )
    
    def _translate_text(self, text: str) -> str:
        """Translate plain text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def _translate_with_urls(self, text: str, urls: List[str]) -> str:
        """Translate text while preserving URLs"""
        protected_text = text
        placeholders = {url: f"__URL_{i}__" for i, url in enumerate(urls)}
        
        for url, placeholder in placeholders.items():
            protected_text = protected_text.replace(url, placeholder)
        
        translated = self._translate_text(protected_text)
        
        for url, placeholder in placeholders.items():
            translated = translated.replace(placeholder, url)
        
        return translated
    
    def batch_translate_dataset(self, emails: List[EmailData]) -> List[EmailData]:
        """Translate entire dataset"""
        return [self.translate_email(email) for email in emails]


# ============================================================================
# HEURISTIC FEATURE EXTRACTION
# ============================================================================

class HeuristicFeatureExtractor:
    """Extract security-relevant heuristic features from emails"""
    
    def __init__(self):
        self.patterns = {
            'urgency': ['urgent', 'immediately', 'action required', 'verify', 'suspend', 
                       'acil', 'hemen', 'doğrula'],
            'financial': ['bank', 'credit card', 'account', 'payment', 'banka', 
                         'kredi kartı', 'hesap'],
        }
    
    def extract_features(self, email: EmailData) -> Dict[str, float]:
        """Extract all heuristic features"""
        text = f"{email.subject} {email.body}".lower()
        
        return {
            'urgency_score': sum(1 for w in self.patterns['urgency'] if w in text) / 3.0,
            'financial_score': sum(1 for w in self.patterns['financial'] if w in text) / 3.0,
            'exclamation_count': text.count('!'),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'url_count': len(email.urls),
            'has_suspicious_tld': float(any(url.endswith(('.tk', '.ml', '.ga', '.xyz')) for url in email.urls)),
            'avg_url_length': sum(len(url) for url in email.urls) / max(len(email.urls), 1),
            'has_ip_address': float(any(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) for url in email.urls)),
            'subject_length': len(email.subject),
            'body_length': len(email.body),
            'has_html': float(bool(re.search(r'<[^>]+>', email.body))),
            'link_ratio': sum(len(url) for url in email.urls) / max(len(email.body), 1),
            'sender_has_numbers': float(bool(re.search(r'\d', email.sender))),
        }


# ============================================================================
# LEMMATIZATION
# ============================================================================

class TurkishLemmatizer:
    """Turkish text lemmatization using Zeyrek"""
    
    def __init__(self, use_lemmatization: bool = False):
        self.use_lemmatization = use_lemmatization
        if use_lemmatization:
            try:
                import zeyrek
                self.analyzer = zeyrek.MorphAnalyzer()
            except ImportError:
                print("Warning: Zeyrek not installed. Skipping lemmatization.")
                self.use_lemmatization = False
    
    def lemmatize(self, text: str) -> str:
        """Lemmatize Turkish text"""
        if not self.use_lemmatization:
            return text
        
        words = text.split()
        lemmas = []
        
        for word in words:
            try:
                results = self.analyzer.lemmatize(word)
                lemma = results[0][1][0] if results and results[0][1] else word
                lemmas.append(lemma)
            except:
                lemmas.append(word)
        
        return ' '.join(lemmas)


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Preprocess text for model input"""
    
    def __init__(self, lowercase: bool = True, use_lemmatization: bool = False):
        self.lowercase = lowercase
        self.lemmatizer = TurkishLemmatizer(use_lemmatization)
    
    def preprocess(self, email: EmailData) -> str:
        """Preprocess email text"""
        text = f"{email.subject} [SEP] {email.body}"
        
        if self.lowercase:
            text = text.lower()
        
        text = self.lemmatizer.lemmatize(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerClassifier(tf.keras.Model):
    """Transformer-based classifier with heuristic feature fusion"""
    
    def __init__(self, model_name: str, num_labels: int = 2, heuristic_dim: int = 13):
        super().__init__()
        self.model_name = model_name
        self.transformer = TFAutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.heuristic_dense = tf.keras.layers.Dense(32, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')
    
    def call(self, inputs, training=False):
        """Forward pass combining transformer and heuristic features"""
        transformer_out = self.transformer(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        pooled = self.dropout(transformer_out.last_hidden_state[:, 0, :], training=training)
        heuristic = self.heuristic_dense(inputs['heuristic_features'])
        combined = tf.keras.layers.Concatenate()([pooled, heuristic])
        
        return self.classifier(combined)


# ============================================================================
# PHISHING DETECTOR
# ============================================================================

class PhishingDetector:
    """Main detector class for training and evaluation"""
    
    def __init__(self, model_name: str, max_length: int = 512, learning_rate: float = 2e-5,
                 batch_size: int = 16, epochs: int = 3, patience: int = 2):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.model = TransformerClassifier(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = HeuristicFeatureExtractor()
    
    def prepare_dataset(self, emails: List[EmailData]) -> tf.data.Dataset:
        """Convert emails to TensorFlow dataset"""
        texts = [self.preprocessor.preprocess(email) for email in emails]
        labels = [email.label for email in emails]
        
        encodings = self.tokenizer(texts, truncation=True, padding='max_length',
                                   max_length=self.max_length, return_tensors='tf')
        
        heuristic_features = np.array([
            list(self.feature_extractor.extract_features(email).values())
            for email in emails
        ])
        
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'heuristic_features': tf.constant(heuristic_features, dtype=tf.float32)
        }
        
        return tf.data.Dataset.from_tensor_slices((dataset_dict, tf.constant(labels, dtype=tf.int32)))
    
    def train(self, train_emails: List[EmailData], val_emails: List[EmailData]):
        """Train the model"""
        train_dataset = self.prepare_dataset(train_emails).shuffle(1000).batch(self.batch_size)
        val_dataset = self.prepare_dataset(val_emails).batch(self.batch_size)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience, restore_best_weights=True
        )
        
        return self.model.fit(train_dataset, validation_data=val_dataset, 
                            epochs=self.epochs, callbacks=[early_stop])
    
    def evaluate(self, test_emails: List[EmailData]) -> Dict[str, float]:
        """Evaluate model on test set"""
        test_dataset = self.prepare_dataset(test_emails).batch(self.batch_size)
        predictions = self.model.predict(test_dataset)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = [email.label for email in test_emails]
        
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': roc_auc_score(true_labels, predictions[:, 1]),
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels)
        }
    
    def predict(self, email: EmailData) -> Tuple[int, float]:
        """Predict single email"""
        dataset = self.prepare_dataset([email]).batch(1)
        prediction = self.model.predict(dataset)
        return np.argmax(prediction), np.max(prediction)
    
    def save_model(self, path: str):
        """Save model weights"""
        self.model.save_weights(path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_weights(path)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run and compare different approaches"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        self.results = {}
    
    def run_approach_1_translated(self, english_emails: List[EmailData], 
                                  translator: MachineTranslator, hyperparams: Dict = None):
        """Approach 1: Train BERTurk on translated Turkish data"""
        print("\n=== Approach 1: BERTurk on Translated Turkish ===")
        
        turkish_emails = translator.batch_translate_dataset(english_emails)
        train, val, test = DataLoader("").split_data(turkish_emails)
        
        params = hyperparams or {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 3}
        detector = PhishingDetector(model_name="dbmdz/bert-base-turkish-cased", **params)
        
        detector.train(train, val)
        results = detector.evaluate(test)
        
        self.results['approach_1_translated'] = results
        detector.save_model(f"{self.output_dir}/approach1_model")
        
        return results
    
    def run_approach_2_multilingual(self, english_emails: List[EmailData],
                                   translated_emails: List[EmailData],
                                   hyperparams: Dict = None, few_shot_samples: int = 0):
        """Approach 2: XLM-R multilingual transfer learning"""
        mode = "zero-shot" if few_shot_samples == 0 else f"few-shot-{few_shot_samples}"
        print(f"\n=== Approach 2: XLM-R Multilingual ({mode}) ===")
        
        train_en, val_en, _ = DataLoader("").split_data(english_emails)
        train_tr, val_tr, test_tr = DataLoader("").split_data(translated_emails)
        
        params = hyperparams or {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 3}
        detector = PhishingDetector(model_name="xlm-roberta-base", **params)
        
        detector.train(train_en, val_en)
        
        if few_shot_samples > 0:
            detector.epochs = 2
            detector.learning_rate = 1e-5
            detector.train(train_tr[:few_shot_samples], val_tr)
        
        results = detector.evaluate(test_tr)
        
        self.results[f'approach_2_{mode}'] = results
        detector.save_model(f"{self.output_dir}/approach2_{mode}_model")
        
        return results
    
    def compare_results(self):
        """Print and save comparison of all approaches"""
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        
        df = pd.DataFrame(self.results).T
        print(df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']])
        df.to_csv(f"{self.output_dir}/comparison_results.csv")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("Initializing Turkish Phishing Detection Project...")
    
    # Load and translate data
    loader = DataLoader("path/to/english/dataset")
    english_emails = loader.load_english_dataset()
    
    translator = MachineTranslator()
    translated_emails = translator.batch_translate_dataset(english_emails)
    
    # Run experiments
    runner = ExperimentRunner(output_dir="./results")
    
    # Approach 1: Translated data with BERTurk
    runner.run_approach_1_translated(english_emails, translator)
    
    # Approach 2: Multilingual transfer with XLM-R
    runner.run_approach_2_multilingual(english_emails, translated_emails, few_shot_samples=0)
    runner.run_approach_2_multilingual(english_emails, translated_emails, few_shot_samples=50)
    runner.run_approach_2_multilingual(english_emails, translated_emails, few_shot_samples=100)
    
    # Compare results
    runner.compare_results()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE!")
    print(f"Results saved to ./results/")
    print("="*80)


if __name__ == "__main__":
    main()