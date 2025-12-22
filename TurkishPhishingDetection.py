"""
Turkish Phishing Email Detection Project
PyTorch Implementation with Transformer Models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, MarianMTModel, MarianTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
from tqdm import tqdm
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

class DataLoaderUtil:
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
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = "Helsinki-NLP/opus-mt-en-tr"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
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
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
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
        return [self.translate_email(email) for email in tqdm(emails, desc="Translating")]


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
# PYTORCH DATASET
# ============================================================================

class EmailDataset(Dataset):
    """PyTorch Dataset for email data"""
    
    def __init__(self, emails: List[EmailData], tokenizer, preprocessor, 
                 feature_extractor, max_length: int = 512):
        self.emails = emails
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email = self.emails[idx]
        text = self.preprocessor.preprocess(email)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        heuristic_features = list(self.feature_extractor.extract_features(email).values())
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'heuristic_features': torch.tensor(heuristic_features, dtype=torch.float32),
            'labels': torch.tensor(email.label, dtype=torch.long)
        }


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerClassifier(nn.Module):
    """Transformer-based classifier with heuristic feature fusion"""
    
    def __init__(self, model_name: str, num_labels: int = 2, heuristic_dim: int = 13, dropout: float = 0.3):
        super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.heuristic_dense = nn.Linear(heuristic_dim, 32)
        self.relu = nn.ReLU()
        
        # Get transformer hidden size
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size + 32, num_labels)
    
    def forward(self, input_ids, attention_mask, heuristic_features):
        """Forward pass combining transformer and heuristic features"""
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = self.dropout(transformer_out.last_hidden_state[:, 0, :])
        heuristic = self.relu(self.heuristic_dense(heuristic_features))
        combined = torch.cat([pooled, heuristic], dim=1)
        
        return self.classifier(combined)


# ============================================================================
# PHISHING DETECTOR
# ============================================================================

class PhishingDetector:
    """Main detector class for training and evaluation"""
    
    def __init__(self, model_name: str, max_length: int = 512, learning_rate: float = 2e-5,
                 batch_size: int = 16, epochs: int = 3, patience: int = 2, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TransformerClassifier(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = HeuristicFeatureExtractor()
        
        print(f"Using device: {self.device}")
    
    def create_dataloader(self, emails: List[EmailData], shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader"""
        dataset = EmailDataset(
            emails, self.tokenizer, self.preprocessor, 
            self.feature_extractor, self.max_length
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(self, train_emails: List[EmailData], val_emails: List[EmailData]):
        """Train the model"""
        train_loader = self.create_dataloader(train_emails, shuffle=True)
        val_loader = self.create_dataloader(val_emails, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                heuristic_features = batch['heuristic_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, heuristic_features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        return history
    
    def _validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                heuristic_features = batch['heuristic_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, heuristic_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return val_loss / len(val_loader), correct / total
    
    def evaluate(self, test_emails: List[EmailData]) -> Dict[str, float]:
        """Evaluate model on test set"""
        test_loader = self.create_dataloader(test_emails, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                heuristic_features = batch['heuristic_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, heuristic_features)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': roc_auc_score(all_labels, all_probabilities),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
    
    def predict(self, email: EmailData) -> Tuple[int, float]:
        """Predict single email"""
        self.model.eval()
        text = self.preprocessor.preprocess(email)
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        
        heuristic_features = list(self.feature_extractor.extract_features(email).values())
        heuristic_tensor = torch.tensor([heuristic_features], dtype=torch.float32)
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            heuristic_tensor = heuristic_tensor.to(self.device)
            
            outputs = self.model(input_ids, attention_mask, heuristic_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return prediction.item(), probabilities[0, prediction].item()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {path}")


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
        train, val, test = DataLoaderUtil("").split_data(turkish_emails)
        
        params = hyperparams or {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 3}
        detector = PhishingDetector(model_name="dbmdz/bert-base-turkish-cased", **params)
        
        detector.train(train, val)
        results = detector.evaluate(test)
        
        self.results['approach_1_translated'] = results
        detector.save_model(f"{self.output_dir}/approach1_model.pt")
        
        return results
    
    def run_approach_2_multilingual(self, english_emails: List[EmailData],
                                   translated_emails: List[EmailData],
                                   hyperparams: Dict = None, few_shot_samples: int = 0):
        """Approach 2: XLM-R multilingual transfer learning"""
        mode = "zero-shot" if few_shot_samples == 0 else f"few-shot-{few_shot_samples}"
        print(f"\n=== Approach 2: XLM-R Multilingual ({mode}) ===")
        
        train_en, val_en, _ = DataLoaderUtil("").split_data(english_emails)
        train_tr, val_tr, test_tr = DataLoaderUtil("").split_data(translated_emails)
        
        params = hyperparams or {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 3}
        detector = PhishingDetector(model_name="xlm-roberta-base", **params)
        
        detector.train(train_en, val_en)
        
        if few_shot_samples > 0:
            detector.epochs = 2
            detector.learning_rate = 1e-5
            detector.train(train_tr[:few_shot_samples], val_tr)
        
        results = detector.evaluate(test_tr)
        
        self.results[f'approach_2_{mode}'] = results
        detector.save_model(f"{self.output_dir}/approach2_{mode}_model.pt")
        
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
    print("Initializing Turkish Phishing Detection Project (PyTorch)...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load and translate data
    loader = DataLoaderUtil("path/to/english/dataset")
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