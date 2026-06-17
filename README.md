# Turkish Phishing Email Detection: Cross-Lingual Transfer Learning Approaches

Phishing attacks represent a pervasive cybersecurity threat, but automated detection systems remain highly unequal, achieving near-perfect accuracy in English while leaving low-resource languages like Turkish underserved. This research addresses the challenge of building robust phishing email detection systems for Turkish in the complete absence of native, authentic, labeled datasets. By leveraging cross-lingual transfer learning and neural machine translation, this work demonstrates how English email datasets can be utilized to train highly accurate Turkish classifiers.

In this work, we compare two primary cross-lingual transfer learning strategies: a machine translation pipeline combined with language-specific monolingual transformers, and multilingual representations utilizing zero-shot and few-shot transfer. Our experiments demonstrate outstanding performance. The best configuration, a Turkish-specific BERT model (BERTurk) trained on machine-translated data, achieves a near-perfect classification performance of 99.73% accuracy and a 99.76% F1-score. Multilingual transfer using XLM-RoBERTa in a zero-shot setting (trained on English emails, evaluated on Turkish emails) yields a highly competitive 95.59% accuracy and a 95.99% F1-score without seeing a single Turkish example during training. A mixed training strategy combining 100 Turkish and 200 English samples regularizes XLM-RoBERTa to achieve 94.02% accuracy. In contrast, a traditional baseline using TF-IDF features with Logistic Regression achieves 98.83% accuracy on full training data, but drops to 82.21% accuracy under few-shot constraints, highlighting the limitations of traditional models under extreme data scarcity.

## Methodology

The research methodology uses the CEAS '08 dataset, which contains approximately 39,000 email samples (44% legitimate, 56% phishing). To prevent data leakage and inflated performance, we remove duplicate emails before partitioning the dataset. The dataset is split into stratified partitions consisting of 70% training, 15% validation, and 15% testing splits.

Since authentic Turkish phishing email collections are unavailable, machine translation is used to create synthetic Turkish text. The Helsinki-NLP neural machine translation system is configured for English-to-Turkish translation. To prevent the translation process from corrupting vital phishing indicators, a URL protection protocol is implemented. All URLs in the emails are extracted and replaced with numeric placeholders before translation, and then restored to their original form post-translation. Subject lines and email bodies are translated independently before being concatenated with a separator token for model input.

Three classification architectures are evaluated:

1. **Traditional Machine Learning Baseline**: Logistic Regression with TF-IDF features, using a maximum of 5,000 n-gram features.
2. **Monolingual Strategy**: The BERTurk model, initialized with pre-trained weights from a cased Turkish BERT model, and fine-tuned on the translated training set with its embeddings and first four encoder layers frozen.
3. **Multilingual Strategy**: The XLM-RoBERTa model, pre-trained on 100 languages. This model is evaluated in three contexts:
   - **Zero-Shot**: Trained entirely on English training data and evaluated on the Turkish test data.
   - **Few-Shot**: Fine-tuned on a tiny, balanced sample of 50 Turkish emails.
   - **Mixed Transfer**: Trained on a hybrid dataset composed of 100 Turkish and 200 English samples to stabilize representations.

## Experimental Results

The models were evaluated using accuracy, precision, recall, F1-score, and AUC-ROC metrics. The performance on the Turkish test set is summarized in the table below:

| Approach | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERTurk** | 99.73% | 99.76% | 99.76% | 99.76% | 0.9999 |
| **TF-IDF Baseline** | 98.83% | 99.11% | 98.78% | 98.95% | 0.9993 |
| **XLM-RoBERTa Zero-Shot** | 95.59% | 97.36% | 94.66% | 95.99% | 0.9947 |
| **XLM-RoBERTa Mixed (100TR+200EN)** | 94.02% | 91.59% | 98.32% | 94.83% | 0.9843 |
| **XLM-RoBERTa Few-Shot (50TR)** | 84.35% | 78.37% | 99.39% | 87.64% | 0.9597 |
| **TF-IDF Few-Shot (50TR)** | 82.21% | 93.36% | 73.33% | 82.14% | 0.9436 |

The metrics comparison across the models highlights the overall performance of the methods:

![Performance metrics comparison across different transfer learning approaches](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/metrics_comparison.png)

### Model Behaviors and Tradeoffs

Analyzing the precision-recall tradeoffs reveals distinct model behaviors. BERTurk and the full-data TF-IDF baseline achieve both high precision and high recall, representing the optimal operating region. XLM-RoBERTa Zero-Shot achieves a balanced trade-off, showing that phishing semantic representations transfer effectively without language-specific adjustment.

Under few-shot constraints (50 training samples), traditional and deep learning models behave in opposite ways. The few-shot TF-IDF baseline prioritizes precision (93.36%) but has a very low recall (73.33%), meaning it misses over a quarter of the phishing attacks. In contrast, the few-shot XLM-RoBERTa model achieves a very high recall of 99.39%, missing almost no phishing attacks, but generates a higher false positive rate, leading to lower precision (78.37%). Combining 100 Turkish samples with 200 English samples in the mixed training setup helps regularize XLM-RoBERTa, scaling down false positives while maintaining robust recall.

The figure below plots the precision-recall tradeoffs for all configurations:

![Precision-Recall tradeoff showing model behaviors under constraints](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/precision_recall.png)

### Error Profiles

Confusion matrices detail the exact error distributions of each classification approach. BERTurk shows near-perfect calibration, generating only 8 false positives and 8 false negatives out of 5,874 test samples. The zero-shot XLM-RoBERTa model demonstrates a conservative classification bias, producing 175 false positives and 84 false negatives. In the few-shot configurations, TF-IDF generates a high number of false negatives (874 missed attacks), whereas XLM-RoBERTa generates a high number of false positives (899 false alarms). The mixed configuration stabilizes this behavior, resulting in 296 false positives and 55 false negatives.

The figures below show the confusion matrices for each classification setting:

![Confusion Matrix for BERTurk](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_berturk.png)

![Confusion Matrix for TF-IDF Turkish Baseline](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_tfidf_turkish_baseline.png)

![Confusion Matrix for XLM-RoBERTa Zero-Shot](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_xlmr_zeroshot.png)

![Confusion Matrix for XLM-RoBERTa Few-Shot 50TR](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_xlmr_fewshot_50tr.png)

![Confusion Matrix for TF-IDF Few-Shot 50TR](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_tfidf_fewshot_50tr.png)

![Confusion Matrix for XLM-RoBERTa Mixed 100TR+200EN](c:/Users/efe20/Desktop/Inventory/Programming/Python/Uni CENG/Text Mining/Turkish Phishing Emails/results/figures/cm_xlmr_mixed_100tr_200en.png)

## Discussion and Key Findings

The results show that phishing indicators—such as urgency cues, financial solicitation, and deceptive intents—are largely language-independent, allowing multilingual models to generalize effectively. While synthetic translated datasets offer high performance for fine-tuning monolingual models like BERTurk, the results represent an upper bound since translation can standardize syntax and correct grammatical errors, which might inflate performance.

Zero-shot multilingual models provide a viable alternative for immediate deployment when target-language training data is entirely unavailable. If a very small set of target-language labels can be obtained, a hybrid mixed-training approach combining English and target-language samples delivers a balanced, production-ready classifier.

## Conclusion

This research confirms that effective Turkish phishing email detection is achievable without native labeled datasets. Fine-tuning BERTurk on machine-translated text yields 99.73% accuracy, while XLM-RoBERTa zero-shot transfer achieves 95.59% accuracy. Future work should focus on validating these cross-lingual transfer methods on large-scale collections of authentic, non-synthetic Turkish phishing emails, investigating resource-efficient architectures, and evaluating adversarial robustness.
