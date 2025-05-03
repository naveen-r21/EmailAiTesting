import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from .test_data_generator import TestDataGenerator
from .evaluation_metrics import EvaluationMetrics

class TestRunner:
    def __init__(self, model):
        self.model = model
        self.test_generator = TestDataGenerator()
        self.metrics = EvaluationMetrics()
        
    def run_tests(self, 
                 num_samples: int = 100,
                 balanced_sentiments: bool = True) -> Dict[str, Any]:
        """Run comprehensive tests on the model."""
        # Generate test dataset
        test_data = self.test_generator.generate_test_dataset(
            num_samples=num_samples,
            balanced_sentiments=balanced_sentiments
        )
        
        # Get model predictions
        predictions = []
        ground_truth = []
        
        for email in test_data:
            # Get model prediction
            prediction = self.model.analyze_email(email)
            
            # Extract ground truth from test data
            true_sentiment = email.get("sentiment", "neutral")
            true_events = email.get("events", [])
            true_summary = email.get("mail_summary", "")
            
            predictions.append({
                "sentiment": prediction["overall_sentiment_analysis"],
                "events": prediction["Events"],
                "summary": prediction["Summary"]
            })
            
            ground_truth.append({
                "sentiment": true_sentiment,
                "events": true_events,
                "summary": true_summary
            })
        
        # Evaluate performance
        sentiment_metrics = self.metrics.evaluate_sentiment(
            [p["sentiment"] for p in predictions],
            [g["sentiment"] for g in ground_truth]
        )
        
        event_metrics = self.metrics.evaluate_event_extraction(
            [p["events"] for p in predictions],
            [g["events"] for g in ground_truth]
        )
        
        summary_metrics = self.metrics.evaluate_summary(
            [p["summary"] for p in predictions],
            [g["summary"] for g in ground_truth]
        )
        
        # Compute overall metrics
        overall_metrics = self.metrics.compute_overall_metrics(
            sentiment_metrics,
            event_metrics,
            summary_metrics
        )
        # Compute per-key and per-event-key pass/fail and overall metrics
        per_key_metrics = self.metrics.evaluate_per_key(predictions, ground_truth)
        
        # Compute simplified metrics for the new report format
        simplified_metrics = self.metrics.compute_simplified_metrics(predictions, ground_truth)
        
        return {
            "sentiment_metrics": sentiment_metrics,
            "event_metrics": event_metrics,
            "summary_metrics": summary_metrics,
            "overall_metrics": overall_metrics,
            "per_key_metrics": per_key_metrics,
            "simplified_metrics": simplified_metrics,
            "test_data": test_data,
            "predictions": predictions,
            "ground_truth": ground_truth
        }
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       output_dir: str = "test_reports") -> str:
        """Generate a comprehensive test report with visualizations."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(report_dir, "raw_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(results, report_dir)
        
        # Generate summary report
        self._generate_summary_report(results, report_dir)
        
        return report_dir
    
    def _generate_visualizations(self, results: Dict[str, Any], report_dir: str):
        """Generate visualizations for the test results."""
        # Sentiment distribution
        sentiments = [p["sentiment"] for p in results["predictions"]]
        plt.figure(figsize=(10, 6))
        sns.countplot(x=sentiments)
        plt.title("Sentiment Distribution in Predictions")
        plt.savefig(os.path.join(report_dir, "sentiment_distribution.png"))
        plt.close()
        
        # Event extraction performance
        metrics_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Value": [
                results["event_metrics"]["precision"],
                results["event_metrics"]["recall"],
                results["event_metrics"]["f1_score"]
            ]
        })
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", data=metrics_df)
        plt.title("Event Extraction Performance")
        plt.savefig(os.path.join(report_dir, "event_extraction_performance.png"))
        plt.close()
        
        # Summary quality distribution
        bleu_scores = []
        for pred, true in zip(results["predictions"], results["ground_truth"]):
            pred_tokens = word_tokenize(pred["summary"].lower())
            true_tokens = word_tokenize(true["summary"].lower())
            score = sentence_bleu([true_tokens], pred_tokens)
            bleu_scores.append(score)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(bleu_scores, bins=20)
        plt.title("Summary Quality (BLEU Score) Distribution")
        plt.savefig(os.path.join(report_dir, "summary_quality_distribution.png"))
        plt.close()
    
    def _generate_summary_report(self, results: Dict[str, Any], report_dir: str):
        """Generate a text summary of the test results."""
        # Create original detailed report
        per_key = results["per_key_metrics"]["per_key_percent"]
        per_event_key = results["per_key_metrics"]["per_event_key_percent"]
        overall_acc = results["per_key_metrics"]["overall_accuracy"]
        overall_prec = results["per_key_metrics"]["overall_precision"]
        overall_rec = results["per_key_metrics"]["overall_recall"]
        overall_f1 = results["per_key_metrics"]["overall_f1"]
        
        # Generate detailed report (original format)
        detailed_summary = f"""
Test Report Summary (Detailed)
============================

Overall Performance
------------------
Overall Score: {results['overall_metrics']['overall_score']:.3f}
Sentiment F1: {results['overall_metrics']['sentiment_f1']:.3f}
Event F1: {results['overall_metrics']['event_f1']:.3f}
Summary BLEU: {results['overall_metrics']['summary_bleu']:.3f}

Sentiment Analysis
-----------------
Accuracy: {results['sentiment_metrics']['accuracy']:.3f}
Precision: {results['sentiment_metrics']['precision']:.3f}
Recall: {results['sentiment_metrics']['recall']:.3f}
F1-Score: {results['sentiment_metrics']['f1_score']:.3f}

Event Extraction
---------------
Precision: {results['event_metrics']['precision']:.3f}
Recall: {results['event_metrics']['recall']:.3f}
F1-Score: {results['event_metrics']['f1_score']:.3f}

Summary Generation
-----------------
Average BLEU Score: {results['summary_metrics']['bleu_score']:.3f}
BLEU Score Std: {results['summary_metrics']['bleu_score_std']:.3f}

Per-Key Pass Rates (%)
----------------------
"""
        for k, v in per_key.items():
            detailed_summary += f"{k}: {v:.1f}%\n"
        detailed_summary += "\nPer-Event-Key Pass Rates (%)\n--------------------------\n"
        for k, v in per_event_key.items():
            detailed_summary += f"{k}: {v:.1f}%\n"
        detailed_summary += f"""

Overall Model Metrics (%)
------------------------
Accuracy: {overall_acc:.1f}%
Precision: {overall_prec:.1f}%
Recall: {overall_rec:.1f}%
F1-Score: {overall_f1:.1f}%
"""
        # Save detailed report
        with open(os.path.join(report_dir, "detailed_summary.txt"), "w") as f:
            f.write(detailed_summary)
        
        # Generate simplified report
        simplified = results["simplified_metrics"]
        
        simplified_summary = f"""
Test Report Summary
==================

Overall Metrics:
--------------
Overall accuracy: {simplified['overall_accuracy']:.1f}%
Sentiment accuracy: {simplified['sentiment_accuracy']:.1f}%
Events accuracy: {simplified['events_accuracy']:.1f}%
Summary accuracy: {simplified['summary_accuracy']:.1f}%

Per-Case Results:
---------------
"""
        # Add per-case results
        for case in simplified["per_case"]:
            simplified_summary += f"\nTest Case {case['mail_id']}:\n"
            simplified_summary += f"Sentiment: {'PASS' if case['sentiment_pass'] else 'FAIL'}\n"
            simplified_summary += f"Summary: {'PASS' if case['summary_pass'] else 'FAIL'}\n"
            simplified_summary += f"Events: {'PASS' if case['events_pass'] else 'FAIL'}\n"
        
        # Add overall pass/fail
        simplified_summary += f"""
Overall Results:
--------------
Overall Sentiment: {'PASS' if simplified['overall_sentiment_pass'] else 'FAIL'}
Overall Summary: {'PASS' if simplified['overall_summary_pass'] else 'FAIL'}
Overall Events: {'PASS' if simplified['overall_events_pass'] else 'FAIL'}
"""
        
        # Save simplified report as the main summary
        with open(os.path.join(report_dir, "summary.txt"), "w") as f:
            f.write(simplified_summary) 