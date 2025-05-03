from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')

class EvaluationMetrics:
    def __init__(self):
        self.sentiment_labels = ["positive", "negative", "neutral"]
        
    def evaluate_sentiment(self, 
                         predictions: List[str], 
                         ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate sentiment classification performance."""
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, labels=self.sentiment_labels, average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def evaluate_event_extraction(self, 
                                predictions: List[List[Dict]], 
                                ground_truth: List[List[Dict]]) -> Dict[str, float]:
        """Evaluate event extraction performance."""
        total_events = sum(len(events) for events in ground_truth)
        correct_events = 0
        
        for pred_events, true_events in zip(predictions, ground_truth):
            for pred_event in pred_events:
                for true_event in true_events:
                    if self._compare_events(pred_event, true_event):
                        correct_events += 1
                        break
        
        precision = correct_events / sum(len(events) for events in predictions) if predictions else 0
        recall = correct_events / total_events if total_events > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def evaluate_summary(self, 
                        predictions: List[str], 
                        ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate summary quality using BLEU score."""
        scores = []
        for pred, true in zip(predictions, ground_truth):
            pred_tokens = word_tokenize(pred.lower())
            true_tokens = word_tokenize(true.lower())
            score = sentence_bleu([true_tokens], pred_tokens)
            scores.append(score)
        
        return {
            "bleu_score": np.mean(scores),
            "bleu_score_std": np.std(scores)
        }
    
    def _compare_events(self, event1: Dict, event2: Dict) -> bool:
        """Compare two events for equality."""
        required_fields = ["Event name", "Date", "Time", "Location"]
        return all(event1.get(field) == event2.get(field) for field in required_fields)
    
    def compute_overall_metrics(self, 
                              sentiment_metrics: Dict[str, float],
                              event_metrics: Dict[str, float],
                              summary_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute overall performance metrics."""
        return {
            "overall_score": (
                sentiment_metrics["f1_score"] * 0.4 +
                event_metrics["f1_score"] * 0.4 +
                summary_metrics["bleu_score"] * 0.2
            ),
            "sentiment_f1": sentiment_metrics["f1_score"],
            "event_f1": event_metrics["f1_score"],
            "summary_bleu": summary_metrics["bleu_score"]
        }
    
    def evaluate_per_key(self, predictions: List[dict], ground_truth: List[dict]) -> dict:
        """
        Evaluate pass/fail for each of the 7 top-level keys and 6 event sub-keys, and compute percentages and overall metrics.
        Returns a dict with per-key pass rates and overall precision, recall, f1, accuracy.
        """
        # Define keys
        top_keys = [
            "Sentiment analysis",
            "overall_sentiment_analysis",
            "feature",
            "category",
            "Summary",
            "Events",
            "mail_id"  # You can replace this with another key if needed
        ]
        event_keys = [
            "Event name",
            "Date",
            "Time",
            "Property Type",
            "Agent Name",
            "Location"
        ]
        
        per_key_results = {k: [] for k in top_keys}
        per_event_key_results = {k: [] for k in event_keys}
        total_cases = len(predictions)
        total_event_fields = 0
        correct_event_fields = 0
        
        for pred, gt in zip(predictions, ground_truth):
            for k in top_keys:
                if k == "Events":
                    # Compare events list
                    pred_events = pred.get("Events", [])
                    gt_events = gt.get("events", [])
                    # For each event in ground truth, try to find a matching event in prediction
                    for gt_event in gt_events:
                        matched = False
                        for pred_event in pred_events:
                            if all(pred_event.get(ek) == gt_event.get(ek) for ek in event_keys):
                                matched = True
                                break
                        per_key_results["Events"].append(1 if matched else 0)
                    # Per-event-key comparison
                    for gt_event in gt_events:
                        for ek in event_keys:
                            total_event_fields += 1
                            found = False
                            for pred_event in pred_events:
                                if pred_event.get(ek) == gt_event.get(ek):
                                    found = True
                                    break
                            per_event_key_results[ek].append(1 if found else 0)
                            if found:
                                correct_event_fields += 1
                else:
                    gt_val = gt.get(k) if k != "Summary" else gt.get("mail_summary")
                    pred_val = pred.get(k)
                    per_key_results[k].append(1 if pred_val == gt_val else 0)
        # Calculate per-key pass rates
        per_key_percent = {k: 100 * sum(v) / len(v) if v else 0 for k, v in per_key_results.items()}
        per_event_key_percent = {k: 100 * sum(v) / len(v) if v else 0 for k, v in per_event_key_results.items()}
        # Overall accuracy: total correct fields / total fields
        total_fields = sum(len(v) for v in per_key_results.values()) + total_event_fields
        total_correct = sum(sum(v) for v in per_key_results.values()) + correct_event_fields
        overall_accuracy = 100 * total_correct / total_fields if total_fields else 0
        # For overall precision/recall/f1, treat each field as a binary classification
        precision = recall = f1 = overall_accuracy  # For this context, treat as same
        return {
            "per_key_percent": per_key_percent,
            "per_event_key_percent": per_event_key_percent,
            "overall_accuracy": overall_accuracy,
            "overall_precision": precision,
            "overall_recall": recall,
            "overall_f1": f1
        }
    
    def compute_simplified_metrics(self, predictions: List[dict], ground_truth: List[dict]) -> dict:
        """Compute simplified metrics and per-case pass/fail status."""
        # Initialize results
        results = {
            "per_case": [],
            "overall_accuracy": 0.0,
            "sentiment_accuracy": 0.0,
            "events_accuracy": 0.0,
            "summary_accuracy": 0.0,
            "overall_sentiment_pass": True,
            "overall_summary_pass": True,
            "overall_events_pass": True
        }
        
        # Event keys to compare
        event_keys = ["Event name", "Date", "Time", "Property Type", "Agent Name", "Location"]
        
        # Track pass/fail counts
        total_cases = len(predictions)
        sentiment_passes = 0
        events_passes = 0
        summary_passes = 0
        
        # Evaluate each test case
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            case_result = {
                "mail_id": gt.get("mail_id", f"Case-{i+1}"),
                "sentiment_pass": False,
                "summary_pass": False,
                "events_pass": False
            }
            
            # Check sentiment
            gt_sentiment = gt.get("sentiment", "neutral")
            pred_sentiment = pred.get("overall_sentiment_analysis", "")
            sentiment_pass = (gt_sentiment == pred_sentiment)
            case_result["sentiment_pass"] = sentiment_pass
            if sentiment_pass:
                sentiment_passes += 1
            else:
                results["overall_sentiment_pass"] = False
                
            # Check summary
            gt_summary = gt.get("mail_summary", "")
            pred_summary = pred.get("Summary", "")
            summary_pass = (gt_summary == pred_summary)
            case_result["summary_pass"] = summary_pass
            if summary_pass:
                summary_passes += 1
            else:
                results["overall_summary_pass"] = False
                
            # Check events
            gt_events = gt.get("events", [])
            pred_events = pred.get("Events", [])
            
            # For simplicity, if no events in ground truth but events in prediction, or vice versa, it's a fail
            if (not gt_events and pred_events) or (gt_events and not pred_events):
                events_pass = False
            else:
                # If both are empty, it's a pass
                if not gt_events and not pred_events:
                    events_pass = True
                else:
                    # Otherwise, check each event
                    events_pass = True
                    # For each event in ground truth, check if there's a matching event in prediction
                    for gt_event in gt_events:
                        event_found = False
                        for pred_event in pred_events:
                            # Check if all fields match
                            if all(pred_event.get(key) == gt_event.get(key) for key in event_keys):
                                event_found = True
                                break
                        # If any event is not found, overall events_pass is False
                        if not event_found:
                            events_pass = False
                            break
            
            case_result["events_pass"] = events_pass
            if events_pass:
                events_passes += 1
            else:
                results["overall_events_pass"] = False
                
            results["per_case"].append(case_result)
        
        # Calculate accuracies
        results["overall_accuracy"] = 100.0 * (sentiment_passes + events_passes + summary_passes) / (total_cases * 3)
        results["sentiment_accuracy"] = 100.0 * sentiment_passes / total_cases
        results["events_accuracy"] = 100.0 * events_passes / total_cases
        results["summary_accuracy"] = 100.0 * summary_passes / total_cases
        
        return results 