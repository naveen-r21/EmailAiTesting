import json
import os
from typing import Dict, Any
import requests
import random

class EmailAnalysisModel:
    """API-based email analysis model."""
    def __init__(self):
        self.api_url = "https://mlemailintegrationservices-gef9fwepguapgwfr.eastus2-01.azurewebsites.net/test_extraction"

    def analyze_email(self, email: dict) -> dict:
        """Analyze an email and return the results."""
        response = requests.post(
            self.api_url,
            json=email,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

def run_single_test(model, email_body, expected_events=None, expected_sentiment=None, expected_summary=None):
    """Run a test with a single email and show the results.
    
    Args:
        model: The email analysis model to test
        email_body: The email body text to analyze
        expected_events: List of expected events (if None, will be auto-generated)
        expected_sentiment: Expected sentiment (if None, will be auto-generated)
        expected_summary: Expected summary (if None, will be auto-generated)
    """
    from .test_data_generator import TestDataGenerator
    from .evaluation_metrics import EvaluationMetrics
    
    # Generate a test email structure
    generator = TestDataGenerator()
    
    # If expected_sentiment is provided, use it, otherwise generate random
    sentiment = expected_sentiment if expected_sentiment else random.choice(["positive", "negative", "neutral"])
    
    # First create an email with random data
    test_email = generator.generate_test_email(sentiment=sentiment)
    
    # Override the email body
    test_email["mail_body"] = email_body
    
    # Override ground truth if provided
    if expected_events is not None:
        test_email["events"] = expected_events
        # Update the summary based on the events
        if expected_summary is None:
            summary = " ".join([f"{e['Event name']} at {e['Location']} on {e['Date']} at {e['Time']}." 
                                for e in expected_events])
            test_email["mail_summary"] = summary
    
    if expected_sentiment is not None:
        test_email["sentiment"] = expected_sentiment
        
    if expected_summary is not None:
        test_email["mail_summary"] = expected_summary
    
    # Get model prediction
    prediction = model.analyze_email(test_email)
    
    # Compare with ground truth
    metrics = EvaluationMetrics()
    
    # Prepare ground truth and prediction as expected by evaluation
    ground_truth = {
        "mail_id": test_email["mail_id"],
        "sentiment": test_email["sentiment"],
        "events": test_email["events"],
        "mail_summary": test_email["mail_summary"]
    }
    
    pred_dict = {
        "sentiment": prediction["overall_sentiment_analysis"],
        "events": prediction["Events"],
        "summary": prediction["Summary"]
    }
    
    # Generate the report
    report = generate_single_test_report(test_email, prediction, ground_truth, pred_dict)
    
    return report

def generate_single_test_report(test_email, prediction, ground_truth, pred_dict):
    """Generate a simplified report for a single test case."""
    # Event keys to compare
    event_keys = ["Event name", "Date", "Time", "Property Type", "Agent Name", "Location"]
    
    # Check sentiment
    gt_sentiment = ground_truth["sentiment"]
    pred_sentiment = prediction["overall_sentiment_analysis"]
    sentiment_pass = (gt_sentiment == pred_sentiment)
    
    # Check summary
    gt_summary = ground_truth["mail_summary"]
    pred_summary = prediction["Summary"]
    summary_pass = (gt_summary == pred_summary)
    
    # Check events
    gt_events = ground_truth["events"]
    pred_events = prediction["Events"]
    events_results = []
    
    if not gt_events and not pred_events:
        events_pass = True
        events_results.append("NA - No events in email")
    elif not gt_events and pred_events:
        events_pass = False
        events_results.append("Failed - Model detected events when none existed")
    elif gt_events and not pred_events:
        events_pass = False
        events_results.append("Failed - Model missed events that existed")
    else:
        events_pass = True
        # For each ground truth event
        for i, gt_event in enumerate(gt_events):
            event_result = {
                "event_idx": i,
                "fields": {}
            }
            
            # Find matching event in prediction
            best_match = None
            best_match_score = 0
            
            for pred_event in pred_events:
                match_score = 0
                for key in event_keys:
                    if pred_event.get(key) == gt_event.get(key):
                        match_score += 1
                if match_score > best_match_score:
                    best_match = pred_event
                    best_match_score = match_score
            
            # If no match found or not a perfect match
            if best_match is None:
                events_pass = False
                for key in event_keys:
                    event_result["fields"][key] = {
                        "expected": gt_event.get(key),
                        "predicted": "Not found",
                        "passed": False
                    }
            else:
                # Compare each field
                for key in event_keys:
                    gt_val = gt_event.get(key)
                    pred_val = best_match.get(key)
                    field_pass = (gt_val == pred_val)
                    if not field_pass:
                        events_pass = False
                    
                    event_result["fields"][key] = {
                        "expected": gt_val,
                        "predicted": pred_val,
                        "passed": field_pass
                    }
            
            events_results.append(event_result)
    
    # Calculate overall metrics
    # For a single test case, we're counting:
    # - sentiment as one field (TP/FP/TN/FN)
    # - summary as one field (TP/FP/TN/FN)
    # - each event field as separate fields
    
    # Count event fields
    total_event_fields = 0
    correct_event_fields = 0
    
    if isinstance(events_results[0], dict):
        for event_result in events_results:
            for key, field in event_result["fields"].items():
                total_event_fields += 1
                if field["passed"]:
                    correct_event_fields += 1
    
    # Total fields: sentiment + summary + event fields
    total_fields = 2 + total_event_fields  # 2 for sentiment and summary
    
    # Correct fields
    correct_fields = (1 if sentiment_pass else 0) + (1 if summary_pass else 0) + correct_event_fields
    
    # Calculate accuracy
    accuracy = correct_fields / total_fields * 100 if total_fields > 0 else 0
    
    # For precision, recall, F1:
    # - We'll treat each field as a binary classification
    # - True Positives (TP): Fields model got right
    # - False Positives (FP): Fields model predicted wrongly
    # - False Negatives (FN): Fields model missed
    
    # For a single evaluation, these are straightforward:
    tp = correct_fields
    fp = total_fields - correct_fields  # Fields that should have been right but weren't
    fn = fp  # Same as fp for this scenario
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create the report
    report = f"""
Test Report for Email: {ground_truth['mail_id']}
===============================================

Email Content:
-------------
{test_email['mail_body']}

Overall Results:
--------------
Sentiment: {'PASS' if sentiment_pass else 'FAIL'} (Expected: {gt_sentiment}, Got: {pred_sentiment})
Summary: {'PASS' if summary_pass else 'FAIL'}
Events: {'PASS' if events_pass else 'FAIL'}

Performance Metrics:
------------------
Precision: {precision:.1f}% (Correct fields out of {tp+fp} fields the model extracted)
Recall: {recall:.1f}% (Correct fields out of {tp+fn} fields that should have been extracted)
F1-Score: {f1:.1f}% (Combined score of precision and recall)

Overall Accuracy: {accuracy:.1f}% (Correct fields: {correct_fields} out of {total_fields} total fields)

Event Details:
------------
"""
    
    if isinstance(events_results[0], str):
        report += events_results[0] + "\n"
    else:
        for event_data in events_results:
            report += f"\nEvent:\n"
            for key, value in event_data["fields"].items():
                report += f"  {key}: {'PASS' if value['passed'] else 'FAIL'} "
                report += f"(Expected: {value['expected']}, Got: {value['predicted']})\n"
    
    report += f"""
Ground Truth Events:
-----------------
{json.dumps(gt_events, indent=2)}

AI's Predicted Events:
--------------
{json.dumps(pred_events, indent=2)}

Ground Truth Summary:
-----------------
{gt_summary}

AI's Predicted Summary:
--------------
{pred_summary}
"""
    
    return report

def main():
    # Initialize the model
    model = EmailAnalysisModel()
    
    # Example of custom email body
    email_body = """
    Hi Emi, I was able to talk with Agent Jennifer and she said that they were open till 5pm tomorrow 
    for furnished house. So 1pm should be comfortable enough for me. Let's meet at the shuttle bus stop 
    like we usually do tomorrow at 1pm. I will also provide you a signed copy of the lease agreements 
    once I get the chance. Thanks, Saaem
    """
    
    # Define expected events that match the email
    expected_events = [
        {
            "Event name": "Meeting with Emi",
            "Date": "2025-05-04",  # Next day from test run
            "Time": "1pm",
            "Property Type": "furnished",
            "Agent Name": "Jennifer",
            "Location": "shuttle bus stop"
        }
    ]
    
    # Define expected summary
    expected_summary = "Meeting with Emi at shuttle bus stop tomorrow at 1pm to discuss a furnished house. Saaem will also provide a signed copy of the lease agreements."
    
    # Run a single test with this email and expected values
    report = run_single_test(
        model, 
        email_body, 
        expected_events=expected_events,
        expected_sentiment="positive",
        expected_summary=expected_summary
    )
    
    # Print the report
    print(report)
    
    # Save the report to a file
    os.makedirs("test_reports", exist_ok=True)
    timestamp = os.path.join("test_reports", f"single_test_{os.path.basename(os.getcwd())}_{os.getpid()}.txt")
    with open(timestamp, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {timestamp}")

if __name__ == "__main__":
    main() 