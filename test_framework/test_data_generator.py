import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class TestDataGenerator:
    def __init__(self):
        self.sentiment_templates = {
            "positive": [
                "Great news! I'm excited to share that {event}",
                "I'm really looking forward to {event}",
                "Thank you for your help with {event}"
            ],
            "negative": [
                "I'm disappointed about {event}",
                "Unfortunately, we need to cancel {event}",
                "I'm concerned about {event}"
            ],
            "neutral": [
                "Regarding {event}",
                "I wanted to inform you about {event}",
                "Please note that {event}"
            ]
        }
        
        self.event_templates = [
            "meeting with {agent} at {location} on {date} at {time}",
            "property viewing at {location} on {date} at {time}",
            "discussion about {property_type} with {agent} on {date}"
        ]
        
        self.locations = ["office", "property site", "cafe", "virtual meeting", "shuttle bus stop"]
        self.agents = ["Jennifer", "Michael", "Sarah", "David", "Emily"]
        self.property_types = ["furnished", "unfurnished", "commercial", "residential", "apartment"]

    def generate_test_email(self, 
                          sentiment: str = None,
                          num_events: int = 1,
                          include_attachments: bool = False) -> Dict[str, Any]:
        """Generate a test email with specified characteristics."""
        if sentiment is None:
            sentiment = random.choice(["positive", "negative", "neutral"])
            
        mail_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        thread_id = f"thread_{random.randint(1000, 9999)}_{random.randint(1000, 9999)}"
        
        events = []
        email_body = ""
        
        # Generate events and build email body
        for _ in range(num_events):
            event_template = random.choice(self.event_templates)
            event_date = (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
            event_time = f"{random.randint(9, 17)}:00"
            
            event_context = {
                "agent": random.choice(self.agents),
                "location": random.choice(self.locations),
                "date": event_date,
                "time": event_time,
                "property_type": random.choice(self.property_types)
            }
            
            event_text = event_template.format(**event_context)
            events.append({
                "Event name": f"Meeting with {event_context['agent']}",
                "Date": event_date,
                "Time": event_time,
                "Property Type": event_context["property_type"],
                "Agent Name": event_context["agent"],
                "Location": event_context["location"]
            })
            
            # Add event to email body
            template = random.choice(self.sentiment_templates[sentiment])
            email_body += template.format(event=event_text) + "\n\n"
        
        summary = " ".join([f"{e['Event name']} at {e['Location']} on {e['Date']} at {e['Time']}." for e in events])
        return {
            "mail_id": mail_id,
            "file_name": ["attachment.pdf"] if include_attachments else [],
            "email": f"test{random.randint(1000, 9999)}@example.com",
            "mail_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "body_type": "html",
            "mail_body": email_body,
            "thread_id": thread_id,
            "mail_summary": summary,
            "sentiment": sentiment,
            "events": events
        }

    def generate_test_dataset(self, 
                            num_samples: int = 100,
                            balanced_sentiments: bool = True) -> List[Dict[str, Any]]:
        """Generate a balanced test dataset."""
        dataset = []
        sentiments = ["positive", "negative", "neutral"]
        
        if balanced_sentiments:
            samples_per_sentiment = num_samples // len(sentiments)
            for sentiment in sentiments:
                for _ in range(samples_per_sentiment):
                    dataset.append(self.generate_test_email(sentiment=sentiment))
        else:
            for _ in range(num_samples):
                dataset.append(self.generate_test_email())
                
        return dataset

    def save_test_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save the test dataset to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2) 