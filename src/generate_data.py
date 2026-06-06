import json
import os
import random
import numpy as np
from faker import Faker
import pandas as pd

fake = Faker()

# --- Configuration ---
NUM_REPORTS = 2500
CATEGORIES = ["Technical", "Catering", "Passenger", "Safety", "Medical"]

WIDE_BODIES = ["A330", "A350", "B777", "B787"]
NARROW_BODIES = ["A321", "B737"]
ALL_FLEET = WIDE_BODIES + NARROW_BODIES

ROUTE_METADATA = {
    "IST-JFK": (655, WIDE_BODIES),
    "IST-NRT": (675, WIDE_BODIES),
    "IST-KIX": (655, WIDE_BODIES),
    "IST-GRU": (820, WIDE_BODIES),
    "IST-SFO": (835, WIDE_BODIES),
    "IST-BKK": (535, WIDE_BODIES),
    "IST-MRU": (570, WIDE_BODIES),
    "IST-LHR": (245, ALL_FLEET),
    "IST-CDG": (225, ALL_FLEET),
    "IST-BER": (175, ALL_FLEET),
    "IST-DXB": (265, ALL_FLEET),
    "IST-FRA": (190, ALL_FLEET),
}


filepath = os.path.join(os.path.dirname(__file__), "templates.json")


def load_templates(filepath=filepath):
    """Load report templates and observation suffixes from external JSON."""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None


def generate_single_report(routes, templates, observations):
    """Generate a single operational report entry without append state."""
    route = random.choice(routes)
    base_duration, allowed_aircraft = ROUTE_METADATA[route]

    category = random.choice(CATEGORIES)
    duration = base_duration + random.randint(-15, 30)

    # Probabilistic urgency assignment based on flight duration
    urgency_weights = [0.2, 0.5, 0.3] if duration > 500 else [0.6, 0.3, 0.1]

    # Format dynamic fields in templates
    category_templates = templates.get(category, [])
    if not category_templates:
        return None

    raw_template = random.choice(category_templates)
    seat_no = f"{random.randint(1, 40)}{random.choice('ABCDEF')}"
    meal_count = random.randint(1, 5)

    report_main = raw_template.format(seat_no=seat_no, meal_count=meal_count)
    observation = random.choice(observations)

    return {
        "report_id": f"AR-{fake.bothify(text='??-####').upper()}",
        "date": fake.date_between(start_date="-1y", end_date="today"),
        "flight_no": f"TK{random.randint(1, 2500)}",
        "aircraft_type": random.choice(allowed_aircraft),
        "route": route,
        "duration_min": duration,
        "crew_seniority_years": random.randint(1, 22),
        "category": category,
        "report_content": f"{report_main} {observation}",
        "urgency": np.random.choice(["Low", "Medium", "High"], p=urgency_weights),
        "delay_caused_min": (
            random.randint(10, 40)
            if category == "Technical" and random.random() > 0.8
            else 0
        ),
    }


def create_synthetic_data(num_rows):
    """Create full dataframe using list comprehension for better performance."""
    external_data = load_templates()
    templates = external_data["ReportTemplates"]
    observations = external_data["ObservationSuffixes"]
    routes = list(ROUTE_METADATA.keys())

    # Use list comprehension to populate the dataset efficiently
    data_list = [
        generate_single_report(routes, templates, observations) for _ in range(num_rows)
    ]

    return pd.DataFrame(data_list)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = create_synthetic_data(NUM_REPORTS)
    df.to_csv("data/cabin_reports.csv", index=False)
    print(f"Success: Generated {NUM_REPORTS} reports from external JSON templates.")
