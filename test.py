import pandas as pd
import random
from datetime import datetime, timedelta

# Feedback samples
positive_feedbacks = [
    "Great event!", "Loved it!", "Amazing job.", "Very happy.", "Excellent session!",
    "Fantastic work.", "Superb!", "Well organized.", "Impressive effort.", "Enjoyed it a lot."
]

negative_feedbacks = [
    "Very disappointed.", "Terrible experience.", "Poorly organized.", "Did not like it.",
    "Waste of time.", "Could be much better.", "Unsatisfactory.", "Not happy.", "Bad event.", "Regret attending."
]

neutral_feedbacks = [
    "It was okay.", "Decent.", "Fine overall.", "Nothing special.", "Mediocre.",
    "So-so.", "Average experience.", "Neutral thoughts.", "It happened.", "Acceptable."
]

# Distribution
total_rows = 1000
num_negative = int(total_rows * 0.80)
num_positive = int(total_rows * 0.10)
num_neutral = total_rows - num_negative - num_positive

# Timestamp generator
def random_timestamp():
    now = datetime.now()
    return (now - timedelta(minutes=random.randint(0, 100000))).strftime("%Y-%m-%d %H:%M:%S")

# Data generation
data = []
for _ in range(num_negative):
    data.append([random_timestamp(), random.choice(negative_feedbacks), random.randint(0, 3), random.randint(1, 10)])
for _ in range(num_positive):
    data.append([random_timestamp(), random.choice(positive_feedbacks), random.randint(1, 10), random.randint(0, 3)])
for _ in range(num_neutral):
    data.append([random_timestamp(), random.choice(neutral_feedbacks), random.randint(0, 3), random.randint(0, 3)])

random.shuffle(data)

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=["timestamp", "feedback", "upvotes", "downvotes"])
df.to_csv("feedback_dataset_1000_custom.csv", index=False)
print("✅ CSV file created: feedback_dataset_1000_custom.csv")
