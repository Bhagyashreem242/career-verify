import pandas as pd
import os

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# Load Kaggle dataset
df = pd.read_csv("fake_job_postings.csv")
df.fillna("", inplace=True)

# Combine description-related fields
df["combined_description"] = (
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)

# Map Kaggle columns → your project schema
final_df = pd.DataFrame({
    "job_title": df["title"],
    "company": df["company_profile"],
    "location": df["location"],
    "description": df["combined_description"],
    "label": df["fraudulent"]
})

# Balance dataset (100 real, 100 fake)
real_jobs = final_df[final_df["label"] == 0].sample(100, random_state=42)
fake_jobs = final_df[final_df["label"] == 1].sample(100, random_state=42)

dataset_200 = pd.concat([real_jobs, fake_jobs]).sample(frac=1, random_state=42)

# Save dataset
dataset_200.to_csv("dataset/real_world_jobs_200.csv", index=False)

print("✅ Real-world inspired dataset created (200 rows)")
print(dataset_200.head())
