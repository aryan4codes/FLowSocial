# import fiftyone as fo

# # Load 5,000 images with animal-related tags
# dataset = fo.zoo.load_zoo_dataset(
#     "open-images-v7",
#     split="train",
#     label_types=["detections", "classifications"],
#     classes=["Cat", "Dog", "Bird", "Horse"],
#     max_samples=5000,
# )
# dataset.export(
#     export_dir="data/raw/",
#     dataset_type=fo.types.ImageDirectory,
# )


import sqlite3

def download_sample_text_data(db_path="project/data/sample.db"):
    """
    Creates (or connects to) a SQLite database and inserts sample text data.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT
        )
    """)
    # Insert sample texts
    sample_texts = [
        ("News Article", "Breaking news: Federated learning is revolutionizing data privacy."),
        ("Blog Post", "Today we talk about text recommendation systems and clustering."),
        ("Research Paper", "This study explores multimodal federated learning in social media."),
    ]
    c.executemany("INSERT INTO texts (title, content) VALUES (?, ?)", sample_texts)
    conn.commit()
    conn.close()
    print("Sample text data downloaded and inserted into the database.")

if __name__ == "__main__":
    download_sample_text_data()