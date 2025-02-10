# import pandas as pd
# import os
# import random

# # Read the CSV files
# detections_df = pd.read_csv('labels/detections.csv')
# classifications_df = pd.read_csv('labels/classifications.csv')
# metadata_df = pd.read_csv('./metadata/image_ids.csv')
# classes_df = pd.read_csv('./metadata/classes.csv', header=None, names=['LabelName', 'DisplayName'])

# # Create a dictionary for label name mapping
# label_map = dict(zip(classes_df['LabelName'], classes_df['DisplayName']))

# def get_image_info(image_id):
#     """Get complete information for a single image"""
#     # Get metadata
#     image_metadata = metadata_df[metadata_df['ImageID'] == image_id].iloc[0]
    
#     # Get detections for the first 10 images only
#     image_detections = detections_df[detections_df['ImageID'] == image_id]
#     detected_objects = []
#     for _, detection in image_detections.iterrows():
#         label = label_map.get(detection['LabelName'], detection['LabelName'])
#         detected_objects.append({
#             'object': label,
#             'confidence': detection['Confidence'],
#             'bbox': {
#                 'xmin': detection['XMin'],
#                 'xmax': detection['XMax'],
#                 'ymin': detection['YMin'],
#                 'ymax': detection['YMax']
#             }
#         })

#     # Get classifications
#     image_classifications = classifications_df[classifications_df['ImageID'] == image_id]
#     classifications = []
#     for _, classification in image_classifications.iterrows():
#         label = label_map.get(classification['LabelName'], classification['LabelName'])
#         classifications.append({
#             'label': label,
#             'confidence': classification['Confidence']
#         })
    
#     return {
#         'image_id': image_id,
#         'metadata': {
#             'original_url': image_metadata['OriginalURL'],
#             'author': image_metadata['Author'],
#             'title': image_metadata['Title'],
#             'license': image_metadata['License']
#         },
#         'detections': detected_objects,
#         'classifications': classifications
#     }

# # Get 5 random image IDs
# random_images = random.sample(list(metadata_df['ImageID']), 5)

# # Print information for random images
# for image_id in random_images:
#     info = get_image_info(image_id)
#     print("\n=== Image Information ===")
#     print(f"ID: {info['image_id']}")
#     print(f"Title: {info['metadata']['title']}")
#     print(f"Author: {info['metadata']['author']}")
    
#     print("\nDetected Objects:")
#     for obj in info['detections']:
#         print(f"- {obj['object']} (confidence: {obj['confidence']:.2f})")
    
#     print("\nClassifications:")
#     for cls in info['classifications']:
#         print(f"- {cls['label']} (confidence: {cls['confidence']:.2f})")
    
#     print("-" * 50)


import sqlite3

def explore_data(db_path="project/data/sample.db"):
    """
    Connects to the SQLite database and prints the available text records.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, title, content FROM texts")
    rows = c.fetchall()
    for row in rows:
        print(f"ID: {row[0]}, Title: {row[1]}, Content: {row[2]}")
    conn.close()

if __name__ == "__main__":
    explore_data()