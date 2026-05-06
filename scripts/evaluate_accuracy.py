import os
import requests
import argparse
import time
import json

API_URL = "http://localhost:8000"

def evaluate_accuracy(test_dir: str, ground_truth_file: str):
    """
    Evaluates the accuracy of the audio identification system.
    ground_truth_file should be a JSON mapping filename -> expected_song_id or expected_title
    """
    if not os.path.isdir(test_dir):
        print(f"Error: {test_dir} is not a valid directory.")
        return

    try:
        with open(ground_truth_file, "r") as f:
            ground_truth = json.load(f)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    print(f"Starting Evaluation on {test_dir}...")
    
    total = 0
    correct = 0
    false_positives = 0
    false_negatives = 0
    
    start_time = time.time()
    
    for filename, expected_match in ground_truth.items():
        file_path = os.path.join(test_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Test file {filename} not found.")
            continue
            
        total += 1
        print(f"Testing {filename} (Expected: {expected_match})...")
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}
                response = requests.post(f"{API_URL}/identify/", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("match"):
                        # Check if matched title or id matches expected
                        # In a real scenario, use ID. Here we might use title for simplicity.
                        if result.get("title") == expected_match or result.get("song_id") == expected_match:
                            correct += 1
                            print("  -> CORRECT")
                        else:
                            false_positives += 1
                            print(f"  -> FALSE POSITIVE (Got: {result.get('title')})")
                    else:
                        if expected_match == "None" or expected_match is None:
                            correct += 1 # Correctly identified as no match
                            print("  -> CORRECT (No Match)")
                        else:
                            false_negatives += 1
                            print("  -> FALSE NEGATIVE (No Match Found)")
                else:
                    print(f"  -> ERROR ({response.status_code})")
        except Exception as e:
            print(f"  -> ERROR ({e})")
            
    elapsed = time.time() - start_time
    
    print("\n--- Evaluation Results ---")
    print(f"Total Queries: {total}")
    if total > 0:
        print(f"Correct: {correct} ({(correct/total)*100:.1f}%)")
        print(f"False Positives: {false_positives} ({(false_positives/total)*100:.1f}%)")
        print(f"False Negatives: {false_negatives} ({(false_negatives/total)*100:.1f}%)")
    print(f"Time Taken: {elapsed:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate identification accuracy.")
    parser.add_argument("directory", help="Path to the directory containing test audio snippets.")
    parser.add_argument("ground_truth", help="Path to JSON file with expected matches.")
    args = parser.parse_args()
    
    evaluate_accuracy(args.directory, args.ground_truth)
