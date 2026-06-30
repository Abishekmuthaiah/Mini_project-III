import kagglehub
import os

def check_dataset():
    path = kagglehub.dataset_download("kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices")
    print(f"DATASET_PATH={path}")
    for root, dirs, files in os.walk(path):
        print(f"DIR: {root} contains {len(files)} files")
        # Print a few file names
        if len(files) > 0:
            for f in files[:2]:
                print(f"  - {f}")

if __name__ == '__main__':
    check_dataset()
