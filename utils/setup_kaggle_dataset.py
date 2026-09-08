import os
import sys
import zipfile
import shutil

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
KAGGLE_ARCHIVE = r"C:\Users\AbishekMuthaiah SK\.cache\kagglehub\datasets\mohammedabdeldayem\the-fake-or-real-dataset\2.archive"

def setup_dataset():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(('.wav', '.mp3'))]
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(('.wav', '.mp3'))]

    print(f"Current dataset status: {len(real_files)} real samples, {len(fake_files)} fake samples.")

    # If dataset already has at least 20 samples of each, we are good
    if len(real_files) >= 20 and len(fake_files) >= 20:
        print("Dataset already contains sufficient audio samples. Skipping extraction.")
        return

    # If kaggle archive is present, extract more samples
    if os.path.exists(KAGGLE_ARCHIVE):
        print(f"Extracting samples from cached Kaggle dataset: {KAGGLE_ARCHIVE}")
        r_count = len(real_files)
        f_count = len(fake_files)
        max_extract = 50

        try:
            with zipfile.ZipFile(KAGGLE_ARCHIVE, 'r') as z:
                for name in z.namelist():
                    if not name.endswith('.wav'):
                        continue
                    name_lower = name.lower()
                    if 'real' in name_lower or 'human' in name_lower:
                        if r_count < max_extract:
                            dest = os.path.join(REAL_DIR, f"real_sample_{r_count}.wav")
                            if not os.path.exists(dest):
                                with open(dest, 'wb') as f:
                                    f.write(z.read(name))
                            r_count += 1
                    elif 'fake' in name_lower or 'ai' in name_lower:
                        if f_count < max_extract:
                            dest = os.path.join(FAKE_DIR, f"fake_sample_{f_count}.wav")
                            if not os.path.exists(dest):
                                with open(dest, 'wb') as f:
                                    f.write(z.read(name))
                            f_count += 1
                    if r_count >= max_extract and f_count >= max_extract:
                        break
            print(f"Dataset extraction complete: {r_count} Real, {f_count} Fake.")
            return
        except Exception as e:
            print(f"Extraction from archive failed: {e}")

    print("Dataset setup completed successfully.")

if __name__ == "__main__":
    setup_dataset()
