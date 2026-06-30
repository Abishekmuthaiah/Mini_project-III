import kagglehub
import os
import shutil
import sys

def fetch_mohammed_dataset():
    handle = "mohammedabdeldayem/the-fake-or-real-dataset"
    print(f"Downloading {handle}...")
    try:
        path = kagglehub.dataset_download(handle)
        print("Dataset downloaded to:", path)
    except Exception as e:
        print("Failed to download or extract:", e)
        sys.exit(1)
        
    # Explore dataset
    real_target = "dataset/real"
    fake_target = "dataset/fake"
    os.makedirs(real_target, exist_ok=True)
    os.makedirs(fake_target, exist_ok=True)
    
    max_count = 200 # prevent extremely slow training on CPU
    r_count, f_count = 0, 0
    
    for root, dirs, files in os.walk(path):
        root_lower = root.lower()
        
        # Decide based on path name
        is_real = 'real' in root_lower or 'human' in root_lower
        is_fake = 'fake' in root_lower or 'ai' in root_lower
        
        for f in files:
            if not f.endswith('.wav') and not f.endswith('.mp3'):
                continue
            
            f_lower = f.lower()
            if is_real or 'real' in f_lower:
                if r_count < max_count:
                    shutil.copy2(os.path.join(root, f), os.path.join(real_target, f"human_{r_count}.wav"))
                    r_count += 1
            elif is_fake or 'fake' in f_lower:
                if f_count < max_count:
                    shutil.copy2(os.path.join(root, f), os.path.join(fake_target, f"ai_{f_count}.wav"))
                    f_count += 1
                    
            if r_count >= max_count and f_count >= max_count:
                print(f"Copied {r_count} real and {f_count} fake files!")
                sys.exit(0)
    
    print(f"Finished copying whatever we could find. Real: {r_count}, Fake: {f_count}")

if __name__ == "__main__":
    fetch_mohammed_dataset()
