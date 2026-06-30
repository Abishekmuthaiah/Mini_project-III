import zipfile
import os

archive_path = r"C:\Users\AbishekMuthaiah SK\.cache\kagglehub\datasets\mohammedabdeldayem\the-fake-or-real-dataset\2.archive"
real_dir = "dataset/real"
fake_dir = "dataset/fake"
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

r_count, f_count = 0, 0
max_count = 500

print("Parsing 3.1GB Kaggle archive natively...")
try:
    with zipfile.ZipFile(archive_path, 'r') as z:
        for name in z.namelist():
            if not name.endswith('.wav'): continue
            
            name_lower = name.lower()
            if 'real' in name_lower or 'human' in name_lower:
                if r_count < max_count:
                    target_path = os.path.join(real_dir, f"human_{r_count}.wav")
                    if not os.path.exists(target_path):
                        with open(target_path, 'wb') as f:
                            f.write(z.read(name))
                    r_count += 1
            elif 'fake' in name_lower or 'ai' in name_lower:
                if f_count < max_count:
                    target_path = os.path.join(fake_dir, f"ai_{f_count}.wav")
                    if not os.path.exists(target_path):
                        with open(target_path, 'wb') as f:
                            f.write(z.read(name))
                    f_count += 1
                    
            if r_count >= max_count and f_count >= max_count:
                print("Hit extraction limits.")
                break
                
    print(f"Extraction successful: {r_count} Real and {f_count} Fake audio samples transferred.")
except Exception as e:
    print("Archive parsing failed:", e)
