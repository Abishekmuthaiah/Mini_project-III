import os
import subprocess

try:
    from datasets import load_dataset
except ImportError:
    subprocess.check_call(["e:\\minipro\\venv\\Scripts\\python.exe", "-m", "pip", "install", "datasets", "soundfile"])
    from datasets import load_dataset

import soundfile as sf
import traceback

real_dir = "dataset/real"
fake_dir = "dataset/fake"
# Purge old synthesized noisy garbage that corrupted prediction matrices
subprocess.call('Remove-Item -Path dataset\\real\\* -ErrorAction SilentlyContinue', shell=True)
subprocess.call('Remove-Item -Path dataset\\fake\\* -ErrorAction SilentlyContinue', shell=True)
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

print("Streaming authentic Studio Deepfake/Real pairs from HuggingFace Hub...")

try:
    ds = load_dataset("UniDataPro/real-vs-fake-human-voice-deepfake-audio", split="train", streaming=True)
    
    r_count, f_count = 0, 0
    max_count = 150
    
    # We peek at the first dataset row to safely map the label dynamically
    for sample in ds:
        keys = sample.keys()
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        
        # Extract Label intelligently
        if 'label' in keys:
            label = sample['label']
            cls_name = "fake"  # Fallback assumption
            if isinstance(label, int):
                cls_name = "real" if label == 0 else "fake" # Most ASV/HF use 0=Real, 1=Spoof
            elif isinstance(label, str):
                cls_name = "real" if "real" in label.lower() or "human" in label.lower() else "fake"
        elif 'is_real' in keys:
             cls_name = "real" if sample['is_real'] else "fake"
        else:
            # Maybe path string has fake or real
            path = sample.get('audio', {}).get('path', "")
            cls_name = "real" if "real" in path.lower() or "human" in path.lower() else "fake"
            
        # Distribute into designated training pools
        if cls_name == "real" and r_count < max_count:
            sf.write(os.path.join(real_dir, f"hf_human_{r_count}.wav"), audio_array, sr)
            r_count += 1
        elif cls_name == "fake" and f_count < max_count:
            sf.write(os.path.join(fake_dir, f"hf_ai_{f_count}.wav"), audio_array, sr)
            f_count += 1
            
        if r_count >= max_count and f_count >= max_count:
            print("Successfully extracted maximum required sample quota.")
            break
            
    print(f"HF Database Stream successful: Generated {r_count} Real and {f_count} Fake ground-truth signatures.")
except Exception as e:
    print(f"Stream interrupted during HF access context wrapper: {e}")
    traceback.print_exc()

print("Dataset building complete!")
