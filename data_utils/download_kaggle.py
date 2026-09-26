import os
import argparse
import subprocess
from pathlib import Path

def download_kaggle_dataset(output_dir):
    """
    Downloads the 'Indian Paintings Dataset' from Kaggle using the official API.
    Requires `~/.kaggle/kaggle.json` to be configured.
    """
    dataset_id = "ajg117/indian-paintings-dataset"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Downloading dataset '{dataset_id}' from Kaggle...")
    
    # Check if kaggle command is available
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[!] Error: 'kaggle' CLI is not installed.")
        print("    Please run: pip install kaggle")
        print("    And ensure your kaggle.json API key is in ~/.kaggle/")
        return
    except Exception as e:
        print(f"[!] Warning: kaggle CLI issue: {e}")
        
    try:
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", dataset_id, 
            "-p", str(out_path)
        ], check=True)
        
        zip_file = out_path / "indian-paintings-dataset.zip"
        if zip_file.exists():
            print(f"[*] Extracting {zip_file}...")
            subprocess.run(["unzip", "-q", str(zip_file), "-d", str(out_path)], check=True)
            zip_file.unlink() # Cleanup zip
            print("[✓] Kaggle dataset successfully downloaded and extracted!")
        else:
            print("[!] Download failed or zip file not found.")
            
    except subprocess.CalledProcessError as e:
        print(f"[!] Kaggle API Error: {e}")
        print("    Ensure your Kaggle API key is configured correctly in ~/.kaggle/kaggle.json")

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle Indian Paintings Dataset")
    parser.add_argument("--output_dir", type=str, default="../data/full", help="Directory to save the dataset")
    args = parser.parse_args()
    
    out_dir = Path(__file__).resolve().parent / args.output_dir
    download_kaggle_dataset(out_dir)

if __name__ == "__main__":
    main()
