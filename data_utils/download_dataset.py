import os
import argparse
import requests
import json
import time
from pathlib import Path
from urllib.parse import quote

# ------------------------------------------------------------------------------
# Folk Art Dataset Downloader (Wikimedia Commons API)
# ------------------------------------------------------------------------------
# Queries the Wikimedia Commons API for open-access folk art images.
# Ensures all images have proper licenses (Public Domain, CC-BY, CC-BY-SA).
# Saves metadata containing credits and source URLs for reproducible research.

KEYWORDS = [
    "Madhubani painting", 
    "Warli painting", 
    "Pattachitra",
    "Gond painting",
    "Kalamkari painting",
    "Tanjore painting"
]

def fetch_wikimedia_images(keyword, num_images, save_dir):
    """
    Fetches image metadata and URLs from Wikimedia Commons.
    """
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {keyword}",
        "gsrlimit": min(num_images, 50), # Max 50 per request
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    
    downloaded = 0
    metadata_list = []
    
    print(f"[*] Searching for '{keyword}'...")
    
    headers = {
        "User-Agent": "FolkArtRestorationBot/1.0 (https://github.com/givemehat/folk-art-restoration; contact@example.com)"
    }
    
    while downloaded < num_images:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"  [!] API Error: {response.status_code}")
            break
            
        data = response.json()
        if 'query' not in data or 'pages' not in data['query']:
            print("  [-] No more results found.")
            break
            
        pages = data['query']['pages']
        
        for page_id, page_data in pages.items():
            if downloaded >= num_images:
                break
                
            if 'imageinfo' not in page_data:
                continue
                
            info = page_data['imageinfo'][0]
            img_url = info['url']
            ext = img_url.split('.')[-1].lower()
            
            if ext not in ['jpg', 'jpeg', 'png']:
                continue
                
            # Extract metadata
            ext_meta = info.get('extmetadata', {})
            artist = ext_meta.get('Artist', {}).get('value', 'Unknown')
            license = ext_meta.get('LicenseShortName', {}).get('value', 'Unknown')
            credit = ext_meta.get('Credit', {}).get('value', 'Unknown')
            
            # Download image
            try:
                img_data = requests.get(img_url, headers=headers, timeout=10).content
                filename = f"{keyword.replace(' ', '_')}_{downloaded:04d}.{ext}"
                filepath = os.path.join(save_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                    
                metadata_list.append({
                    "filename": filename,
                    "keyword": keyword,
                    "url": img_url,
                    "artist": artist,
                    "license": license,
                    "credit": credit
                })
                downloaded += 1
                if downloaded % 10 == 0:
                    print(f"  [+] Downloaded {downloaded}/{num_images} for '{keyword}'")
                    
            except Exception as e:
                print(f"  [!] Failed to download {img_url}: {e}")
                
            time.sleep(0.1) # Be nice to the API
            
        if 'continue' in data:
            params.update(data['continue'])
        else:
            break
            
    return metadata_list

def main():
    parser = argparse.ArgumentParser(description="Download Folk Art Dataset from Wikimedia Commons")
    parser.add_argument("--output_dir", type=str, default="../data/full", help="Directory to save images")
    parser.add_argument("--images_per_class", type=int, default=100, help="Target number of images per category")
    parser.add_argument("--sample_mode", action="store_true", help="Download only a tiny subset for testing (10 per class)")
    args = parser.parse_args()

    target_num = 10 if args.sample_mode else args.images_per_class
    out_dir = Path(__file__).resolve().parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to: {out_dir}")
    
    all_metadata = []
    
    for kw in KEYWORDS:
        meta = fetch_wikimedia_images(kw, target_num, str(out_dir))
        all_metadata.extend(meta)
        
    # Save metadata JSON
    meta_path = out_dir / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=4)
        
    print(f"\n[✓] Download complete. Total images: {len(all_metadata)}")
    print(f"[✓] Metadata saved to {meta_path}")
    print("NOTE: For the full 1.5GB dataset, we recommend running with --images_per_class 500 or utilizing the Kaggle fallback (see README_DATASET.md).")

if __name__ == "__main__":
    main()
