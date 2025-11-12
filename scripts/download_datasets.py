"""
Download datasets from Roboflow for Drywall QA segmentation project.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from roboflow import Roboflow
except ImportError:
    print("Roboflow not installed. Installing...")
    os.system("pip install roboflow")
    from roboflow import Roboflow

def download_dataset(workspace, project, version, dataset_name, output_dir):
    """Download a dataset from Roboflow."""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}")
    
    # Initialize Roboflow
    # Note: You'll need to set your API key as environment variable
    # export ROBOFLOW_API_KEY="your_api_key_here"
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("WARNING: ROBOFLOW_API_KEY not set. Please set it as an environment variable.")
        print("You can get your API key from: https://app.roboflow.com/")
        print("\nTrying to download without API key (may require manual download)...")
        return False
    
    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        
        # Try specified version first, then try to get latest
        try:
            dataset = project_obj.version(version).download("yolov8", location=output_dir)
        except:
            # If version doesn't exist, try to get the latest version
            print(f"  Version {version} not found, trying to get latest version...")
            versions = project_obj.list_versions()
            if versions:
                latest_version = versions[0] if isinstance(versions, list) else versions
                print(f"  Using version: {latest_version}")
                dataset = project_obj.version(latest_version).download("yolov8", location=output_dir)
            else:
                raise Exception("No versions found")
        
        print(f"✓ Successfully downloaded {dataset_name} to {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {str(e)}")
        print(f"\nPlease download manually from:")
        if dataset_name == "Drywall-Join-Detect":
            print("https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect")
        else:
            print(f"https://universe.roboflow.com/{workspace}/{project}")
        return False

def main():
    """Main function to download both datasets."""
    base_dir = Path(__file__).parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset 1: Drywall-Join-Detect
    dataset1_dir = raw_data_dir / "drywall-join-detect"
    download_dataset(
        workspace="objectdetect-pu6rn",
        project="drywall-join-detect",
        version=1,  # Update version number if needed
        dataset_name="Drywall-Join-Detect",
        output_dir=str(dataset1_dir)
    )
    
    # Dataset 2: Cracks
    dataset2_dir = raw_data_dir / "cracks"
    download_dataset(
        workspace="test-eswkr",
        project="cracks-3ii36-aocvl",
        version=1,  # Update version number if needed
        dataset_name="Cracks",
        output_dir=str(dataset2_dir)
    )
    
    print("\n" + "="*60)
    print("Dataset download process completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify datasets are in data/raw/")
    print("2. Run preprocessing script: python scripts/preprocess_data.py")

if __name__ == "__main__":
    main()

