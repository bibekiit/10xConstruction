"""
Helper script to update workspace name for cracks dataset.
Run this and provide the new workspace name when prompted.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from roboflow import Roboflow
except ImportError:
    print("Roboflow not installed. Please install it first.")
    sys.exit(1)

def find_workspace_for_project(api_key, project_name):
    """Try to find workspace containing a project."""
    rf = Roboflow(api_key=api_key)
    
    # Try common workspace names
    common_workspaces = [
        "10xConstruction",
        "10x-construction", 
        "10x_construction",
        "bibekabantika",  # Common username pattern
    ]
    
    print(f"Searching for project '{project_name}'...")
    for ws_name in common_workspaces:
        try:
            workspace = rf.workspace(ws_name)
            projects = workspace.list_projects()
            for proj in projects:
                if project_name.lower() in proj.lower():
                    print(f"✓ Found project in workspace: {ws_name}")
                    return ws_name
        except:
            continue
    
    print("Could not automatically find workspace.")
    return None

if __name__ == "__main__":
    api_key = os.getenv("ROBOFLOW_API_KEY", "FQQRU5Cbf1JLgKSTaVke")
    
    print("="*60)
    print("Workspace Finder for Cracks Dataset")
    print("="*60)
    
    # Try to find automatically
    workspace = find_workspace_for_project(api_key, "cracks")
    
    if workspace:
        print(f"\nSuggested workspace: {workspace}")
        print("\nTo update the download script, run:")
        print(f"  sed -i '' 's/workspace=\"10xConstruction\"/workspace=\"{workspace}\"/' scripts/download_datasets.py")
    else:
        print("\nPlease provide the new workspace name:")
        new_workspace = input("Workspace name: ").strip()
        if new_workspace:
            print(f"\nTo update the download script, run:")
            print(f"  sed -i '' 's/workspace=\"10xConstruction\"/workspace=\"{new_workspace}\"/' scripts/download_datasets.py")
            print(f"\nOr manually edit scripts/download_datasets.py line 68:")
            print(f"  Change: workspace=\"10xConstruction\"")
            print(f"  To:     workspace=\"{new_workspace}\"")

