import subprocess
import sys
import os

def run_command(command):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}':")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("Setting up DeepLens project...")
    
    # Upgrade pip first
    print("\n1. Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install requirements one by one to handle dependencies better
    print("\n2. Installing core dependencies...")
    core_deps = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.2",
        "spacy==3.7.2",
        "textblob>=0.15.3",
        "nltk>=3.6.3",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "beautifulsoup4>=4.9.3",
        "tweepy>=4.4.0",
        "google-api-python-client>=2.0.0",
        "gnews==0.4.2"
    ]
    
    for dep in core_deps:
        print(f"Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    print("\n3. Installing ML dependencies...")
    ml_deps = [
        "transformers>=4.11.0",
        "torch>=1.9.0"
    ]
    
    for dep in ml_deps:
        print(f"Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    print("\n4. Installing frontend dependencies...")
    frontend_deps = [
        "gradio>=2.0.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0"
    ]
    
    for dep in frontend_deps:
        print(f"Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    # Install spacy model
    print("\n5. Installing spaCy model...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm")
    
    print("\nSetup completed!")
    print("\nYou can now:")
    print("1. Run the pipeline:")
    print("   python run_pipeline.py --query 'your query' --platforms gnews youtube twitter")
    print("\n2. Start the dashboard:")
    print("   python app/dashboard.py")

if __name__ == "__main__":
    main()