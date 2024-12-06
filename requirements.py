import os
import subprocess

# Filepath to requirements.txt
requirements_file = "requirements.txt"

# Function to install dependencies from requirements.txt
def install_requirements():
    if os.path.exists(requirements_file):
        try:
            print(f"Installing dependencies from {requirements_file}...")
            subprocess.check_call(["pip", "install", "-r", requirements_file])
            print("All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while installing dependencies: {e}")
    else:
        print(f"Error: {requirements_file} not found. Please create the file and add your dependencies.")

# Execute the installation
if __name__ == "__main__":
    install_requirements()