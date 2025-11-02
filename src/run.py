import subprocess
# List of Python file to run
files = ["CA.py", "USS.py", "Visualization.py"]
# Run each file
for script in files:
    subprocess.run(["python", script])
