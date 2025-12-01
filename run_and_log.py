import subprocess
import sys
from datetime import datetime

# Run main.py and capture output
result = subprocess.run([sys.executable, 'main.py'], 
                       capture_output=True, 
                       text=True,
                       cwd='.')

# Print to console
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)

# Save to file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'experiments/training_log_{timestamp}.txt'

with open(log_filename, 'w') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)

print(f"\n\nLog saved to {log_filename}")
print(f"Exit code: {result.returncode}")

