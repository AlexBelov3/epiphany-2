import subprocess
import time
import os

# List of arguments
args = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# Duration to let each process run (in seconds)
duration = 10 #* 60  # 10 minutes

for arg in args:
    # Define the output file name
    output_file = f"{arg}_output.txt"

    # Start the process
    process = subprocess.Popen(['python3', 'adversarial.py', arg], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Let the process run for the specified duration
    time.sleep(duration)

    # Terminate the process
    process.terminate()

    # Capture the output and error
    try:
        stdout, stderr = process.communicate(timeout=10)  # Give it a moment to clean up
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()

    # Write the output to the file
    with open(output_file, 'wb') as f:
        f.write(stdout)
        if stderr:
            f.write(b'\n--- STDERR ---\n')
            f.write(stderr)

    print(f"Finished running adversarial.py {arg}, output saved to {output_file}")

print("All processes finished.")
