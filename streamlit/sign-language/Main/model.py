from datetime import datetime
import time
import os
import llama

def process_string(input_string):
    return llama.translator.translate(input_string)

def main():
    input_filename = 'streamlit/sign-language/Main/broken.txt'
    output_filename = 'streamlit/sign-language/Main/correct.txt'

    processed_timestamps = set()

    while True:
        if os.path.exists(input_filename):
            with open(input_filename, 'r') as input_file:
                lines = input_file.readlines()

            # Check if there are new lines in the input file
            if len(lines) > 0:
                with open(output_filename, 'a') as output_file:
                    for line in lines:
                        parts = line.strip().split(' : ')
                        if len(parts) == 2:
                            timestamp, input_string = parts
                            
                            # Check if the timestamp has already been processed
                            if timestamp in processed_timestamps:
                                continue  # Skip processing if already processed
                            
                            output_string = process_string(input_string)
                            formatted_output = f"{timestamp} : {output_string}\n"
                            output_file.write(formatted_output)
                            
                            # Add the timestamp to the set of processed timestamps
                            processed_timestamps.add(timestamp)
        
        # Sleep for a while before checking again
        time.sleep(0.25)  # Adjust the time interval as per your requirement

if __name__ == "__main__":
    main()
