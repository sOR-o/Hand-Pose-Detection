from datetime import datetime
import time

def clear_file_content(filename):
    try:
        # Open the file in write mode, which truncates the file
        with open(filename, 'w') as file:
            file.write('')

    except Exception as e:
        pass

# Function to check the output file for new strings
def read_and_delete_first_line(filename):
    try:
        # Read the file
        with open(filename, 'r+') as file:
            lines = file.readlines()

            # If the file is empty, return None
            if not lines:
                return None

            # Get the first line
            first_line = lines[0]

            # Delete the first line from the file
            del lines[0]

            # Move the file cursor to the beginning
            file.seek(0)

            # Write remaining lines back to the file
            file.writelines(lines)

            # Truncate the file to remove the remaining content
            file.truncate()
            return first_line.strip()  # Remove newline character if present
        
    except FileNotFoundError:
        return None
    
def add_string_to_file(string, filename):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_string = f"{timestamp} : {string}"
    
    # Check if the timestamp already exists in the file
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith(timestamp):
                return  # If timestamp exists, do not add the string
    
    # If timestamp doesn't exist, append the string to the file
    with open(filename, 'a') as file:
        file.write(formatted_string + '\n')