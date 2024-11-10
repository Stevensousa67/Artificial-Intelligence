''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/09/2024
    version: 1.1
    Description: This is a utils file for the Population Collapse project that contains shared functions and constants.'''

import pandas as pd

FILE_PATH = './Population-Collapse/resources/Population History.xlsx'   # Steven's file path
# FILE_PATH = r'C:\Users\njp12\Desktop\AI Project Files\Population History.xlsx' # Nick's file path

def load_data(file_path):
    '''Load the data from the specified file path.'''

    try:
        data = pd.read_excel(file_path, sheet_name='Data')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_user_selection(options, prompt):
    '''Get user selection from a list of options with the specified prompt.'''
    
    print(prompt)
    for option in options:
        print(option)
    
    # Determine the type of the options
    option_type = type(options[0])
    
    while True:
        selection = input("\nPlease enter your choice: ")
        
        # Convert the selection to the appropriate type
        try:
            selection = option_type(selection)
        except ValueError:
            print(f"Invalid choice. Please enter a valid {option_type.__name__}.")
            continue
        
        if selection in options:
            return selection
        else:
            print("Invalid choice. Please choose from the list.")