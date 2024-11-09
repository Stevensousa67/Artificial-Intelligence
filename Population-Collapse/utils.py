''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/07/2024
    Description: This is a utils file for the Population Collapse project that contains shared functions and constants.
    version: 1.0'''

import pandas as pd

FILE_PATH = './Population-Collapse/resources/Population History.xlsx'

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
    selection = input("\nPlease enter your choice: ")
    while selection not in options:
        print("Invalid choice. Please choose from the list.")
        selection = input("\nPlease enter your choice: ")
    return selection