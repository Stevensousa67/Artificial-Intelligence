''' Authors: Steven Sousa & Nicolas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 12/02/2024
    version: 1.0
    Description: This file will preprocess source data uploaded via upload.html. Preprocessing includes: 
        1. Removing columns "Country Code" and "Time Code, if they exist"
        2. Removal of all rows of a country if even 1 row contains blank values, represented by '..' in any row or column.
        3. Remove "Data from database: Population estimates and projections" and "Last Updated: " from the end of the data. '''

import pandas as pd

def preprocess_data(file_path):
    '''Preprocess the data from the specified file path.'''
    try:
        # Read the uploaded data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sheet_name='Data')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, sheet_name='Data')
        else:
            return "Unsupported file format"  # Handle other unsupported filetypes

        # Clean column names by removing content inside square brackets
        df.columns = df.columns.str.replace(r'\[.*?\]', '', regex=True).str.strip()

        # Remove columns "Country Code" and "Time Code"
        if 'Country Code' in df.columns:
            df.drop(columns=['Country Code'], inplace=True)
        if 'Time Code' in df.columns:
            df.drop(columns=['Time Code'], inplace=True)

        # Remove any country that contains blank values, represented by '..' in any row or column
        df = df.replace('..', pd.NA)
        df = df.dropna()

        # Remove "Data from database: Population estimates and projections" and "Last Updated: " from the end of the data
        df = df[~df['Country Name'].str.contains('Data from database: Population estimates and projections', na=False)]
        df = df[~df['Country Name'].str.contains('Last Updated: ', na=False)]

        # Save the cleaned data to a new file in .xlsx format and in the same location as the original file.
        cleaned_file_path = file_path.replace('.xlsx', '_cleaned.xlsx')
        df.to_excel(cleaned_file_path, index=False)

        return cleaned_file_path
    except Exception as e:
        return f"Error processing file: {e}"