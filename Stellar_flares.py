# Read stars from M dwarf stars catalogue and then calculate flare frequency Distribution(dN/dE) and Flare Duty cycle(time% in flares) and Flare rate(#/hr) of each star.

import os
import gzip
import shutil
from astropy.io import ascii, fits
import pandas as pd
import time
import numpy as np
from astropy.table import Table

# Function to extract .gz file
def extract_gz(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Create the folder if it doesn't exist
os.makedirs("Dwarfs_catalog", exist_ok=True)
    
def parse_row_table1(row):
    """
    Parses a row from table1.dat and extracts the following columns:

    PMI Number, HIP Number, TYC Number, CNS3 Number, Right Ascension, Declination, Proper Motion, Proper Motion in RA, Proper Motion in Dec, Parallax, Source

        # Column 1: PMI Number (elements 0-16)
        col1 = row[0:16].strip()
        # Column 2: HIP Number (elements 16-24)
        col2 = row[16:24].strip()
        # Column 3: TYC Number (elements 24-36)
        col3 = row[24:36].strip()
        # Column 4: CNS3 Number (elements 36-47)
        col4 = row[36:47].strip()
        # Column 5: Right Ascension (α) (elements 47-65)
        col5 = row[47:65].strip()
        # Column 6: Declination (δ) (elements 48-55)
        col6 = row[65:77].strip()
        # Column 7: Proper Motion (μ) (elements 56-59)
        col7 = row[77:84].strip()
        # Column 8: Proper Motion in RA (μα) (elements 56-59)
        col8 = row[84:91].strip()
        # Column 9: Proper Motion in Dec (μδ) (elements 60-63)
        col9 = row[91:98].strip()
        # Column 10: Parallax (π) +- (elements 64-67)
        col10 = row[98:113].strip()
        # Column 11: Source of Parallax (elements 68-71)
        col11 = row[113:119].strip()

        return (col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11) 
    """
    # Return all parsed columns as a tuple  
    return (row[0:16].strip(), 
            row[16:24].strip(), 
            row[24:36].strip(), 
            row[36:47].strip(), 
            row[47:65].strip(), 
            row[65:77].strip(), 
            row[77:84].strip(), 
            row[84:91].strip(), 
            row[91:98].strip(), 
            row[98:113].strip(), 
            row[113:119].strip())

def parse_row_table2(row):

    """
    Parses a row from table2.dat and extracts the following columns:
    
    SUPERBLINK, X-Ray, FUV, NUV, BT, VT, BJ, RF, IN, J, H, Ks, V, V_J, Paraalax phot, Subtype

        # Column 1: PMI Number (elements 0-16)
        col1 = row[0:16].strip()
        # Column 2: xray counts (elements 16-24)
        col2 = row[16:24].strip()
        # Column 3: FUV (elements 16-24)
        col3 = row[24:30].strip()
        # Column 4: NUV (elements 30-36)
        col4 = row[30:36].strip()
        # Column 5: BT (elements 36-42)
        col5 = row[36:42].strip()
        # Column 6: VT (elements 42-48)
        col6 = row[42:48].strip()
        # Column 7: Bj (elements 48-53)
        col7 = row[48:53].strip()
        # Column 8: RF (elements 53-58)
        col8 = row[53:58].strip()
        # Column 9: IN (elements 58-63)
        col9 = row[58:63].strip()
        # Column 10: J (elements 63-69)
        col10 = row[63:69].strip()
        # Column 11: H (elements 69-75)
        col11 = row[69:75].strip()
        # Column 12: Ks (elements 75-81)
        col12 = row[75:81].strip()
        # Column 13: V (elements 81-87)
        col13 = row[81:87].strip()
        # Column 14: V-J (elements 87-93)
        col14 = row[87:93].strip()
        # Column 15: Parallax phot (elements 93-108)
        col15 = row[93:108].strip()
        # Column 16: Subtype (elements 108-113)
        col16 = row[108:113].strip()

        return (col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16)
    """
    return (row[0:16].strip(), 
            row[16:24].strip(), 
            row[24:30].strip(), 
            row[30:36].strip(), 
            row[36:42].strip(), 
            row[42:48].strip(), 
            row[48:53].strip(), 
            row[53:58].strip(), 
            row[58:63].strip(), 
            row[63:69].strip(), 
            row[69:75].strip(), 
            row[75:81].strip(), 
            row[81:87].strip(), 
            row[87:93].strip(), 
            row[93:108].strip(), 
            row[108:113].strip())

# Check if the FITS files already exist
if not os.path.exists(f"Dwarfs_catalog{os.sep}table2.fits"):
    # Download the files directly into the specified folder
    os.system("wget -P Dwarfs_catalog -nc ftp://cdsarc.cds.unistra.fr/pub/cats/J/AJ/142/138/table1.dat.gz")
    os.system("wget -P Dwarfs_catalog -nc ftp://cdsarc.cds.unistra.fr/pub/cats/J/AJ/142/138/table2.dat.gz")
    
    # Extract .gz files
    extract_gz(f"Dwarfs_catalog{os.sep}table1.dat.gz",f"Dwarfs_catalog{os.sep}table1.dat")
    extract_gz(f"Dwarfs_catalog{os.sep}table2.dat.gz",f"Dwarfs_catalog{os.sep}table2.dat")
    print("Extraction complete!")

    # Read and convert the extracted table

    start_time = time.time()

    # Prepare a list to store the structured data
    parsed_data = []
    parsed_data2= []

    # Read the table1.dat file
    with open(f"Dwarfs_catalog{os.sep}table1.dat", "r") as file:
        rows = file.readlines()

    # Parse each row and extract the columns
    for row in rows:
        row = row.strip()
        parsed_row = parse_row_table1(row)
        # print(parsed_row)
        parsed_data.append(parsed_row)

    # Read the table2.dat file
    with open(f"Dwarfs_catalog{os.sep}table2.dat", "r") as file:
        rows = file.readlines()

    # Parse each row and extract the columns
    for row in rows:
        row = row.strip()
        # Parse the row and extract the required columns
        parsed_row = parse_row_table2(row)
        parsed_data2.append(parsed_row)


    # Convert the parsed data to a pandas DataFrame
    df1= pd.DataFrame(parsed_data, columns=["PMI Number", "HIP Number", "TYC Number", "CNS3 Number", "Right Ascension", "Declination", "Proper Motion", "Proper Motion in RA", "Proper Motion in Dec", "Parallax", "Source"])
    df2 = pd.DataFrame(parsed_data2, columns=["PMI Number", "X-Ray", "FUV", "NUV", "BT", "VT", "BJ", "RF", "IN", "J", "H", "Ks", "V", "V_J", "Parallax phot", "Subtype"])


    # Function to decode byte-strings if needed
    def decode_data(data):
        return [item.decode('utf-8') if isinstance(item, bytes) else item for item in data]

    # Decode the parsed rows before storing them
    parsed_data = [decode_data(row) for row in parsed_data]
    parsed_data2 = [decode_data(row) for row in parsed_data2]

    # save the data asfits file
    table1 = Table(rows=parsed_data, names=("PMI Number", "HIP Number", "TYC Number", "CNS3 Number", "Right Ascension", "Declination", "Proper Motion", "Proper Motion in RA", "Proper Motion in Dec", "Parallax", "Source"))
    table2 = Table(rows=parsed_data2, names=("PMI Number", "X-Ray", "FUV", "NUV", "BT", "VT", "BJ", "RF", "IN", "J", "H", "Ks", "V", "V_J", "Parallax phot", "Subtype"))
    
    print("Table1")
    print(table1)
    print("\nTable2")
    print(table2)

    # Save to FITS
    table1.write(f"Dwarfs_catalog{os.sep}table1.fits", format='fits', overwrite=True)
    table2.write(f"Dwarfs_catalog{os.sep}table2.fits", format='fits', overwrite=True)

    print("Time taken to read and convert the files: ", time.time() - start_time)

    print(f"FITS file saved in Dwarfs_catalog{os.sep}\n")
    del parsed_data, parsed_data2, table1, table2, rows
else:
    print("FITS files already exist. No need to download or convert.")

    table1 = Table.read(f"Dwarfs_catalog{os.sep}table1.fits")
    table2 = Table.read(f"Dwarfs_catalog{os.sep}table2.fits")

    # Convert the Astropy Table to a Pandas DataFrame, while decoding any byte columns
    df1 = table1.to_pandas().map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df2 = table2.to_pandas().map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Display the first 10 rows of the DataFrame
print(df1.head(10),"\n")
print(df2.head(10))


