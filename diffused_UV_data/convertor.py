import pandas as pd
import time

# Step 1: Read CSV file
# CSV
start_time = time.time()
df = pd.read_csv("diffused_UV_data\RA_sorted_flux_1500.csv",header=None)
df.columns = ['ra', 'dec', 'flux']
csv_read_time = time.time() - start_time

# Step 2: Save DataFrame as Parquet and Feather files
df.to_parquet("diffused_UV_data\RA_sorted_flux_1500.parquet")
df.to_feather("diffused_UV_data\RA_sorted_flux_1500.feather")

# Step 3: Measure time taken to read CSV, Parquet, and Feather files

# Parquet
start_time = time.time()
df = pd.read_parquet("diffused_UV_data\RA_sorted_flux_1500.parquet")
parquet_read_time = time.time() - start_time

# Feather
start_time = time.time()
df = pd.read_feather("diffused_UV_data\RA_sorted_flux_1500.feather")
feather_read_time = time.time() - start_time
# print("df1:",df)
df = pd.read_feather("diffused_UV_data\RA_sorted_flux_1500.feather").iloc[:, [0, 1, 2]]
df.columns = ['RA', 'Dec', 'Flux']
print("df2:",df)

# Output the times
print(f"Time taken to read CSV: {csv_read_time:.6f} seconds")
print(f"Time taken to read Parquet: {parquet_read_time:.6f} seconds")
print(f"Time taken to read Feather: {feather_read_time:.6f} seconds")
