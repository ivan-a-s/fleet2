import fastsim as fsim
from scipy.signal import savgol_filter
import pandas as pd

# Filename
# df = pd.read_csv("drive_cycles/Fleet DNA Drayage Representative_.csv")
df = pd.read_csv("drive_cycles/Fleet DNA Local Delivery Representative_.csv")
df = pd.read_csv("drive_cycles/Urban Dynamometer Driving Schedule for Heavy-Duty Vehicles (UDDS HD).csv")
df = pd.read_csv("drive_cycles/CARB Heavy Heavy-Duty Diesel Truck (HHDDT) Cruise Segment.csv")
# df = pd.read_csv("drive_cycles/Fleet DNA Regional-Haul Representative_.csv")
# df = pd.read_csv("drive_cycles/Fleet DNA Long-Haul Representative_.csv")

# Rename columns
df = df.rename(columns={
    "Time (seconds)": "time_seconds",
    "Speed (mph)": "speed_mph",
    "Grade (rise/run)": "grade"
})

# Change units
df["speed_meters_per_second"] = df["speed_mph"] * 0.44704

# Savitzky-Golay smoothing
df["speed_meters_per_second"] = savgol_filter(
    df["speed_meters_per_second"],
    window_length=61,
    polyorder=3
)

# --- keep only FASTSim-required columns ---
cyc_df = df[["time_seconds", "speed_meters_per_second"]]#, "grade"]]

# --- convert to dict-of-lists (FASTSim format) ---
cyc_dict = {
    "time_seconds": cyc_df["time_seconds"].tolist(),
    "speed_meters_per_second": cyc_df["speed_meters_per_second"].tolist(),
    # "grade": cyc_df["grade"].tolist()
}

# --- build cycle ---
cyc = fsim.Cycle.from_pydict(cyc_dict)

import json
# If cyc.to_json() gives a raw string, you can format it like this:
data = json.loads(cyc.to_json())
with open("drive_cycles/cruise_hdt.json", "w") as f:
    json.dump(data, f, indent=4)