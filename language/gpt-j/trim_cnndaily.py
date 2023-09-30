import os
import sys
import json

def trim_json(ifile, length=10):
    if os.path.exists(ifile):
        
        with open(ifile) as INFILE:
            data = json.load(INFILE)
        
        dir = os.path.dirname(ifile)
        filename = f"{os.path.splitext(os.path.basename(ifile))[0]}_{length}_samples.json"
        filepath = f"{dir}/{filename}"

        with open(filepath, "w") as OUTFILE:
            json.dump(data[0:length], OUTFILE, indent=True)
        
        print(f"[INFO]: Generated {filepath}")
    else:
        raise FileNotFoundError(f"{ifile} not found")

if __name__ == "__main__":
    trim_json(sys.argv[1], int(sys.argv[2]))