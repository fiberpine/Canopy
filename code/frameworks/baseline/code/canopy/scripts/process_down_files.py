import os
import numpy as np
import sys

MS_TO_DELETE = 60_000

def get_timestamp(line):
    if line[0] == '#':
        return 0
    else:
        return int(line.split(' ')[0])

def main():
    assert len(sys.argv) == 2
    filepath = sys.argv[1]
    assert os.path.exists(filepath) and os.path.isfile(filepath)
    filename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)
    assert filename.startswith("down-")
    output_filename = "trimmed-" + filename[5:]
    output_path = os.path.join(dirname, output_filename)
    if os.path.exists(output_path):
        print(f"Skipping: {output_path} since trimmed file already exists")
    else:
        print(f"Processing: {output_path}")
        with open(filepath, 'r') as f:
            data = f.readlines()
        timestamps = np.asarray(list(map(get_timestamp, data)))
        last_timestamp = timestamps[-1]
        print(f"\tLast timestamp: {last_timestamp}")
        print(f"\tDeleting last {MS_TO_DELETE}ms")
        
        idx = np.argwhere(timestamps > last_timestamp - MS_TO_DELETE)[0].item()        
        filtered_data = data[:idx]
        print("Length: ", len(filtered_data))

        with open(output_path, 'w') as f:
            for op_string in filtered_data:
                # op_string = op_string.replace('\n', '')        
                f.write(op_string)

if __name__ == "__main__":
    main()

