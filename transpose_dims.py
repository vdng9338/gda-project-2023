import numpy as np
import json
import sys

if len(sys.argv) < 3:
    print(f'Usage: python3 {sys.argv[0]} <input.json> [dim1] [dim2] ... [dimk] <output.json>')
    sys.exit(1)
f = open(sys.argv[1], 'r')
arr = np.array(json.load(f))
f.close()
indices = list(map(int, sys.argv[2:-1]))
arr = arr[:,indices]
f = open(sys.argv[-1], 'w')
json.dump(arr.tolist(), f)
f.close()