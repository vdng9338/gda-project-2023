import json
import numpy as np
import sys

def main():
    if len(sys.argv) < 4:
        print(f'Usage: python3 {sys.argv[0]} <file1.json> <file2.json> <output.json>')
        sys.exit(1)
    f1 = open(sys.argv[1], 'r')
    shape1 = np.array(json.load(f1))
    f1.close()
    f2 = open(sys.argv[2], 'r')
    shape2 = np.array(json.load(f2))[:, [1, 0]]
    result = np.zeros((shape1.shape[0], 4))
    for i in range(shape1.shape[0]):
        pos = i/(shape1.shape[0]-1)*(shape2.shape[0]-1)
        index2 = int(pos)
        frac = pos-int(pos)
        if index2 < len(shape2)-1:
            result[i] = np.concatenate((shape1[i], shape2[index2] + frac*(shape2[index2+1] - shape2[index2])))
        else:
            result[i] = np.concatenate((shape1[i], shape2[index2]))
    out = open(sys.argv[3], 'w')
    json.dump(result.tolist(), out)
    out.close()

if __name__ == "__main__":
    main()