# Read pth files.
import torch
import torch.nn as nn

if __name__ == "__main__":
    
    data = torch.load("/DATA_EDS2/yangry/experiments/bicycle_uniform/chkpnt30000.pth")
    print(data)