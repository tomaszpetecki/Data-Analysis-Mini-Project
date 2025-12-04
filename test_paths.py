import sys
from pathlib import Path


print("CWD:", Path.cwd())
print("sys.path:")
for p in sys.path:
    print("   ", p)