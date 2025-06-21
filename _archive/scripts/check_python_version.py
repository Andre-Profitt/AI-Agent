import sys
print(f"Current Python version: {sys.version}")
if sys.version_info[:2] == (3, 12):
    print("✅ Python 3.12 is active!")
else:
    print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} is active. Please switch to 3.12")
