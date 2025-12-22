import sys
print(f"Python Executable: {sys.executable}")
print("Attempting to import easyocr...")
try:
    import easyocr
    print("SUCCESS: easyocr imported.")
except ImportError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")

print("Attempting to import pdfplumber...")
try:
    import pdfplumber
    print("SUCCESS: pdfplumber imported.")
except ImportError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")
