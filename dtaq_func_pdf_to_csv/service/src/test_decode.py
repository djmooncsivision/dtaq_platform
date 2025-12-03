
import sys
print(f"Encoding: {sys.stdout.encoding}")
b = b'\xec\x83\x81'
print(f"Bytes: {b!r}")
try:
    decoded = b.decode('utf-8')
    print(f"Decoded: {decoded!r}")
except Exception as e:
    print(f"Error: {e}")
