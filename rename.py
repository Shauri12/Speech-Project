with open('phase4_5_evaluation.py', 'r') as f:
    t = f.read()
t = t.replace("'DCT': 'DCT'", "'DCT': 'DFT'")
with open('phase4_5_evaluation.py', 'w') as f:
    f.write(t)
