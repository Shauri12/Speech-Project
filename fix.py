with open('phase4_5_evaluation.py', 'r') as f:
    text = f.read()
text = text.replace("['psnr']", "['snr']")
with open('phase4_5_evaluation.py', 'w') as f:
    f.write(text)
