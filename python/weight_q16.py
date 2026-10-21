# genere weights_q610.bin depuis params.h  (a lancer sur le PC)
import re, numpy as np

txt  = open("params.h").read()
body = re.search(r"WEIGHTS\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;", txt, re.S).group(1)

# chaque valeur ressemble a "0.02758791f" -> on enleve le 'f' et on parse
floats = [float(t.strip().rstrip('f')) for t in body.split(',') if t.strip()]
arr = np.array(floats, dtype=np.float32)
print("nb poids lus :", arr.size)          # doit afficher 60384

q = np.clip(np.round(arr * 1024), -32768, 32767).astype(np.int16)  # Q6.10 + saturation
q.tofile("weights_q610.bin")
print("ecrit weights_q610.bin :", q.nbytes, "octets")   # 60384*2 = 120768








