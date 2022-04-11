from Trainer import *
from Utilities import *

f = open("nn.pkl", 'rb')
nn = pickle.load(f)
f.close()

x, nim, im = load_image("test.png", CLASSES)
p = nn.forward(nim)
print(predictions(p))