import json, os, glob
from pathlib import Path
from PIL import Image

raw = Path(os.environ['nnUNet_raw']) / 'Dataset101_SRC'

# Fix 1: Convert all RGBA/palette PNGs to RGB in imagesTr and imagesTs
converted = 0
for folder in ['imagesTr', 'imagesTs']:
    for p in (raw / folder).glob('*.png'):
        img = Image.open(p)
        if img.mode != 'RGB':
            img.convert('RGB').save(p)
            converted += 1
print('Converted ' + str(converted) + ' images to RGB')

# Fix 2: Update dataset.json — 3 channel entries for RGB
n = len(list((raw / 'imagesTr').glob('*.png')))
d = json.load(open(raw / 'dataset.json'))
d['channel_names'] = {'0': 'R', '1': 'G', '2': 'B'}
d['numTraining']   = n
json.dump(d, open(raw / 'dataset.json', 'w'), indent=2)
print('dataset.json updated: channels=3, numTraining=' + str(n))
print(json.dumps(d, indent=2))
