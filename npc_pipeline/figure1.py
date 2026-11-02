from pathlib import Path
import shutil
from PIL import Image
from .common import new_output
def run():
    out=new_output('Figure1_reference_export')
    src=Path(__file__).resolve().parents[1]/'assets/figure1_reference.png'
    shutil.copy2(src,out/'Figure1_reference.png')
    Image.open(src).convert('RGB').save(out/'Figure1_reference.pdf',resolution=96.)
    print('Archived illustration export (not a vector redraw):',out)
    return out
