from PIL import Image
from glob import glob

# a_filed
files = glob("devel_data/source/a_field/*")
bbox = (350, 50, 900, 450)
for file in files:
    out_name = file.replace("a_filed", "croped").replace(".JPG", "a_filed.JPG")
    Image.open(file).crop(bbox).save(out_name)

# b_pond
files = glob("devel_data/source/b_pond/*")
bbox = (350, 0, 830, 250)
for file in files:
    out_name = file.replace("b_pond", "croped").replace(".JPG", "b_pond.JPG")
    Image.open(file).crop(bbox).save(out_name)

