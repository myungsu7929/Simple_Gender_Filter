package_path = "."
import sys
sys.path.append(package_path)
from PIL import Image
import torch
from gender_filter import GenderFilter

GF = GenderFilter()

test_male = Image.open('gender_filter/male.png').convert('RGB')
test_female = Image.open('gender_filter/female.png').convert('RGB')

print(GF(test_male))
print(GF(test_female))