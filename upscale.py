from lapsrn import Net
from PIL import Image
import jittor as jt
import jittor.transform as transform
import numpy as np
import cv2

def upscale(name, factor):
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

	# load pretrained model
	cnn = Net()
	cnn.load("./model_best.pkl")

	# run lapSRN
	img_arr = img.astype(float)
	img_arr = img_arr / 255.0
	input_var = jt.Var(img_arr).float().view(1, -1, img_arr.shape[0], img_arr.shape[1])
	HR_2x, HR_4x = cnn(input_var)

	hi_res = jt.squeeze(HR_4x, 0)
	hires_img = hi_res

	return hires_img

def main():
	upscale("Bromo_2.jpg", 2)
	pass

if __name__ == "__main__":
	main()