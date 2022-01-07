from lapsrn import Net
from PIL import Image
import jittor as jt
import jittor.transform as transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse

def upscale_from_file(name, factor, model="./model_best.pkl"):
	img = cv2.imread(name, cv2.IMREAD_COLOR)

	(B, G, R)= cv2.split(img)

	B_r = upscale(B.astype(float), factor, model).astype(np.uint8)
	G_r = upscale(G.astype(float), factor, model).astype(np.uint8)
	R_r = upscale(R.astype(float), factor, model).astype(np.uint8)

	return cv2.merge([B_r, G_r, R_r])

def upscale(img, factor, model):
	
	# load pretrained model
	cnn = Net()
	cnn.load(model)

	# run lapSRN
	img_arr = img.astype(float)
	img_arr = img_arr / 255.0
	input_var = jt.Var(img_arr).float().view(1, -1, img_arr.shape[0], img_arr.shape[1])
	HR_2x, HR_4x = cnn(input_var)

	if factor==2:
		sel_factor = HR_2x
	elif factor == 4:
		sel_factor = HR_4x

	hi_res = np.array(sel_factor.data[0].astype(np.float32))
	hi_res *= 255.0
	hi_res[hi_res < 0.0] = 0.0
	hi_res[hi_res > 255.0] = 255.0
	hi_res = hi_res[0,:,:]

	return hi_res

def main():
	parser = argparse.ArgumentParser(description='LapSRN Single Image Upscaling')
	parser.add_argument("--model", type=str, default="./model_best.pkl", help="Select a trained model")
	parser.add_argument("--input", type=str, default="", help="Select an input image to be enlarged")
	parser.add_argument("--scale", type=int, default=4, help="Enlargement scale, 2 or 4")
	parser.add_argument("--cuda", type=bool, default=False,help="use CUDA")
	opt = parser.parse_args()

	if opt.cuda:
		print("Using CUDA")
		jt.flags.use_cuda = 1

	im = upscale_from_file(opt.input, opt.scale, opt.model)
	
	cv2.imwrite("out.jpg", im)

if __name__ == "__main__":
	main()