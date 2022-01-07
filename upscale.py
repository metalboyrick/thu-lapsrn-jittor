from lapsrn import Net
from PIL import Image
import jittor as jt
import jittor.transform as transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

def upscale_from_file(name, factor, model="./model_best.pkl"):
	img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	sio.savemat('./np_matrix_input.mat', {'down':img})
	img = sio.loadmat("./np_matrix_input.mat")['down']
	return upscale(img, factor, model).astype(np.uint8)

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
	im = upscale_from_file("Bromo_3.jpg", 2, "trained_models/lapsrn_model_epoch_10.pkl")
	cv2.imwrite("BROMO_OUT.jpg", im)

if __name__ == "__main__":
	main()