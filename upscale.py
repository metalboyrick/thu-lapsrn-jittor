from lapsrn import Net
from PIL import Image
import jittor as jt
import jittor.transform as transform
import numpy as np

def upscale(name, factor):
	img = Image.open(str(name))
	img = img.convert('L')				# transform to grayscale

	# load pretrained model
	cnn = Net()
	cnn.load("./model_best.pkl")

	# run lapSRN
	img_arr = np.asarray(img).astype("float")
	img_arr = img_arr / 255.0
	input_var = jt.Var(img_arr).float().view(1, -1, img_arr.shape[0], img_arr.shape[1])
	HR_2x, HR_4x = cnn(input_var)

	hi_res = jt.squeeze(HR_4x, 0)
	hires_img = transform.to_pil_image(hi_res)

	hires_img.save(+"out.jpg")

	hires_img
	# return the image

def main():
	upscale("Bromo_2.jpg", 2)
	pass

if __name__ == "__main__":
	main()