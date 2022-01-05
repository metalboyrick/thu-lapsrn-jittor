from lapsrn import *
from PIL import Image
import jittor as jt
import jittor.transform as transform

def upscale(name, factor):
	img = Image.open(str(name))
	img = img.convert('L')				# transform to grayscale

	# load pretrained model
	cnn = Net()
	cnn.load("model_epoch_100.pth")

	# run lapSRN
	img = img.astype("float")
	img = img/255.0
	input_var = jt.float32(img)
	HR_2x, HR_4x = cnn(input_var)

	hi_res = jt.squeeze(HR_4x, 0)
	hires_img = transform.to_pil_image(hi_res)

	hires_img.save(+"out.jpg")


	# return the image

def main():
	upscale("Bromo_2.jpg", 2)
	pass

if __name__ == "__main__":
	main()