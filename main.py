# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

#For downloading the images from the camera
import config_param
import random, string
import requests
import datetime

#For sending picture to telegram
import telegram

#For logging
import logging
logging.basicConfig(level=logging.INFO,
					#filename="AI Surveillance.log",
					format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger("Surveillance AI")

###################################################################################################################################################################
### Helper functions for download and manipulation of images ###
def randostring(length):
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for i in range(length))


def generate_url_image():
	url = rf'http://{config_param.reolink_ip}/cgi-bin/api.cgi?cmd=Snap&channel=0' \
		  rf'&rs={randostring(16)}' \
		  rf'&user={config_param.reolink_user}' \
		  rf'&password={config_param.reolink_pw}'
	return url


def filename_current_time():
    currentDT = datetime.datetime.now()
    file_name = "{}-{}-{} {}.{}.{}.jpg".format(currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)
    return file_name

### Helper functions for visualization of tensorflow data ###
def display_image(image):
	fig = plt.figure(figsize=(20, 15))
	plt.grid(False)
	plt.imshow(image)


def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color,font,thickness=8,display_str_list=()):
	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size
	(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
	draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
			(left, top)],
			width=thickness,
			fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
	display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
	total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

	if top > total_display_str_height:
		text_bottom = top
	else:
		text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
	for display_str in display_str_list[::-1]:
		text_width, text_height = font.getsize(display_str)
		margin = np.ceil(0.05 * text_height)
		draw.rectangle([(left, text_bottom - text_height - 2 * margin),
						(left + text_width, text_bottom)],
					   fill=color)
		draw.text((left + margin, text_bottom - text_height - margin),
				  display_str,
				  fill="black",
				  font=font)
		text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
	"""Overlay labeled boxes on an image with formatted scores and label names."""
	colors = list(ImageColor.colormap.values())

	try:
		font = ImageFont.truetype("LiberationMono-Regular.ttf",
							  50)
	except IOError:
		print("Font not found, using default font.")
		font = ImageFont.load_default()

	for i in range(min(boxes.shape[0], max_boxes)):
		if scores[i] >= min_score:
		  ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
		  display_str = "{}: {}%".format(class_names[i].decode("ascii"),
										 int(100 * scores[i]))
		color = colors[hash(class_names[i]) % len(colors)]
		image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
		draw_bounding_box_on_image(
			image_pil,
			ymin,
			xmin,
			ymax,
			xmax,
			color,
			font,
			display_str_list=[display_str])
		np.copyto(image, np.array(image_pil))
	return image

##################################################################################################################################################################
### Loading TF Model
with tf.Graph().as_default():

	logging.info("Started loading TF-Model into memory")

	detector = hub.Module(r"models/smallAndFast") #Model saved locally. If from web, have to use 'module_handle'
	image_string_placeholder = tf.placeholder(tf.string)
	decoded_image = tf.image.decode_jpeg(image_string_placeholder)
	# Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
	# of size 1 and type tf.float32.
	decoded_image_float = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
	module_input = tf.expand_dims(decoded_image_float, 0)
	result = detector(module_input, as_dict=True)
	init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

	session = tf.Session()
	session.run(init_ops)

	logging.info("Successfully loaded TF-Model into memory")

#Running loop every 5 seconds
while True:
	cycle_start_time = time.clock()
	### Downloading image from webcam
	inference_start_time = time.clock()
	r = requests.get(generate_url_image())
	logging.info("Image download took %.2f seconds." % (time.clock()-inference_start_time))
	image_string = r.content

	### Run image through TF-Model
	inference_start_time = time.clock()
	result_out, image_out = session.run(
		[result, decoded_image],
		feed_dict={image_string_placeholder: image_string})

	logging.info("Analysis took %.2f seconds." % (time.clock()-inference_start_time))

	### Log if wheter or not a person is in the image
	value = 69
	if value in result_out["detection_class_labels"]:
		logging.info("A person was found in the image")

		# Draw image with boxes around people
		indices = np.where(result_out["detection_class_labels"]==69) # 69=b'Person'
		image_with_boxes = draw_boxes(
			np.array(image_out), result_out["detection_boxes"][indices],
			result_out["detection_class_entities"][indices], 
			result_out["detection_scores"][indices],
			min_score=0.1)

		#saving picture to jpeg
		file_path = "saved_pictures/"+filename_current_time()

		inference_start_time = time.clock()
		out_image = tf.image.encode_jpeg(image_with_boxes, quality=80)
		with tf.Session():
			tf.write_file(file_path, out_image).run()
		logging.info("Image was saved to jpeg in %.2f seconds." % (time.clock()-inference_start_time))

		#sending picture to telegram
		bot = telegram.Bot(config_param.api_key)
		bot.send_photo(chat_id=-368960751, photo=open(file_path, 'rb'))

		logging.info("Full cycle for one picture with saving was performed in %.2f seconds.\n" % (time.clock()-cycle_start_time))

	else:
		logging.info("No person was detected")
		logging.info("Full cycle for one picture without person detection was performed in %.2f seconds.\n" % (time.clock()-cycle_start_time))


	time.sleep(5)