import mkl #shouldn't be need but fix a bug with faiss and mkl linking
mkl.get_max_threads()
from flask import Flask, send_from_directory, request, render_template
import faiss
import os
import io
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import csv
import base64


def create_app():
	IMG_HEIGHT = 224
	IMG_WIDTH = 224

	index = faiss.read_index('app/1M4_index_PCA180.faiss')
	with open('app/img_id_1M4.index', 'rb') as file:
		img_id = pickle.load(file)

	zooscan_id = {}
	with open('app/zooscan2_id.csv') as csvfile:
		zooscan_reader = csv.reader(csvfile)
		for row in zooscan_reader:
			zooscan_id[row[2]] = row[3]

	mobile_net_layer = hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4', trainable=False)
	cnn_model = tf.keras.models.load_model('app/model.h5', custom_objects={'KerasLayer': mobile_net_layer})
	image_input = tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
	output = cnn_model.layers[1](image_input)

	feature_cnn_model = tf.keras.Model(inputs=image_input, outputs=output)
	feature_cnn_model.compile(
	  optimizer=tf.keras.optimizers.Adam(),
	  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy'])


	def expand2square(pil_img, background_color):
	    width, height = pil_img.size
	    if width == height:
	        return pil_img
	    elif width > height:
	        result = Image.new(pil_img.mode, (width, width), background_color)
	        result.paste(pil_img, (0, (width - height) // 2))
	        return result
	    else:
	        result = Image.new(pil_img.mode, (height, height), background_color)
	        result.paste(pil_img, ((height - width) // 2, 0))
	        return result
	    
	def format_image(img):
	    img = expand2square(img, (255, 255, 255))
	    return img.resize((IMG_WIDTH, IMG_HEIGHT))

	def find_5_closest(img):
	    vector = feature_cnn_model(img[np.newaxis,:]).numpy()
	    distance, neighbors = index.search(vector, 5)
	    return list(map(lambda x: (img_id[x]), neighbors[0]))

	def img_to_base64_jpg(img):
		with io.BytesIO() as img_bytes:
			img.save(img_bytes, format='jpeg')
			base64_bytes = base64.b64encode(img_bytes.getvalue())
		return base64_bytes.decode('ascii')

	app = Flask(__name__)

	@app.route('/')
	def send_index():
		return send_from_directory('static', 'index.html')

	@app.route('/<path:path>')
	def send_static(path):
	    return send_from_directory('static', path)

	@app.route('/tarasimsearch', methods=['GET','POST'])
	def sim_search():
		img = request.files['img_query']
		img = Image.open(img)
		query = np.array(format_image(img))
		img_links = list(map(lambda x: zooscan_id[x], find_5_closest(query)))
		inline_img = img_to_base64_jpg(img)
		return render_template('query_result.html', img_links=img_links, img_query=inline_img)

	return app