from manuscript.detectors import EAST

# Model initialization
model = EAST()

# Path to the image
img_path = r"example\ocr_example_image.jpg"

# Inference with visualization
result = model.predict(img_path, vis=True, sort_reading_order=True, profile=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
