from manuscript.detectors import EAST

# Model initialization
model = EAST()

# Path to the image
img_path = r"C:\Users\pasha\OneDrive\Рабочий стол\scale_1200.png"

# Inference with visualization
result = model.predict(img_path, vis=True, sort_reading_order=True, profile=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
