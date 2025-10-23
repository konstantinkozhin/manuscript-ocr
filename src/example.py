from PIL import Image


from manuscript import OCRPipeline
from manuscript.detectors import EASTInfer
from manuscript.recognizers import TRBAInfer

# инициализация
pipeline = OCRPipeline(
    detector=EASTInfer(
        weights_path=r"C:\east_quad_23_05.pth",
        score_thresh=0.9,
        quantization=2,
    ),
    recognizer=TRBAInfer(
        model_path=r"C:\Users\USER\Desktop\OCR_MODELS\exp_4_model_32\best_acc_weights.pth",
    ),
)

# инфер
page, image = pipeline.process(
    r"C:\shared\data0205\Archives020525\test_images\430.jpg", vis=True, profile=True
)
print(page)

pil_img = Image.fromarray(image)

pil_img.show()
