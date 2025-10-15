from PIL import Image


from manuscript import OCRPipeline
from manuscript.detectors import EASTInfer
from manuscript.recognizers import TRBAInfer

# инициализация
pipeline = OCRPipeline(
    detector=EASTInfer(score_thresh=0.9),
    recognizer=TRBAInfer(
        model_path=r"C:\Users\USER\Desktop\OCR_MODELS\exp_4_model_32\best_acc_weights.pth",
    )
)

# инфер
page, image = pipeline.process(r"C:\Users\USER\Desktop\scale_1200.jpg", vis=True)
print(page)

pil_img = Image.fromarray(image)

pil_img.show()
