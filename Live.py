import fastai
from fastai import *
from fastai.vision import *
import pathlib
import cv2


path = Path('Dataset')

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
print("DATA")
learn = create_cnn(data, models.resnet50, metrics=accuracy)
print("LEARN")
learn.load('stage1')
print("LOADED")

cap = cv2.VideoCapture(0)
while True:
    if cap.grab():
        flag, frame = cap.retrieve()
        if not flag:
            continue
        else:
            
            cv2.imwrite('test1.jpg',frame)
            img = open_image(pathlib.PosixPath('./test1.jpg'))
            label,index, pred = learn.predict(img)
            cv2.putText(frame, "Number = "+str(label), (380, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 215), 2)
            cv2.putText(frame, "Prob = {0:.4f}".format(torch.max(pred).item()), (380, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('video', frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

