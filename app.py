import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse, FileResponse
import numpy as np
import io
from PIL import Image
import torch
import RRDBNet_arch as arch 
import cv2
# from ISR.models import RDN

app = FastAPI()

model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# def load_image_into_numpy_array(data):
#     return np.array(Image.open(io.BytesIO(data)))



# rdn = RDN(weights='noise-cancel')



@app.post("/")
async def root(file: UploadFile = File(...)):

    # image = load_image_into_numpy_array(await file.read())
    contents = io.BytesIO(await file.read())
    # # nparr = np.fromstring(contents, np.uint8)
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # sr_img = rdn.predict(image)
    # Image.fromarray(sr_img)

    # if len(image) > 2 :
    #     # done = image.tobytes()
    # else:
    #     done = "not done"
    cv2.imwrite('results.png', output) 

    return FileResponse("results.png", media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)