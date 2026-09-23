import cv2, numpy as np, torch, sys
src="imagenes/foto-original.jpg"
img=cv2.imread(src); H,W=img.shape[:2]
import os
SC=int(os.environ.get('SC','2'))
pw,ph=W//SC,H//SC
small=cv2.resize(img,(pw,ph),interpolation=cv2.INTER_AREA)
P=lambda pts: np.array(pts,np.int32)
regions=[
 P([(0,500),(345,500),(372,950),(368,1000),(330,1012),(252,1068),(208,1068),(185,1015),(40,1015),(0,965)]),  # perchero principal
 P([(600,395),(702,395),(702,850),(600,850)]),                                                     # jeans colgados derecha
 P([(492,398),(702,398),(702,670),(560,670),(560,470),(492,470)]),                                  # estantes / pilas
]
person=P([(430,388),(345,395),(335,450),(329,520),(327,600),(334,640),(339,665),(344,750),(351,820),(352,862),(360,882),(376,882),(373,950),(368,1000),(364,1052),(468,1062),(464,1102),(506,1102),(512,1062),(624,1052),(640,1074),(672,1074),(668,1038),(626,1008),(612,822),(634,806),(636,688),(598,676),(582,606),(562,528),(562,478),(546,438),(496,418),(492,402),(440,393)])
mask=np.zeros((H//2,W//2),np.uint8)
for r_ in regions: cv2.fillPoly(mask,[r_],255)
cv2.fillPoly(mask,[person],0)
mask=cv2.resize(mask,(pw,ph),interpolation=cv2.INTER_NEAREST)
if "--show" in sys.argv:
    o=small.copy(); o[mask>0]=(o[mask>0]*0.4+np.array([0,0,255])*0.6).astype(np.uint8)
    cv2.polylines(o,[person],True,(0,255,0),1); cv2.imwrite("maskview.jpg",o); sys.exit()
# modelo: github.com/enesmsahin/simple-lama-inpainting/releases (big-lama.pt)
m=torch.jit.load("big-lama.pt",map_location="cpu").eval()
def pad8(a): 
    h,w=a.shape[:2]; return np.pad(a,((0,(8-h%8)%8),(0,(8-w%8)%8))+((0,0),)*(a.ndim-2),mode="reflect")
x=torch.from_numpy(pad8(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))).permute(2,0,1)[None].float()/255
mk=torch.from_numpy(pad8(mask))[None,None].float()/255
with torch.no_grad(): r=m(x,(mk>0).float())[0].permute(1,2,0).numpy()[:ph,:pw]
r=cv2.cvtColor((np.clip(r,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
cv2.imwrite("lama_small.png",r)
big=cv2.resize(r,(W,H),interpolation=cv2.INTER_CUBIC).astype(np.float32)
bm=cv2.resize(mask,(W,H),interpolation=cv2.INTER_LINEAR)
bm=cv2.GaussianBlur(bm,(0,0),1.5).astype(np.float32)/255
out=img*(1-bm[...,None])+big*bm[...,None]
cv2.imwrite("sin_jeans.jpg",out.astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,95])
cv2.imwrite(f"prev_sc{SC}.jpg",cv2.resize(out.astype(np.uint8),(W//2,H//2)))
