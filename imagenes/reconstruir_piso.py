import cv2, numpy as np
src="imagenes/foto-original.jpg"
img=cv2.imread(src).astype(np.float32); H,W=img.shape[:2]
lama=cv2.imread("sin_jeans.jpg").astype(np.float32)   # resultado LaMa (SC=3)
exec(open("imagenes/quitar_jeans_lama.py").read().split("person=")[0].split("regions=")[1].join(["regions=",""]) if False else "")
# máscara a resolución completa (coordenadas de vista previa x2)
import re
code=open("imagenes/quitar_jeans_lama.py").read()
regions=eval(re.search(r"regions=(\[.*?\n\])",code,re.S).group(1).replace("P(","np.array(").replace("])","],np.int32)"))
person=eval(re.search(r"person=P\((\[.*?\])\)",code).group(1))
person=np.array(person,np.int32)*2
mask=np.zeros((H,W),np.uint8)
for r in regions: cv2.fillPoly(mask,[r*2],255)
cv2.fillPoly(mask,[person],0)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
# --- modelo de color del piso beige
samp=[(60,2020,180,2120),(1200,1580,1380,1660),(1290,1560,1390,1600)]
X=[];Y=[]
for x0,y0,x1,y1 in samp:
    p=img[y0:y1,x0:x1].reshape(-1,3); gx,gy=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
    X.append(np.stack([np.ones(gx.size),gx.ravel()/W,gy.ravel()/H],1)); Y.append(p)
X=np.vstack(X);Y=np.vstack(Y)
coef,*_=np.linalg.lstsq(X,Y,rcond=None)
base=np.array([150,196,224],np.float32)
shade=(0.93+0.07*np.clip((yy-1000)/1100,0,1))[...,None]
beige=base*shade
rng=np.random.default_rng(0)

# juntas de baldosa en perspectiva (punto de fuga aprox. (1014,920) en px completos)
grout=np.zeros((H,W),np.uint8)
vx,vy=1014,920
for n in range(-12,8):
    xb=1014+n*300; cv2.line(grout,(vx,vy),(int(vx+(xb-vx)*3),int(vy+(2200-vy)*3)),255,3)
d0=1.0
for n in range(-3,30):
    d=d0+n*0.45
    if d<=0.05: continue
    y=int(vy+1280*d0/d); cv2.line(grout,(0,y),(W,y),255,3)
grout=cv2.GaussianBlur(grout,(0,0),1.2).astype(np.float32)[...,None]/255
beige=beige*(1-0.0*grout)
# variacion suave de iluminacion (mas claro cerca de la luz del espejo, a la izquierda)
beige=beige*(1.04-0.06*np.clip(xx/1400,0,1))[...,None]
beige+=cv2.GaussianBlur(rng.normal(0,2.5,(H,W)).astype(np.float32),(0,0),1.0)[...,None]
# --- alfombra/piso oscuro: poligono con esquina lejana en (490,1670) y borde en y=1670
mat=np.zeros((H,W),np.uint8)
cv2.fillPoly(mat,[np.array([(100,2270),(490,1670),(1420,1670),(1420,2576),(0,2576),(0,2400)],np.int32)],255)
# textura de madera: franja real del piso oscuro, repetida hacia arriba
strip=img[2150:2450,1040:1400]              # piso oscuro junto a los pies
sh,sw=strip.shape[:2]
wood=np.zeros_like(img)
for y in range(0,H,sh):
    for x in range(0,W,sw):
        t=strip if (y//sh)%2==0 else strip[::-1]
        wood[y:y+sh,x:x+sw]=t[:min(sh,H-y),:min(sw,W-x)]
matf=cv2.GaussianBlur(mat,(0,0),3).astype(np.float32)[...,None]/255
Ll=cv2.cvtColor(lama.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
Ll=cv2.GaussianBlur(Ll,(0,0),6)
ref=np.median(Ll[mask>0])
mod=np.clip((Ll/ref),0.75,1.15)**0.6
beige=beige*mod[...,None]
floor=beige*(1-matf)+wood*matf
# usar piso reconstruido en el hueco por debajo de y=1060 (izq.) / y=1300 (der.)
use=np.zeros((H,W),np.float32)
use[1010:, :760]=1; use[1300:,1180:]=1
use=cv2.GaussianBlur(use,(0,0),30)
m=cv2.GaussianBlur(mask,(0,0),2).astype(np.float32)/255
fill=lama*(1-use[...,None])+floor*use[...,None]
out=img*(1-m[...,None])+fill*m[...,None]
out=np.clip(out,0,255).astype(np.uint8)
cv2.imwrite("final.jpg",out,[cv2.IMWRITE_JPEG_QUALITY,95])
cv2.imwrite("prevf.jpg",cv2.resize(out,(W//2,H//2))[330:1150,:760])
