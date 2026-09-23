import cv2, numpy as np
src="imagenes/foto-original.jpg"
img=cv2.imread(src); h,w=img.shape[:2]
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
H,S,V=hsv[...,0],hsv[...,1],hsv[...,2]
blue=((H>=85)&(H<=130)&(S>=6)&(V>=30)).astype(np.uint8)*255
region=np.zeros((h,w),np.uint8)
for x0,y0,x1,y1 in [(0,1000,720,2010),(1230,960,1410,1660),(960,820,1390,1250)]:
    region[y0:y1,x0:x1]=255
person=(np.array([(330,380),(455,380),(440,400),(565,400),(565,640),(605,680),(640,700),(640,830),(630,1040),(675,1060),(670,1100),(455,1105),(360,1010),(345,870),(322,700),(322,560)],np.float32)*2).astype(np.int32)
cv2.fillPoly(region,[person],0)
m=cv2.bitwise_and(blue,region)
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
m=cv2.GaussianBlur(m,(0,0),2.5).astype(np.float32)/255
new=hsv.copy()
new[...,0]=168                      # rosa
new[...,1]=np.clip(S*1.4+25,0,255)
out=cv2.cvtColor(new.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32)
res=img*(1-m[...,None])+out*m[...,None]
cv2.imwrite("imagenes/foto-pantalones-rosa.jpg",res.astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,95])
