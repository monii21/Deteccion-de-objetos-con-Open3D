import numpy as np
import cv2
import glob

#cargamos los parametros de la camara guardados 
with np.load('cameraParams.npz') as file:
    CameraMatrix,Dist=[file[i] for i in ('cameramatrix','dist')]


#funcion que se encarga de dibujar la piramide en el fotograma
def drawPiramid(img, corners, piramid_points):
    piramid_points=np.int32(piramid_points).reshape(-1,2)

    #dibujamos la base cuadrada de la piramide
    img = cv2.drawContours(img, [piramid_points[:4]],-1,(255,0,0),-3)
    
    #dibujamos cada una de las aristas de la piramide
    #funcion .ravel() nos convierte los puntos en un array plano
    img = cv2.line(img,tuple(piramid_points[0].ravel()),tuple(piramid_points[4].ravel()),(255,0,0),5)
    img = cv2.line(img,tuple(piramid_points[1].ravel()),tuple(piramid_points[4].ravel()),(255,0,0),5)
    img = cv2.line(img,tuple(piramid_points[2].ravel()),tuple(piramid_points[4].ravel()),(255,0,0),5)
    img = cv2.line(img,tuple(piramid_points[3].ravel()),tuple(piramid_points[4].ravel()),(255,0,0),5)

    return img

# termination criterio
#cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER= tipo de criterio usado
#30 = numero maximo de iteraciones hasta que se da por satisfecho para ver si la esquina esta ya bien
#0.001= epsilon, cambio min que tiene que haber entre dos iteraciones de buscar esquinas para que siga buscando o no

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# preparar puntos de objeto, como (0,0,0,0), (1,0,0,0), (2,0,0,0)...., (6,5,0)
#el "3" es por la X,Y,Z. 
#el 9*6 es el tamaño de la cuadricula
#OJO, significa el numero de esquinitas, no el numero de cuadrados
#crea la matriz para generar los puntos 3D de las intersecciones entre los cuadrados
objp = np.zeros((6*9,3), np.float32) 
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Puntos correspondientes a los ejes
axis = np.float32([[2,0,0],[0,2,0],[0,0,-2]]).reshape(-1,3)

#puntos correspondientes a los vertices de la piramide
piramid= np.float32([[2,1,0],[2,5,0],[6,5,0],[6,1,0],[4,3,-5]]).reshape(-1,3)


capture = cv2.VideoCapture(0)
while (capture.isOpened()):
    ret, img = capture.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    # Encuentra las esquinas del tablero de ajedrez
    #9,6 es el tamaño de la cuadricula 9x6
    #ret= es un booleano que informa si ha sido detectada o no luna esquina
    #corners= almacena las coordenadas de la supuesta esquina detectada
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # Si se encuentran, añada puntos de objeto, puntos de imagen (despues de refinarlos)
    #si se encuentra la esquina, se añade a los puntos 3d
    if ret == True:
        #aumentamos la precision de las esquinas, es decir, encuentra con "mayor" precision el pixel exacto donde esta la esquina
        #(11,11) = es el tamaño de la ventana de pixeles para encontrar la esquina
        #(-1,-1) = indica que no usamos ese parametro
        #criteria = criterio empleado para aumentar la precision de las esquinas
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) 
        

        #funcion que nos devuelven los parametros extrinsecos
        ret,rvecs,tvecs= cv2.solvePnP(objp,corners2,CameraMatrix,Dist)
        

        #proyectamos los puntos 2D sobre el patro 3D
        piramid_points,_ = cv2.projectPoints(piramid,rvecs,tvecs,CameraMatrix,Dist)
        axis_points,_ = cv2.projectPoints(axis,rvecs,tvecs,CameraMatrix,Dist)

        #dibujamos los ejes
        img = cv2.line(img,tuple(corners2[0].ravel()),tuple(axis_points[0].ravel()),(255,0,0),5)
        img = cv2.line(img,tuple(corners2[0].ravel()),tuple(axis_points[1].ravel()),(0,255,0),5)
        img = cv2.line(img,tuple(corners2[0].ravel()),tuple(axis_points[2].ravel()),(0,0,255),5)

        img=drawPiramid(img, corners2, piramid_points)
    
    cv2.imshow('img',img)
    if (cv2.waitKey(1) == ord('s')):
        break