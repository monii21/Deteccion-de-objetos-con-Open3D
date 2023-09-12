import numpy as np
import cv2
import glob
#TAMAÑO DE LOS CUADRADITOS= 
 
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
 
# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imagenes.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for k in range(0,5):
  cad = 'Img[0-' + str(k) +']*.*'
  images = glob.glob(cad)
  
  for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  

    # Encuentra las esquinas del tablero de ajedrez
    #9,6 es el tamaño de la cuadricula 9x6
    #ret= es un booleano que informa si ha sido detectada o no luna esquina
    #corners= almacena las coordenadas de la supuesta esquina detectada
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
  
    # Si se encuentran, añada puntos de objeto, puntos de imagen (despues de refinarlos)
    #si se encuentra la esquina, se añade a los puntos 3d
    if ret == True:
      objpoints.append(objp) #añade el objeto a la lista de objpoints, que es los puntos en 3d
      
      #aumentamos la precision de las esquinas, es decir, encuentra con "mayor" precision el pixel exacto donde esta la esquina
      #(11,11) = es el tamaño de la ventana de pixeles para encontrar la esquina
      #(-1,-1) = indica que no usamos ese parametro
      #criteria = criterio empleado para aumentar la precision de las esquinas
      corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) 
      imgpoints.append(corners2) #añade la esquina a imgpoints, que es los puntos en 2d
  
      # Dibuja y muestra las esquinas
      #se le pasa la imagen, el tamaño del tablero, las coordenadas de las esquinas y el valor ret, si ret=False pasa y no dibuja nada
      img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
      #cv2.imshow('img',img)
      #cv2.waitKey(500)

  ###################### CALIBRATION ######################

  #ret= parametro anterior, nos informa si las esquinas han sido detectadas
  #mtx= la matriz intrinsica de la camara
  #dist= coeficientes de distorsion de la lente
  #rvecs= matriz de rotacion
  #tvecs= vector de traslacion
  ret, CameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,(640,480),None,None)
  np.savez("cameraParams",cameramatrix=CameraMatrix,dist=dist);
  print("Camera Calibrated", ret)
  print("\nCamera Matrix:\n", CameraMatrix)
  print("\nDistortion Parameters:\n", dist)
  print("\nRotation Vectors\n:", rvecs)
  print("\nTranslation Vectors\n",tvecs)
  
  ###################### UNDISTORTION ######################
  #obtenemos la nueva matriz de la camara
  """
  img=cv2.imread('Img062.jpg')
  height, width= img.shape[:2]
  newCameraMatrix, roi= cv2.getOptimalNewCameraMatrix (CameraMatrix,dist, (width,height),1, (width,height))


  dst = cv2.undistort(img, CameraMatrix, dist, None, newCameraMatrix)
  
  # crop the image (recortar la imagen)
  x,y,w,h = roi
  dst = dst[y:y+h, x:x+w]
  cv2.imwrite('calibresult.png',dst)
  """
  ###################### REPROJECTION ERROR ######################

  mean_error = 0

  for i in range(len(objpoints)):
    imgPoints2,_ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], CameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgPoints2, cv2.NORM_L2) /len(imgPoints2)
    mean_error += error

  print("\n total error: {}".format(mean_error/len(objpoints)))
  print("\n\n\n")
  #OJO!! el error nos lo da en pixeles


#calculamos el error de re-proyeccion para ver como ha salido la calibración
cv2.destroyAllWindows()