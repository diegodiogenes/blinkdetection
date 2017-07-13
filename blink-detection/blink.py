# -*- coding: utf-8 -*-
# Hugo Soares -- Baseado em  Soukupova e Cech apude  Adrian Rosebrock

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import zmq, zmq_tools


#  Pontos da regiao do olho
#         p1        p2
#     p0                p3
#         p5        p4

# EAR = eye aspect ratio =  [ || p2 - p6 || + ||p3 - p5|| ] / 2*||p1-p4||

def calcular_ear(olho):
	A = dist.euclidean(olho[1], olho[5])
	B = dist.euclidean(olho[2], olho[4])
	C = dist.euclidean(olho[0], olho[3])
	ear = (A + B) / (2.0 * C)

	return ear

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

""" Area minima para se considerar uma piscada (area min) e quantidade minimas
de frames consecutivos em que essa area minima nao seja alcancada para se
considerar uma piscada"""
area_min = 0.24
area_max = 0.30
frames_consecutivos_min = 2

""" Contador = numeros de frames seguidos em que a ear < area_min.
Total de piscadas  """
contador = 0

print("[INFO] carregando shape...")
preditor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor(preditor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


"""
Inicializa o ZMQ
"""
print("[INFO] iniciando o servidor ZMQ")
zmq_ctx = zmq.Context()
blink_pub = zmq_tools.Msg_Streamer(zmq_ctx, 'tcp://127.0.0.1:50020')

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

timestamp = time.time()
key = None
fps = 0.0
ear = 0

while True:
	if fileStream and not vs.more():
		break

	frame = vs.read()
	frame = imutils.resize(frame, width=650)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# laplacian = cv2.Laplacian(gray,cv2.CV_64F)
	# sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=5)  # x
	# sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=5)  # y


	rects = detector(gray, 0)

	# Timestamp
	n_timestamp = time.time()
	fps = 1.0 / (n_timestamp - timestamp)
	timestamp = n_timestamp

	major_shape = ()
	max_size = 0

	for rect in rects:
		shape = preditor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		max_p = np.max(shape, axis=0)
		min_p = np.min(shape, axis=0)

		if np.sum(max_p - min_p) > max_size:
			major_shape = shape
			max_size = np.sum(max_p - min_p)

	if len(major_shape) > 0:

		#
		# face = gray[min_p[1]:max_p[1], min_p[0]:max_p[0]]
		#
		# cv2.imshow("face", face)

		# eyes = eye_cascade.detectMultiScale(face)
		#
		# for (ex,ey,ew,eh) in eyes:
		# 	cv2.rectangle(frame,(ex + min_p[0] ,ey + min_p[1]),(ex + min_p[0] + ew,ey + eh + min_p[1]),(0,255,0),2)

		for p in major_shape:
			cv2.circle(frame, tuple(p), 2, (255, 0, 0), -1)

		leftEye = major_shape[lStart:lEnd]
		rightEye = major_shape[rStart:rEnd]
		leftEAR = calcular_ear(leftEye)
		rightEAR = calcular_ear(rightEye)

		# print leftEye[:, -1]
		# print np.max(leftEye[:, -1])

		# rect = leftEye

		# lmax = np.max(leftEye, axis=0)
		# lmin = np.min(leftEye, axis=0)
		# olho_esquerdo = cv2.resize(gray[lmin[1]:lmax[1], lmin[0]:lmax[0]], (120, 40))
		#
		# i, olho_esquerdo_ = cv2.threshold(olho_esquerdo,60,255,cv2.THRESH_BINARY)
		#
		# olho_esquerdo2_ = cv2.Laplacian(olho_esquerdo_, cv2.CV_32F)
		#
		# olho_esquerdo2_ = cv2.convertScaleAbs(olho_esquerdo2_)
		#
		# a = olho_esquerdo2_ - olho_esquerdo_
		#
		# circles = cv2.HoughCircles(a,cv2.cv.CV_HOUGH_GRADIENT,2,200,
		#                 	param1=50,param2=30,minRadius=20,maxRadius=0)
		#
		# # circles = np.uint16(np.around(circles))
		#
		# if circles != None:
		# 	for i in circles[0,:]:
		# 		# draw the outer circle
		# 		cv2.circle(olho_esquerdo,(i[0],i[1]) ,i[2],(0,255,0), 2)
		# 		# draw the center of the circle
		# 		cv2.circle(olho_esquerdo,(i[0],i[1]),2,(0,0,255), 3)


		# circles = np.uint16(np.around(circles))

		# for i in circles[0,:]:
		#     # draw the outer circle
		#     cv2.circle(olho_esquerdo,(i[0],i[1]),i[2],(0,255,0),2)
		#     # draw the center of the circle
		#     cv2.circle(olho_esquerdo,(i[0],i[1]),2,(0,0,255),3)

		# a = olho_esquerdo.shape[:2]
		# print a
		#
		# lmax = np.max(rightEye, axis=0)
		# lmin = np.min(rightEye, axis=0)
		# olho_direito = cv2.resize(gray[lmin[1]:lmax[1], lmin[0]:lmax[0]], (120, 40))
		#
		# # i, olho_direito_ = cv2.threshold(olho_direito,40,150,cv2.THRESH_BINARY)
		# olho_direito_ = cv2.cvtColor(cv2.medianBlur(olho_direito, 5), cv2.COLOR_GRAY2BGR)
		#
		#
		# cv2.imshow("Olho esquerdo", olho_esquerdo)
		# cv2.imshow("Olho direito", olho_direito_)


		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

		# if (ear < area_min and contador == 0) or (ear < area_max and contador > 0):
		if ear < area_min:
			if contador == 0:
				tempo_inicio_piscada = timestamp

				blink_entry = {
					'topic': 'close',
					'timestamp': timestamp
				}

				blink_pub.send('blinks', blink_entry)

				print 'close'

			contador += 1
		else:
			# Abriu novamente o olho
			if contador >= frames_consecutivos_min:
				tempo = timestamp - tempo_inicio_piscada

			# Envia a informação do pisca na rede
				blink_entry = {
					'topic': 'blink',
					'tempo': tempo,
					'timestamp': timestamp
				}

				blink_pub.send('blinks', blink_entry)

				print "piscouu com tempo: ", tempo

			if contador != 0:
				blink_entry = {
					'topic': 'open',
					'timestamp': timestamp
				}

				print 'open'

				blink_pub.send('blinks', blink_entry)

			contador = 0

		cv2.putText(frame, u"Area: {:.2f}".format(ear), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.imshow("Frame", frame)

	# cv2.imshow("Gray", laplacian)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("d"):
		area_min += 0.01
		area_max += 0.01
		print "Area minima alterada: ", area_min
	if key == ord("a") and area_min > 0.1:
		area_min -= 0.01
		area_max += 0.01
		print "Area minima alterada: ", area_min
	if key == ord("w"):
		frames_consecutivos_min += 1
		print "Quantidade de frames alterada: ", frames_consecutivos_min
	if key == ord("s") and frames_consecutivos_min > 1:
		frames_consecutivos_min += 1
		print "Quantidade de frames alterada: ", frames_consecutivos_min

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
