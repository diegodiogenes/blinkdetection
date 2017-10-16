# -*- coding: utf-8 -*-

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


def calcular_ear(olho):
	A = dist.euclidean(olho[1], olho[5])
	B = dist.euclidean(olho[2], olho[4])
	C = dist.euclidean(olho[0], olho[3])
	ear = (A + B) / (2.0 * C)

	return ear

def calcular_boca(boca):
	C = dist.euclidean(boca[0], boca[6])
	D = dist.euclidean(boca[12], boca[18])
	E = dist.euclidean(boca[13], boca[17])
	F = dist.euclidean(boca[14], boca[16])
	G = dist.euclidean(boca[14], boca[18])
	H = dist.euclidean(boca[12], boca[16])
	ear_boca = (D+E+F+G)/((3.0 * H) + C)

	return ear_boca

def distancia_nariz(nariz,boca):
	#A = dist.euclidean(nariz[5], boca[12]) #boca[0]
	B = dist.euclidean(nariz[6], boca[14]) #boca[5]
	#C = dist.euclidean(nariz[7], boca[4])
	#D = dist.euclidean(nariz[4], boca[2])

	dis_nariz = B  #(A + C + B) / (3.0 * D)

	return dis_nariz

def distancia_sobrancelha(sobrancelha,olho):
	A = dist.euclidean(sobrancelha[0], olho[0])
	B = dist.euclidean(sobrancelha[1], olho[5])
	C = dist.euclidean(sobrancelha[2], olho[4])
	D = dist.euclidean(sobrancelha[3], olho[4])
	E = dist.euclidean(sobrancelha[4], olho[3])

	dis_sobrancelha = (B+C+D)/(A+E)*3.0

	return dis_sobrancelha

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

""" Area minima para se considerar uma piscada (area min) e quantidade minimas
de frames consecutivos em que essa area minima nao seja alcancada para se
considerar uma piscada"""
area_min = 0.24
area_boca = 14.0
area_max = 0.30
frames_consecutivos_min = 2
boca_min = 0
nariz_min = 0
sobrancelha_min = 0

""" Contador = numeros de frames seguidos em que a ear < area_min.
Total de piscadas  """
contador = 0

print("[INFO] carregando shape...")
preditor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor(preditor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(bStart, bEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(slStart, slEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(srStart, srEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

"""
Inicializa o ZMQ
"""
print("[INFO] iniciando o servidor ZMQ")
zmq_ctx = zmq.Context()
blink_pub = zmq_tools.Msg_Streamer(zmq_ctx, 'tcp://127.0.0.1:50020')

print("[INFO] starting video stream thread...")
vs = VideoStream(src=1).start()
fileStream = False
time.sleep(1.0)
tempo_abriu_boca = 0
timestamp = time.time()
key = None
fps = 0.0
ear = 0
ear_boca = 0
dis_nariz = 0
dis_sobrancelha = 0
aberta = False
calibrar = False
conta = 0
aux = 0
aux2 = 0
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
			cv2.circle(frame, tuple(p), 2, (0, 0, 0), -1)

		leftEye = major_shape[lStart:lEnd]
		rightEye = major_shape[rStart:rEnd]
		rightEyebrow = major_shape[srStart:srEnd]
		leftEyebrow = major_shape[slStart:slEnd]
		boca = major_shape[bStart:bEnd]
		nariz = major_shape[nStart:nEnd]
		jaw = major_shape[jStart:jEnd]
		distRight = distancia_sobrancelha(rightEyebrow, rightEye)
		distLeft = distancia_sobrancelha(leftEyebrow, leftEye)
		leftEAR = calcular_ear(leftEye)
		rightEAR = calcular_ear(rightEye)
		bocaEAR = calcular_boca(boca)
		dis_nariz = distancia_nariz(nariz,boca)

		dis_sobrancelha = (distRight+distLeft)/2

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		bocaHull = cv2.convexHull(boca)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [bocaHull], -1, (65, 223, 107), 1)

		#calibrar e fazer a média
		inicio = time.time()
		cont = 0
		while(calibrar):
			tempod = time.time() - inicio

			boca_min += bocaEAR
			nariz_min += dis_nariz
			sobrancelha_min += dis_sobrancelha
			cont += 1
			if(tempod > 0.5):
				boca_min = (boca_min/cont) * 1.02
				nariz_min = (nariz_min/cont) * 1.04
				sobrancelha_min = (sobrancelha_min/cont) * 1.01

				print boca_min
				print nariz_min
				print sobrancelha_min

				calibrar = False


		if (bocaEAR > boca_min*1.2):
			if conta == 0:
				abriu_boca = timestamp

				print "abriu a boca"

			conta += 1
		else:
			if conta >= frames_consecutivos_min:
				tempo_boca = timestamp - abriu_boca

				print "abriu boca com tempo", tempo_boca

				if(tempo_boca > 2.0):
					blink_entry = {
						'topic': 'abrir_boca'
					}

				blink_pub.send('blinks', blink_entry)

			if conta !=0:
				print "boca fechada"

			conta = 0
		dif_rela = (abs(dis_sobrancelha - sobrancelha_min)/dis_sobrancelha)*100
		if(dis_sobrancelha > sobrancelha_min * 1.04):
			if aux2 == 0:
				levantou = timestamp

				print "levantou a sobrancelha"

			conta+=1
		else:
			if aux2 >= frames_consecutivos_min:
				tempo_levantada = timestamp - levantou

				print "levantou a sobrancelha com tempo:", tempo_levantada

				if(tempo_levantada > 2.0):
					blink_entry = {
						'topic': 'sobrancelha'
					}

				blink_pub.send('blinks', blink_entry)

			if aux2 !=0:
				print "sobrancelha normal"

			aux2 = 0

		dif_rel = (abs(dis_nariz-nariz_min)/dis_nariz)*100
		#print dif_rel
		if(dis_nariz < nariz_min and dif_rel>14.0):
			if aux == 0:
				inicio_muxoxo = timestamp

				print "começou muxoxo"

			aux += 1
		else:
			if aux >= frames_consecutivos_min:
				tempo_muxoxo = timestamp - inicio_muxoxo

				print "fez muxoxo com tempo", tempo_muxoxo

				if(tempo_muxoxo>2.0):
					blink_entry ={
						'topic':'muxoxo'
					}

				blink_pub.send('blinks', blink_entry)

			if aux != 0:
				print "normal"

			aux = 0

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
				if(tempo>2.0):
					blink_entry = {
						'topic': 'piscar',
						'tempo': tempo,
						'timestamp': timestamp
					}

					blink_pub.send('blinks', blink_entry)

					print "piscou com tempo: ", tempo

			if contador != 0:
				blink_entry = {
					'topic': 'open',
					'timestamp': timestamp
				}

				print 'open'

				blink_pub.send('blinks', blink_entry)

			contador = 0

		cv2.putText(frame, u"Olho: {:.2f}".format(ear), (250, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, u"Boca: {:.2f}".format(bocaEAR), (470, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 223, 107), 2)

		cv2.putText(frame, u"Distancia: {:.2f}".format(dis_nariz), (450, 100),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 223, 107), 2)

		cv2.putText(frame, u"Sobrancelha: {:.2f}".format(dis_sobrancelha), (0, 100),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (65, 223, 107), 2)

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
	if key == ord("c"):
		calibrar = True
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
