import cv2

videoName = 'v_sbr3HKm2Y9I.mp4'
vidcap = cv2.VideoCapture(videoName)

fps = vidcap.get(cv2.CAP_PROP_FPS)
step = int(round(fps/10))

success,image = vidcap.read()
count = 0
success = True
saveCount = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  saveCount +=1
  for i in range(step):
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1
print(count, saveCount)
print(step)