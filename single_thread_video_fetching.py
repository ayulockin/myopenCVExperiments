import cv2
import datetime
import numpy as np

# By moving these blocking I/O operations to a separate
# thread and maintaining a queue of decoded frames

cap = cv2.VideoCapture("videos/BirdNoSound.mp4")
start = datetime.datetime.now()
num_frames = 0

while True:
    _,frame = cap.read()

    if not _:
        break

##    frame = cv2.resize(frame, (400,400))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    cv2.putText(frame, "Slow Method", (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    num_frames+=1  

end = datetime.datetime.now()
elapsed = (end-start).total_seconds()
print("[INFO] elasped time: {:.2f}".format(elapsed))
print("[INFO] approx. FPS: {:.2f}".format(num_frames/elapsed))

cap.release()
cv2.destroyAllWindows()

    
    














