import threading
import time
from playsound import playsound
# playsound('golf_shot.wav')
# print("00000000000000000")
# playsound('준비됐으면퍼팅.wav')


def sound_method1():

    playsound('golf_shot.wav')
    # time.sleep(2)  # Simulate a long-running task

def sound_method2():
    playsound('ready_putting.wav')
    # time.sleep(2)  # Simulate a long-running task

def sound_method3():
    playsound('jazzy_fail.wav')

# Create two threads
thread1 = threading.Thread(target=sound_method1)
thread2 = threading.Thread(target=sound_method2)
thread3 = threading.Thread(target=sound_method3)
# Start the threads
thread1.start()
thread2.start()
thread3.start()
# Wait for both threads to finish
# thread1.join()
# thread2.join()
# thread3.join()

print("Doneeeeeeeeeeeeeeeeeeeeeeeee")