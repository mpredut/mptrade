

import time
import json
from multiprocessing import shared_memory


BUF_SIZE = 1024
shmname = "trend_data"

def shmConnectForRead(name):
    shm = None
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            print(f"Shared memory does not exist yet. Waiting...")
            time.sleep(1)
    print(f"Conectat la shared memory {name} for read!")
    return shm
    
def shmConnectForWrite(name):
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=BUF_SIZE)
    except FileExistsError:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            # The segment was deleted between the two attempts.
            shm = shared_memory.SharedMemory(name=name, create=True, size=BUF_SIZE)
    print(f"Conectat la shared memory {name} for write!")
    return shm
      

def shmRead(shm):
    if shm is None:
        print(f"SHM fail !!!!")
        return None
        
    try:            
        length = int.from_bytes(shm.buf[:4], "little")
        if length == 0:
            return None  # Nothing has been written yet.
        raw = bytes(shm.buf[4:4+length])
        
        return json.loads(raw.decode("utf-8"))
    except:
        print(f"shmRead fail !!!!")
        return None
    

def shmWrite(shm, data: dict):
    if data is None or shm is None:
        print(f"write fail !!!!")
        return
        
    payload = json.dumps(data).encode("utf-8")
    if len(payload) >= BUF_SIZE - 4:
        raise ValueError("The message is too large for the buffer")

    if payload is None:
        print(f"json dump encoded fail !!!!")
        return 
    try:
        # Store message length as a four-byte integer in the first four bytes.
        shm.buf[:4] = len(payload).to_bytes(4, "little")
        shm.buf[4:4+len(payload)] = payload
    except :
        print(f"SHM is DEAD and fail !!!!")
