import asyncore
import socket
import queue
import threading

import cv2
from PIL import Image
import numpy as np

import struct
import datetime
from ast import literal_eval

import torch.multiprocessing as mp

import time
import redis


PRIORITYS = {
    "COMMAND": 1,
    "AIMG": 2,
    "INFO": 3,
    "NULL": 10,
    "OTHER": 100,
}

STATUS_CLOSED      = 'closed'
STATUS_INIT        = 'initializing'
STATUS_CONNECTED   = 'connected'
STATUS_NETERROR    = 'net_error'
STATUS_ERROR       = 'error'


class STMClient(asyncore.dispatcher):
    def __init__(self, host, port, buffer_queue, out_queue, stop_queue):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((host, port))

        self.recv_buffer = b""
        self.BUFFER_SIZE = 4096000

        self.buffer_queue = buffer_queue
        self.out_queue = out_queue
        self.stop_queue = stop_queue
        self.redis_pool = redis.ConnectionPool(host='172.17.0.1', port=6379, decode_responses=True)
        self.app_redis = redis.Redis(connection_pool=self.redis_pool)

    def handle_connect(self):
        pass

    def handle_error(self):
        self.app_redis.set('status', STATUS_NETERROR)
        print('**net error**')

    def handle_close(self):
        self.close()

    def handle_read(self):
        recv_buffer = self.recv(self.BUFFER_SIZE)
        self.buffer_queue.put(recv_buffer)

    def writable(self):
        return (self.out_queue.qsize() > 0)

    def handle_write(self):
        try:
            a_result = self.out_queue.get()
            self.send(a_result)
            print('send out message')
        except:
            print("send out_queue error")

    def run(self):
        asyncore.loop(0.5)
        self.stop_queue.get()


class STMWorker(mp.Process):
    def __init__(self, host, port, buffer_queue, out_queue, stop_queue):
        super(STMWorker, self).__init__()
        print("assigned queue")
        self.host = host
        self.port = port
        self.buffer_queue = buffer_queue
        self.out_queue = out_queue
        self.stop_queue = stop_queue

    def run(self):
        self.client = STMClient(self.host, self.port, self.buffer_queue, self.out_queue, self.stop_queue)
        print("start client")
        self.client.run()
        print("end client")


class BatchWorker(mp.Process):
    def __init__(self, img_queue, batch_queue, batch_size, stop_queue, clear_flag):
        super(BatchWorker, self).__init__()
        self.img_queue = img_queue
        self.batch_queue = batch_queue
        self.batch_size = batch_size
        self.stop_queue = stop_queue
        self.clear_flag = clear_flag

    def run(self):
        waiting_array = []
        while not self.stop_queue.empty():
            if self.clear_flag.value > 0:
                waiting_array = []
                while not self.img_queue.empty():
                    try:
                        self.img_queue.get_nowait()
                    except:
                        pass
                while not self.batch_queue.empty():
                    try:
                        self.batch_queue.get_nowait()
                    except:
                        pass
                continue
            try:
                a_img = self.img_queue.get_nowait()
                if a_img is not None:
                    waiting_array.append(a_img)

                    if len(waiting_array) == self.batch_size:
                        try:
                            imgs = np.array(waiting_array)
                            waiting_array = []
                            self.batch_queue.put(imgs)
                        except:
                            print("batch_queue put error.")
                else:
                    print("cannot handle a img")
            except:
                pass
        print(" batch worker stopped")
        self.img_queue.cancel_join_thread()
        self.batch_queue.cancel_join_thread()



class BufferWorker(mp.Process):
    def __init__(self, buffer_queue, img_queue, out_queue, stop_queue, clear_flag):
        super(BufferWorker, self).__init__()
        self.buffer_queue = buffer_queue
        self.img_queue = img_queue
        self.out_queue = out_queue
        self.stop_queue = stop_queue
        self.clear_flag = clear_flag

        self.recv_buffer = b""
        self.remain = 0
        self.remain_buffer = b""
        self.recv_index = 0
        self.construct_datasize = False # construct datasize

        self.MetaData = ['start M', 'Stop', 'Acq Data', 'Time', 'Image info', 'Image back']
        self.meta_defined = True
        self.s = 0


    def run(self):
        while not self.stop_queue.empty():
            try:
                if self.remain == 0:
                    a_buffer = self.buffer_queue.get_nowait()
                    if len(a_buffer) < 4:
                        self.remain = len(a_buffer) - 4
                        self.construct_datasize = True
                        self.remain_buffer = a_buffer
                        continue
                    datasize = int.from_bytes(a_buffer[:4], 'big') + 4
                elif self.remain > 0:
                    a_buffer = self.remain_buffer
                    if len(a_buffer) < 4:
                        self.remain = len(a_buffer) - 4
                        self.construct_datasize = True
                        continue
                    datasize = int.from_bytes(a_buffer[:4], 'big') + 4
                else:
                    a_buffer = self.buffer_queue.get_nowait()
                    if self.construct_datasize:
                        a_buffer = self.remain_buffer + a_buffer
                        if len(a_buffer) < 4:
                            self.remain = len(a_buffer) - 4
                            self.remain_buffer = a_buffer
                            continue
                        self.construct_datasize = False
                        datasize = int.from_bytes(a_buffer[:4], 'big') + 4
                    else:
                        datasize = abs(self.remain)

                diff = len(a_buffer) - datasize
                if diff == 0:
                    if self.remain < 0:
                        self.recv_buffer = self.recv_buffer + a_buffer
                    else:
                        self.recv_buffer = a_buffer
                    self.process()
                    self.recv_buffer = b""
                elif diff > 0:
                    if self.remain < 0:
                        self.recv_buffer = self.recv_buffer + a_buffer[:datasize]
                    else:
                        self.recv_buffer = a_buffer[:datasize]
                    self.process()
                    self.remain_buffer = a_buffer[datasize:]
                elif diff < 0:
                    if self.remain < 0:
                        self.recv_buffer = self.recv_buffer + a_buffer
                    else:
                        self.recv_buffer = a_buffer
                else:
                    print("Fatal error: out of control")
                self.remain = diff

            except queue.Empty:
                continue
            except queue.Full:
                print("queue Full")
            except:
                print("another except")
                print(self.remain, len(a_buffer), datasize)
                self.remain = 0
                self.recv_buffer = b""

    def process(self):
        if not self.meta_defined:
            print("meta data length", len(self.recv_buffer))
            datasize = int.from_bytes(self.recv_buffer[:4], 'big')
            meta_buffer = self.recv_buffer[4:]
            meta_len = int.from_bytes(meta_buffer[:4], 'big')
            print("meta nums: ", meta_len)

            placeholder = 4
            for i in range(0, meta_len):
                m_ele_len = int.from_bytes(meta_buffer[placeholder: placeholder+4], 'big')
                placeholder = placeholder + 4
                a_name = meta_buffer[placeholder: placeholder +
                                     m_ele_len].decode()
                self.MetaData.append(a_name)
                placeholder = placeholder + m_ele_len
                self.meta_defined = True

                self.s = time.time()
            print(self.MetaData)
        else:
            recv_buf = self.recv_buffer[:4]
            datasize = int.from_bytes(recv_buf, 'big')
            recv_buf = self.recv_buffer[4:]
            meta_id = int.from_bytes(recv_buf[:2], 'big')
            data_buffer = recv_buf[2:]
            element = self.MetaData[meta_id]
            if element == "Acq Data":
                # print("Acq Data")
                self.recv_index = self.recv_index + 1
                # print(self.recv_index)
                try:
                    image = cv2.imdecode(np.frombuffer(data_buffer, np.uint8), -1)
                    # print(f'image: {image.shape}')
                    self.img_queue.put(image)
                except:
                    print("error process:", self.recv_index)
                finally:
                    pass
            elif element == "start M":
                a = int.from_bytes(data_buffer, 'big')
                # self.in_messages.put((PRIORITYS["COMMAND"], ("start M", a)))
                print("start M", a)
                sss = time.time()
                with self.clear_flag.get_lock():
                    self.clear_flag.value = 1
                print(">>>>>>>>>clear_flag start")
                while True:
                    if self.clear_flag.value == 0:
                        break
                eee = time.time()
                print(f"=========================clear_flag success: {(eee-sss)*1000}ms")
            elif element == "Stop":
                a = int.from_bytes(data_buffer, 'big')
                print("stop", a)
                # self.in_messages.put((PRIORITYS["COMMAND"], ("Stop", a)))
            elif element == "Time":
                try:
                    timestamp = struct.unpack('>QQ', data_buffer)[0]
                    result = datetime.datetime.fromtimestamp(timestamp - 2082844800).strftime('%Y-%m-%d %H:%M:%S')
                    # print(result)
                except:
                    print(len(data_buffer))
            else:
                print('unrecognize data length:', len(data_buffer))


