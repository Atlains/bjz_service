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


PRIORITYS = {
    "COMMAND": 1,
    "AIMG": 2,
    "INFO": 3,
    "NULL": 10,
    "OTHER": 100,
}


class STMClient(asyncore.dispatcher):
    def __init__(self, host, port, img_queue, out_queue):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((host, port))

        self.MetaData = []
        self.meta_defined = False
        self.recv_buffer = b""
        self.BUFFER_SIZE = 4096000

        self.in_messages = queue.Queue()
        self.in_buffers = queue.Queue()
        self.in_arrays = []
        self.remain = 0
        self.remain_buffer = b""
        self.parse_remain = False


        self.out_messages = queue.Queue()
        self.out_buffer = b""
        self.current_image = 0
        self.recv_index = 0
        self.stop = False
        self.start_measure = 0

        self.img_queue = img_queue
        self.out_queue = out_queue

    def handle_connect(self):
        pass

    def handle_close(self):
        self.close()
        print("**connection closed**")

    def handle_read(self):
        recv_buffer = self.recv(self.BUFFER_SIZE)
        self.in_buffers.put(recv_buffer)


    def writable(self):
        return (self.out_messages.qsize() > 0)

    def getIDfromName(self, name):
        if name in self.MetaData:
            return self.MetaData.index(name)
        return -1

    def handle_write(self):
        try:
            priority, a_out_msg = self.out_messages.get_nowait()
            if priority == PRIORITYS['INFO']:
                meta_id = self.getIDfromName("Image info")
                meta_id_bytes = meta_id.to_bytes(2, 'big')
                data_len = len(a_out_msg) + 2
                data_len_bytes = data_len.to_bytes(4, 'big')
                self.out_buffer = data_len_bytes + meta_id_bytes + a_out_msg
                self.send(self.out_buffer)
        except:
            pass

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
        else:
            recv_buf = self.recv_buffer[:4]
            datasize = int.from_bytes(recv_buf, 'big')
            recv_buf = self.recv_buffer[4:]
            meta_id = int.from_bytes(recv_buf[:2], 'big')
            data_buffer = recv_buf[2:]
            element = self.MetaData[meta_id]
            if element == "Acq Data":
                self.recv_index = self.recv_index + 1
                try:
                    self.in_messages.put((PRIORITYS["AIMG"], data_buffer))
                    # print("put a img")
                except:
                    print("error process:", self.recv_index)
                finally:
                    pass
            elif element == "start M":
                a = int.from_bytes(data_buffer, 'big')
                self.in_messages.put((PRIORITYS["COMMAND"], ("start M", a)))
            elif element == "Stop":
                a = int.from_bytes(data_buffer, 'big')
                self.in_messages.put((PRIORITYS["COMMAND"], ("Stop", a)))
            elif element == "Time":
                try:
                    timestamp = struct.unpack('>d', data_buffer)[0]
                    result = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    print(len(data_buffer))
            else:
                print('unrecognize data length:', len(data_buffer))

    def parse_buffer(self):
        while not self.stop:
            try:
                if self.remain == 0:
                    a_buffer = self.in_buffers.get_nowait()
                    datasize = int.from_bytes(a_buffer[:4], 'big') + 4
                elif self.remain > 0:
                    a_buffer = self.remain_buffer
                    datasize = int.from_bytes(a_buffer[:4], 'big') + 4
                else:
                    a_buffer = self.in_buffers.get_nowait()
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



    def work_thread(self):
        while not self.stop:
            try:
                priority, a_in_msg = self.in_messages.get_nowait()
                if priority == PRIORITYS['AIMG']:
                    self.current_image = self.current_image + 1
                    data_buffer = a_in_msg

                    image = cv2.imdecode(np.frombuffer(data_buffer, np.uint8), -1)
                    image = np.array(image)
                    # cv2.imwrite("%d.png" % self.current_image, image)

                    self.img_queue.put(image)

                elif priority == PRIORITYS["COMMAND"]:
                    cmd, value = a_in_msg
                    print(cmd, value)
                    if cmd == "Stop" and value:
                        self.stop = True
                else:
                    print('no default handle for priority', priority)
            except:
                pass
            try:
                if not self.out_queue.empty():
                    a_result = self.out_queue.get()
                    # print("get results: ", a_result)
                    self.out_messages.put((PRIORITYS["INFO"], a_result.encode()))
            except:
                pass

    def run(self):
        try:
            buffering_thread = threading.Thread(target=self.parse_buffer)
            working_thread = threading.Thread(target=self.work_thread)
            buffering_thread.start()
            working_thread.start()
            print("here3")
            asyncore.loop(0.5)
            print("here4")
            self.stop = True
            working_thread.join()
            print("working thread closed")
            # buffering_thread.join()
            print("buffering thread closed")
        except:
            self.stop = True
        finally:
            self.stop = True

class STMWorker(mp.Process):
    def __init__(self, host, port, img_queue, out_queue, stop_queue):
        super(STMWorker, self).__init__()
        print("assigned queue")
        self.host = host
        self.port = port
        self.img_queue = img_queue
        self.out_queue = out_queue
        self.stop_queue = stop_queue

    def run(self):
        self.client = STMClient(self.host, self.port, self.img_queue, self.out_queue)
        print("initialize client")
        print("start client")
        self.client.run()

        self.stop_queue.get()
        print("stop_queue empty: ", self.stop_queue.empty())
        print("end client")


class BatchWorker(mp.Process):
    def __init__(self, img_queue, batch_queue, batch_size, stop_queue):
        super(BatchWorker, self).__init__()
        self.img_queue = img_queue
        self.batch_queue = batch_queue
        self.batch_size = batch_size
        self.stop_queue = stop_queue

    def run(self):
        waiting_array = []
        while not self.stop_queue.empty():
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


