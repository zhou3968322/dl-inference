# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/5/12

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import unittest
import os, io
import glob
import json

from queue import Empty, Queue
from threading import Thread
import unittest
import time, logging
import requests, base64


class ProducerThread(Thread):

    def __init__(self, img_path_list, queue, sum_messages=100):
        super(ProducerThread, self).__init__()
        json_data_list = []
        for img_path in img_path_list:
            with open(img_path, "rb") as fr:
                base64_file = base64.b64encode(fr.read()).decode()
            json_data = {
                "image": base64_file,
                "img_path": img_path
            }
            json_data_list.append(json_data)
        self.json_data_list = json_data_list
        self._output = {}
        assert isinstance(queue, Queue)
        self.queue = queue
        self.sum_messages = sum_messages

    @property
    def result(self):
        return self._output

    def run(self) -> None:
        message_num = 0
        while True:
            for json_data in self.json_data_list:
                json_data.update({
                    "task_id": message_num
                })
                self.queue.put(json_data)
                message_num += 1
                print("produced message_num:{}".format(message_num))
                if message_num >= self.sum_messages:
                    return


class ConsumerThread(Thread):

    def __init__(self, queue, name="Consumer"):
        super(ConsumerThread, self).__init__(name=name)
        self.queue = queue
        self.name = name
        assert isinstance(self.queue, Queue)
        self.sum_time = 0
        self.handle_sum = 0
        self.url = "http://0.0.0.0:51000/predict"

    def run(self) -> None:
        while True:
            try:
                time.sleep(0.01)
                data = self.queue.get(timeout=10)
                data = data.copy()
                task_id = data.pop("task_id")
                img_path = data.pop("img_path")
                base64_file = data.pop("image")
                bt = time.time()
                response = requests.post(self.url, json={"image": base64_file})
                if response.status_code != 200:
                    print("failed to process:{}".format(img_path))
                assert response.status_code == 200
                inference_cost = time.time() - bt
                self.sum_time += inference_cost
                self.handle_sum += 1
                if self.handle_sum % 2 == 0:
                    print("{} handle file sum:{}, average cost:{}".format(self.name, self.handle_sum, self.sum_time / self.handle_sum))
                self.queue.task_done()
            except Empty:
                return


class TestOcrConsumer(unittest.TestCase):

    def setUp(self):
        pass

    def test_ocr_consumer(self):
        # file_dir = "/workspace/data/test_big_imgs"
        file_dir = "/workspace/data/all_document_imgs"
        # sum_messages = 200
        consumer_nums = 8
        img_path_list1 = glob.glob(os.path.join(file_dir, "*.png"))
        img_path_list2 = glob.glob(os.path.join(file_dir, "*.jpg"))
        img_path_list = img_path_list1 + img_path_list2
        import random
        random.shuffle(img_path_list)
        sum_messages = len(img_path_list)
        queue = Queue(sum_messages)
        producer = ProducerThread(img_path_list, queue=queue, sum_messages=sum_messages)
        producer.start()

        consumers = []

        for i in range(consumer_nums):
            name = "Consumer-{}".format(i)
            consumer = ConsumerThread(queue=queue, name=name)
            consumer.start()
            consumers.append(consumer)

        producer.join()
        print("all msg has produced")
        bt = time.time()
        for consumer in consumers:
            consumer.join()
        print("all cost:{}".format(time.time() - bt))


if __name__ == '__main__':
    unittest.main()

