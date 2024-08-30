import socket
import cv2
import numpy as np
from run import run
import os
import json
import shutil

dir_name = ['cub200_1', 'cub200_2', 'aircraft_1', 'aircraft_2', 'food-101_1', 'food-101_2', 'nabirds_1', 'nabirds_2', 'vegfru_1', 'vegfru_2']
roots = ['../datasets/CUB_200_2011', 
        '../datasets/CUB_200_2011',
        '../datasets/aircraft',
        '../datasets/aircraft',
        '../datasets/food-101',
        '../datasets/food-101',
        '../datasets/nabirds',
        '../datasets/nabirds',
        '../datasets/vegfru',
        '../datasets/vegfru']

def receive():
    service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    service.bind(("10.101.128.69", 9090))
    service.listen(5)
    print("开始监听 10.101.128.69:9090...")

    while True:
        conn, addr = service.accept() 
        print('连接地址：', addr)
        image_name = conn.recv(1024)
        # byte to str
        image_name = image_name.decode("utf-8")
        service.shutdown(socket.SHUT_RDWR)
        service.close()
        return image_name

def get_retrieval_image_path(image_name, datasets, root, info, code_size):
    return run(image_name, datasets, root, info, code_size)

def send_retrieval_image_path(retrieval_image_path):
    json_string = json.dumps(retrieval_image_path)
    print(len(retrieval_image_path))
    print('json = ', json_string)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('10.101.110.137', 9091))
    
    client.send(json_string.encode('utf-8'))
    client.shutdown(socket.SHUT_RDWR)
    client.close()
    
def cpoy_images(retrieval_image_path, idx, origin_path):
    
    print(retrieval_image_path)

    newPath = os.path.join("test_images", dir_name[idx])
    os.makedirs(newPath, exist_ok=True)

    with open(os.path.join(newPath, dir_name[idx]+".txt"), 'w') as f:
        json_str = json.dumps(retrieval_image_path)
        f.write(json_str)
    print(origin_path)
    path = os.path.join(roots[idx], origin_path)
    shutil.copy(path, os.path.join(newPath, "query.jpg"))


    for path in retrieval_image_path:
        basename = os.path.basename(path)
        path = os.path.join(roots[idx], path)

        shutil.copy(path, os.path.join(newPath, basename))



    
if __name__ == '__main__':
    os.makedirs("test_images", exist_ok=True)
    # while True:
        # image_name = receive()
    '''
        cub-2011 - 239:"Crested_Auklet_0044_1825.jpg", 522 : "Red_Winged_Blackbird_0045_4526.jpg"
        aircraft - 1:"1514522.jpg" 707-320类   34:"2123466.jpg" SR-20类
        food101 - 21251:"1011328.jpg"  500:"994232.jpg"
        nabirds - 1:"05f0b9d1-de12-4a01-81be-8d6dc42a57f5.jpg" 类 938:"0623d563-cbb8-414c-ae51-b3427bf33475.jpg" 357类
        vegfru: - 1:v_12_01_0496.jpg 第20类蔬菜   2499 f_10_04_0151.jpg 201水果类

    '''
    idx = 9
    image_name = "f_10_04_0151.jpg"
    datasets = "vegfru"
    root = roots[idx]
    info = 'VegFru'
    code_size = [32]
                  
    retrieval_image_path, origin_path = get_retrieval_image_path(image_name, datasets, root, info, code_size)
    # print(retrieval_image_path)
    # send_retrieval_image_path(retrieval_image_path)
    cpoy_images(retrieval_image_path, idx, origin_path)
