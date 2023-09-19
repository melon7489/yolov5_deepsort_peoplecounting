# python 3.6
 
import random
import time
 
from paho.mqtt import client as mqtt_client
 
 
broker = '192.168.1.87'
port = 1883
topic = "/python/mqtt_template"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
 
 
def connect_mqtt():
    '''
    连接mqtt代理
    return mqtt客户端
    '''
    def on_connect(client, userdata, flags, rc):
        '''
        连接函数
        '''
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
 
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client
 

def publish(client):
    '''
    发布消息 1s发送1个序号
    @param client mqtt客户端
    '''
    msg_count = 0
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1

def publish_msg(client, msg):
    '''
    发布消息
    @param client mqtt客户端
    @param msg 发送的消息
    '''
    time.sleep(1)
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")
 
 
def run():
    client = connect_mqtt()
    for i in range(10):
        print(i)
        publish_msg(client, i)
    client.loop_start()
    
 
 
# if __name__ == '__main__':
    # run()