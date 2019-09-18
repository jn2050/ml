from utils.env import ENV
import boto3


MessageAttributes={
    'AWS.SNS.SMS.SenderID': {
        'DataType': 'String',
        'StringValue': 'DLbot'
    }
}


class SMS():
    def __init__(self):
        self.client = boto3.client(
            'sns',
            aws_access_key_id=ENV.get('aws_access_key_id'),
            aws_secret_access_key=ENV.get('aws_secret_access_key'),
            region_name="eu-west-1"
        )

    def send(self, to, msg):
        self.client.publish(PhoneNumber=to, Message=msg, MessageAttributes=MessageAttributes)


sms = SMS()


def send_sms(to, msg):
    sms.send(to, msg)
