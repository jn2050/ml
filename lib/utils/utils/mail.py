import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from os.path import basename
from multiprocessing import Process
from utils.env import ENV
import sys

#https://realpython.com/python-send-email/


def send_mail_async(to, subject=None, message=None, html=None, files=[]):
    p = Process(target=send_mail, args=(to, subject, message, html, files), daemon=True)
    p.start()


def send_mail(to, subject=None, message=None, html=None, files=[]):
    smtp_server = 'smtp.gmail.com'
    port = 465
    user = ENV.get('EMAIL_BOT_USER')
    password = ENV.get('EMAIL_BOT_PASSWORD')
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject if subject is not None else ''
    if not type(to) is list:
        to = [to]
    msg['To'] = ', '.join(to)
    body = MIMEText(message, "plain") if html is None else MIMEText(html, 'html')
    msg.attach(body)
    for file in files:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(open(file, "rb").read())
        encoders.encode_base64(part)
        arg = 'attachment; filename="' + basename(file) + '"'
        part.add_header('Content-Disposition', arg)
        msg.attach(part)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        try:
            res = server.login(user, password)
            with open('/Users/jneto/dev/proj/lib/python/utils/output.txt', 'w') as f:
                f.write(f'\nIN 2: {res}\n') 
            res = server.sendmail(user, to, msg.as_string())
        except Exception as e:
            print(f'MAIL SENDER ERROR: {e}')
