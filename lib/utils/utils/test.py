from utils import send_mail, send_sms, notify
import time

#notify('jfn', how='mail', subject='exp subject', message='exp message')
notify('jfn', how='mail_async', subject='exp subject', message='exp message'); time.sleep(10)
#notify('jfn', how='sms', message='exp message')
