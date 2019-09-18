from utils.mail import send_mail, send_mail_async
from utils.sms import send_sms
from meta.address_book import address_book

def notify(who, how='mail', subject=None, message=None):
    if message is None:
        print('Nothing to notify')
        return
    if how == 'mail':
        to = address_book[who]['mail']
        send_mail(to, subject=subject, message=message)
    if how == 'mail_async':
        to = address_book[who]['mail']
        send_mail_async(to, subject=subject, message=message)
    if how == 'sms':
        to = address_book[who]['phone']
        send_sms(to, message)
