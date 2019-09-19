import json
from utils.mail import send_mail, send_mail_async
from utils.sms import send_sms


default_address_book = {
    'jneto': {'phone': '+351966221506', 'mail': 'joao.filipe.neto@gmail.com'},
    'jranito': {'phone': '+351966221505', 'mail': 'joao.vasco.ranito@gmail.com '},
}


def read_address_book():
    try:
        with open('./.addressbook') as f:
            d = json.loads(f.read())
    except IOError:
        return None
    return d


def notify(who, how='mail', subject=None, message=None):
    if address_book is None:
        address_book = read_address_book()
        if address_book is None:
            address_book = default_address_book
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
