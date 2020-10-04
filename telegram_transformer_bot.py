import time
import telebot
from model import predict, input_dict
TOKEN = "1214344625:AAGtZ34OgYUOQSMLgwC9tiGkqhJIsJJ-1Bg"
bot = telebot.TeleBot(token=TOKEN)
bot.delete_webhook()


def validate(mess):
    for word in mess.text.split():
        if word not in input_dict.keys():
            return False
    return True


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'hello')


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, 'type something')


@bot.message_handler(func=lambda msg: validate(msg))
def reply(message):
    bot.reply_to(message, predict(message.text))


while True:
    try:
        bot.polling(none_stop=True)
    except Exception:
        time.sleep(15)