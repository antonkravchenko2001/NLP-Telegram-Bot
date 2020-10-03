from model import predict, input_dict
import telebot
from flask import Flask, request
import os


server = Flask(__name__)
PORT = int(os.environ.get('PORT', 5000))


def check_validity(message):
    m = message.text.split()
    for word in m:
        if word not in input_dict.keys():
            return False
    return True


bot_token = '1214344625:AAGtZ34OgYUOQSMLgwC9tiGkqhJIsJJ-1Bg'
bot = telebot.TeleBot(token=bot_token)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Welcome!')


@bot.message_handler(func=lambda mess: check_validity(mess) == True)
def reply(message):
    bot.reply_to(message, predict(message.text))


@server.route('/' + bot_token, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!" "200"


@server.route('/')
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://yourherokuappname.herokuapp.com/' + bot_token)
    return "!" "200"


if __name__ == '__main__':
    server.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))



