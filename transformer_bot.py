import os
from flask import Flask, request
import telebot
from model import predict, input_dict

TOKEN = '1214344625:AAGtZ34OgYUOQSMLgwC9tiGkqhJIsJJ-1Bg'
bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)


def validate(mess):
    for word in mess.split():
        if word not in input_dict.keys():
            return False
    return True


@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hi, I'm transformer bot!")


@bot.message_handler(func=lambda message: validate(message) == True)
def echo_message(message):
    bot.reply_to(message, predict(message.text))


@server.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://telegram-nlp-bot.herokuapp.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))