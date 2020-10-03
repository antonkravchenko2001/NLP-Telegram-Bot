import os
import telebot
from model import predict, input_dict
from flask import Flask, request

server = Flask(__name__)
TOKEN = "1214344625:AAGtZ34OgYUOQSMLgwC9tiGkqhJIsJJ-1Bg"
bot = telebot.TeleBot(token=TOKEN)


def check_validity(message):
    for word in message.split():
        if word not in input_dict:
            return False
        return True


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Welcome!')


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, 'Type something')


@bot.message_handler(func=lambda msg: check_validity(msg.text))
def reply(message):
    prediction = predict(message.text)
    bot.reply_to(message, prediction)


@server.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://telegram-transformer-bot.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))




"""
while True:
    try:
        bot.polling(none_stop=True)
    except Exception:
        time.sleep(15)
"""