import os
import telebot
from flask import Flask, request
server = Flask(__name__)
TOKEN = "1214344625:AAGtZ34OgYUOQSMLgwC9tiGkqhJIsJJ-1Bg"
bot = telebot.TeleBot(token=TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Welcome!')


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, 'Type something')


@bot.message_handler(func=lambda msg: msg is not None)
def reply(message):
    bot.reply_to(message, message.text)


@server.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://aqueous-wave-71376.herokuapp.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))

