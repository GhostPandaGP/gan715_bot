from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import logging
from urllib.parse import urljoin
import os
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.webhook import get_new_configured_app
from aiogram.bot import api
import asyncio
from aiohttp import web
import random
from time import sleep

from nns.style_transfer import SlowClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = '1353718289:AAFcZDTQrl4J1CPiGXUvIMzegd6HSMCyK-E'
PORT = 32100
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())
available_nn_names = ["style transfer", "gan"]

WEBHOOK_PATH = '/webhook/' + TOKEN
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = 32102

PROJECT_NAME = 'damp-earth-87185'

WEBHOOK_HOST = f'https://{PROJECT_NAME}.herokuapp.com'  # Enter here your link from Heroku project settings
WEBHOOK_URL_PATH = '/webhook/' + TOKEN
# WEBHOOK_URL = WEBHOOK_HOST + WEBHOOK_URL_PATH
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)


BASE_DIR = os.getcwd()
DESTINATION_USER_PHOTO = 'pytorch-CycleGAN-and-pix2pix/photo/'

PATH = {
    'style_transfer': 'data/style_transfer',
    'gan': 'data/gan'
}


class OrderFood(StatesGroup):
    waiting_for_nn_name = State()
    waiting_photo = State()
    waiting_second_photo = State()


@dp.message_handler(commands="nn", state="*")
async def food_step_1(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    for name in available_nn_names:
        keyboard.add(name)
    await message.answer("Выберите нейронку:", reply_markup=keyboard)
    await OrderFood.waiting_for_nn_name.set()


@dp.message_handler(state=OrderFood.waiting_for_nn_name, content_types=types.ContentTypes.TEXT)
async def food_step_2(message: types.Message, state: FSMContext):  # обратите внимание, есть второй аргумент
    if message.text.lower() not in available_nn_names:
        await message.reply("Пожалуйста, выберите нейронку, используя клавиатуру ниже.")
        return
    await state.update_data(chosen_nn=message.text.lower())

    await OrderFood.next()  # для простых шагов можно не указывать название состояния, обходясь next()
    await message.answer("Теперь отправьте основную фотографию:", reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=OrderFood.waiting_photo, content_types=types.ContentTypes.PHOTO)
async def food_step_3(message: types.Message, state: FSMContext):

    user_data = await state.get_data()
    logger.info(user_data)
    if user_data['chosen_nn'] == available_nn_names[0]:
        filename = 'content.jpg'
        destination = f"{PATH['style_transfer']}/{filename}"
        await message.photo[-1].download(destination=destination)
        await OrderFood.next()
        await message.answer(f"Отправьте, пожалуйста, вторую фотографию (фотографию стиля):")
    else:

        try:
            filename = 'photo.jpg'
            destination = DESTINATION_USER_PHOTO + filename
            os.system("bash pytorch-CycleGAN-and-pix2pix/scripts/download_cyclegan_model.sh horse2zebra")
            await bot.send_message(message.from_user.id, 'Фотография обрабатывается...')
            await message.photo[-1].download(destination=destination)
            os.system(
                "python pytorch-CycleGAN-and-pix2pix/test.py --dataroot 'pytorch-CycleGAN-and-pix2pix/photo' --name "
                "horse2zebra_pretrained --model test --no_dropout --gpu_ids -1")
            output_path = 'results/horse2zebra_pretrained/test_latest/images/photo_fake.png'
            with open(output_path, 'rb') as photo:
                await bot.send_photo(message.from_user.id, photo)
            os.remove(destination)
            os.remove(output_path)
        except Exception as e:
            await bot.send_message(message.from_user.id, f'🤒 Ошибка обработки фотографии: {e}')

        await message.answer(f"Это работает Gan")
        await message.answer(f"Введите команду /nn для того, чтобы посчитать новую фотографию.")
        await state.finish()


@dp.message_handler(state=OrderFood.waiting_second_photo, content_types=types.ContentTypes.PHOTO)
async def food_step_4(message: types.Message, state: FSMContext):

    filename = 'style.jpg'
    destination = f"{PATH['style_transfer']}/{filename}"
    await message.photo[-1].download(destination=destination)

    result = SlowClass()
    output = result.run("data/style_transfer/style.jpg",
                        "data/style_transfer/content.jpg")
    output_path = "data/style_transfer/result.jpg"
    result.save(output_path)
    with open(output_path, 'rb') as photo:
        await bot.send_photo(message.from_user.id, photo)
    await message.answer(f"Это работает Style Transfer")

    await message.answer(f"Введите команду /nn для того, чтобы посчитать новую фотографию.")
    await state.finish()


@dp.message_handler(commands=['help'])
async def send_menu(message: types.Message):
    """отправиь список команд бота"""
    await message.reply(
        text="""
        Это GAN715_Bot. Выберите одну из представленных команд:\n
        /start - приветсвенное сообщение;\n
        /help -- увидеть помощь;\n
        /nn -- выбрать тип нейронной сети (реализован медленный style transfer и gan, 
        преобразовывающий лошадь в зебру.
        """
    )


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    """отправиь список команд бота"""
    await message.reply("Доброго времени суток!\n"
                        "Я - GAN715_Bot!)\n"
                        "Введите команду /nn для начала работы!")
    # await send_menu(message=message)


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)
    # await state.set_state(TestStates.all()[int(argument)])


def main():
    # executor.start_polling(dp)
    executor.start_webhook(listen="0.0.0.0",
                           port=int(PORT),
                           url_path=TOKEN)
    executor.bot.setWebhook()


async def on_startup(dp):
    await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)
    # insert code here to run it after start


if __name__ == '__main__':
    app = get_new_configured_app(dispatcher=dp, path=WEBHOOK_URL_PATH)
    app.on_startup.append(on_startup)
    web.run_app(app, host='0.0.0.0', port=os.getenv('PORT'))  # Heroku stores port you have to listen in your app
    # executor.start_polling(dp)
