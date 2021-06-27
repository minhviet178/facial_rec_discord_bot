import discord 
import os
import requests
import json
import random
import os
import time
import cv2

DISCORD_TOKEN = 'ODU2Mzg2MTAwMDc1MTY3NzU0.YNARtQ.v1NrKqbPGVl4efRxtU6G0gDbeTI'

client = discord.Client()

# 50 shades of Obok
sad_words = ['sad', 'depressed', 'unhappy', 'miserable', 'angry', 'depressing', 'crying']
starter_encouragements = [
    'Come on! You can do it!',
    'Stay strong!',
    'Go buy yourself an ice cream O.O',
    "I swear I did not eat the leftover fish. Please don't be sad O.O"
]
games_list = 'For games I have these suggestions:\n - Try Gartic! A simple but fun drawing and guessing game\n https://gartic.io/ \n - Try Kahoot! A great quiz maker for online study during the Covid-19 situation :( \n https://kahoot.com/ \n - Try Khan Academy! If you want crippling depression and more Indian tutors xD \n https://www.khanacademy.org/'
intro = "My name is Obok and I'm from Korea. You can follow me at https://www.instagram.com/obok_dabok/ I'll make sure to buy you a cup of Starbucks next time we meet O.O"
help = 'Here are a list of what I can do: \n -register: Register as a new user! \n -hello: Greetings \n -games: Suggest what games you guys could play \n -intro: Introduce myself! \n -clear: Clear all of my previous messages on this text channel!'

def get_quote():
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + " - " + json_data[0]['a']
    return(quote)

clear = {}
checked_in = []

@client.event
async def on_ready():
    welcome = client.get_channel(857112102778306570)
    print('We have logged in as {0.user}'.format(client))

    for guild in client.guilds:
        if guild.name == os.getenv('DISCORD_GUILD'):
            break

    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')

    for i in range(3):
        path = 'D:\Study\CoderSchool\Final Project\Facial_Recognition-master\Facial_Recognition-master'
        f = open (os.path.join(path,'student.json'), "r")
        attendance = json.loads(f.read())

        for name in attendance.keys() :
            if name not in checked_in:
                await welcome.send(f'Hello {name}')
                checked_in.append(name)
        with open("student.json", "w") as write_file:
            json.dump(clear, write_file, indent=4)
        time.sleep(2)

    print('Facial Recognition Ends')



@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = message.content

    if msg.startswith('-hello'):
        print('Say Hello!')
        await message.channel.send("Hello! My name is Obok and I'm here to help!")
        await message.channel.send(file = discord.File('204119981_2997591130479686_3364082421680985816_n.jpg'))

    if msg.startswith('-games'):
        print('Game Time!')
        await message.channel.send(games_list)

    if msg.startswith('-clear'):
        print("clear intent")
        history = await message.channel.history(limit = 1000).flatten()
        await message.delete()
        for m in history:
            if (m.author.name == 'Obok'):
                try:
                    await m.delete()
                except:
                    pass

    if msg.startswith('-intro'):
        print('Intro Time!')
        await message.channel.send(intro)
        await message.channel.send(file = discord.File('205425125_540335477094778_3873316618434549859_n.jpg'))
    
    if msg.startswith('-help'):
        print('Help!')
        await message.channel.send(help)

    # Inspire users

    if any(word in msg for word in sad_words):
        await message.channel.send(random.choice(starter_encouragements))

    if msg.startswith('-register'):
        await message.channel.send('Visit this link to register as a new student! http://localhost:8501/')
    
print('Running Discord!')
client.run(DISCORD_TOKEN)