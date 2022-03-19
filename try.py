from BackTranslation import BackTranslation

trans = BackTranslation(
    url=[
        "translate.google.com",
        "translate.google.co.kr",
    ],
    proxies={"http": "127.0.0.1:1234", "http://host.name": "127.0.0.1:4012"},
)

result = trans.translate(
    "hello, what are you doing now? class will start soon.", src="en", tmp="fr"
)
print("French:", result.result_text)

result = trans.translate(
    "hello, what are you doing now? class will start soon.", src="en", tmp="es"
)
print("Spanish:", result.result_text)

result = trans.translate(
    "hello, what are you doing now? class will start soon.", src="en", tmp="de"
)
print("German:", result.result_text)

result = trans.translate(
    "hello, what are you doing now? class will start soon.", src="en", tmp="ru"
)
print("Russian:", result.result_text)

result = trans.translate(
    "hello, what are you doing now? class will start soon.", src="en", tmp="it"
)
print("Italian:", result.result_text)


result = trans.translate(
    "Memphis Grizzlies EETdE EETrE BTrB BTdB James Harden EETdE BTdB James Harden EETdE BTdB $28,299,399 EETdE BTdB Houston Rockets EETdE EETrE BTrB BTdB DeMar DeRozan EETdE BTdB $27,739,975 EETdE BTdB Toronto Raptors EETdE EETrE EETableE",
    src="en",
    tmp="zh-cn",
)
print("Chinese:", result.result_text)


result = trans.translate("EETdE BTdB $28,530,608 EETdE BTdB ", src="en", tmp="zh-cn")
print("Chinese:", result.result_text)

# combined does not work, has to be changed to es
result = trans.translate(
    "EETdE BTdB $28,530,608 EETdE BTdB Memphis Grizzlies EETdE EETrE BTrB BTdB James Harden EETdE BTdB $28,299,399 EETdE BTdB Houston Rockets EETdE EETrE BTrB BTdB DeMar DeRozan EETdE BTdB $27,739,975 EETdE BTdB Toronto Raptors EETdE EETrE EETableE",
    src="en",
    tmp="es",
)
print("Chinese:", result.result_text)

result = trans.translate(
    "EETdE BTdB $34,682,550 EETdE BTdB Golden State Warriors EETdE EETrE BTrB BTdB LeBron James EETdE BTdB $33,285,709 EETdE BTdB Cleveland Cavaliers EETdE EETrE BTrB BTdB Paul Millsap EETdE BTdB $31,269,231 EETdE BTdB Denver Nuggets EETdE EETrE BTrB BTdB Gordon Hayward EETdE BTdB $29,727,900 EETdE BTdB Boston Celtics EETdE EETrE BTrB BTdB Blake Griffin EETdE BTdB $29,512,900 EETdE BTdB Los Angeles Clippers EETdE EETrE BTrB BTdB Kyle Lowry EETdE BTdB $28,703,704 EETdE BTdB Toronto Raptors EETdE EETrE BTrB BTdB Russell Westbrook EETdE BTdB $28,530,608 EETdE BTdB Oklahoma City Thunder EETdE EETrE BTrB BTdB Mike Conley , Jr . EETdE BTdB $28,530,608 EETdE BTdB Memphis Grizzlies EETdE EETrE BTrB BTdB James Harden EETdE BTdB $28,299,399 EETdE BTdB Houston Rockets EETdE EETrE BTrB BTdB DeMar DeRozan EETdE BTdB $27,739,975 EETdE BTdB Toronto Raptors EETdE EETrE EETableE",
    src="en",
    tmp="fr",
)
print("Chinese:", result.result_text)

# Chinese: Hello, what are you doing now?The class will start quickly.
# French: Hello, what are you doing now? Class will start soon.
# Spanish: Hello, what are you doing? The class will start soon.
# German: Hello, what are you doing right? Class will start soon.
# Russian: Hello, what are you doing now? The class will start soon.
# Italian: Hi, what are you doing now? The class will start soon.
