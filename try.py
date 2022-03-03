from BackTranslation import BackTranslation
trans = BackTranslation(url=[
      'translate.google.com',
      'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

result = trans.translate('hello, what are you doing now? class will start soon.', src='en', tmp = 'fr')
print("French:", result.result_text)

result = trans.translate('hello, what are you doing now? class will start soon.', src='en', tmp = 'es')
print("Spanish:", result.result_text)

result = trans.translate('hello, what are you doing now? class will start soon.', src='en', tmp = 'de')
print("German:", result.result_text)

result = trans.translate('hello, what are you doing now? class will start soon.', src='en', tmp = 'ru')
print("Russian:", result.result_text)

result = trans.translate('hello, what are you doing now? class will start soon.', src='en', tmp = 'it')
print("Italian:", result.result_text)

result = trans.translate('EETdE BTdB BUlB BLiB Walt Disney signed a contract with M.J. Winkler to produce a series of Alice Comedies , beginning the Disney company under its original name Disney Brothers Cartoon Studio , with brothers Walt and Roy Disney , as equal partners . EELiE EEUlE EETdE BTdB BUlB BLiB First Alice s comedy , Alice s Wonderland , was released EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1926 EETdE BTdB BUlB BLiB Disney Brothers Cartoon Studio changes name to The Walt Disney Studio shortly after moving into the new studio on Hyperion Avenue in the Silver Lake district . EELiE EEUlE EETdE BTdB EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1927 EETdE BTdB EETdE BTdB BUlB BLiB Oswald the Lucky Rabbit debuts . EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1928 EETdE BTdB BUlB BLiB Walt loses the Oswald s series contract . EELiE BLiB Walt and Roy come up with Mickey and Minnie Mouse . EELiE EEUlE EETdE BTdB BUlB BLiB Mickey Mouse debuts in Plane Crazy EELiE BLiB Steamboat Willie ( the first synchronized sound cartoon ) EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1929 EETdE BTdB BUlB BLiB On December 16 , The Walt Disney Studio is replaced by Walt Disney Productions , Ltd . Three other companies , Walt Disney Enterprises , Disney Film Recording Company , and Liled Realty and Investment Company , are also formed . EELiE EEUlE EETdE BTdB BUlB BLiB The Skeleton Dance ( the first Silly Symphonies cartoon ) EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1930 EETdE BTdB BUlB BLiB Distribution moved to Columbia Pictures EELiE EEUlE EETdE BTdB EETdE BTdB EETdE BTdB BUlB BLiB The Mickey Mouse comic strips by Floyd Gottfredson EELiE EEUlE EETdE EETrE BTrB BTdB 1931 EETdE BTdB EETdE BTdB EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1932 EETdE BTdB BUlB BLiB Distribution moved from Columbia Pictures to United Artists EELiE EEUlE EETdE BTdB BUlB BLiB Flowers and Trees ( the first Technicolor cartoon and first animated short to win the Academy Award for Best Animated Short Film of 1932 ) EELiE BLiB Mickey s Revue ( which features the premiere of Goofy , originally called Dippy Dawg ) EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1933 EETdE BTdB EETdE BTdB BUlB BLiB Three Little Pigs ( winner of Academy Award for Best Animated Short Film of 1933 ) EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1934 EETdE BTdB EETdE BTdB BUlB BLiB The Wise Little Hen ( which features the premiere of Donald Duck ) EELiE BLiB The Tortoise and the Hare ( Winner of Academy Award for Best Animated Short Film of 1934 ) EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1936 EETdE BTdB BUlB BLiB Distribution moved from United Artists to RKO Radio Pictures . EELiE EEUlE EETdE BTdB EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1937 EETdE BTdB BUlB BLiB Walt Disney s first feature film Snow White and the Seven Dwarfs was released EELiE EEUlE EETdE BTdB BUlB BLiB Snow White and the Seven Dwarfs EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1938 EETdE BTdB BUlB BLiB On September 29 , Walt Disney Enterprises , Disney Film Recording Company , and Liled Realty and Investment Company and Walt Disney Productions , Ltd . are merged to form Walt Disney Productions . EELiE EEUlE EETdE BTdB EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1940 EETdE BTdB BUlB BLiB Studio moves to Burbank , California EELiE BLiB Company goes public EELiE EEUlE EETdE BTdB BUlB BLiB Pinocchio EELiE BLiB Fantasia EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1941 EETdE BTdB BUlB BLiB A bitter animators strike occurs EELiE BLiB The studio begins making morale - boosting propaganda films for the United States during World War II EELiE EEUlE EETdE BTdB BUlB BLiB Dumbo EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1942 EETdE BTdB EETdE BTdB BUlB BLiB Bambi EELiE EEUlE EETdE BTdB EETdE BTdB BUlB BLiB Donald Duck comics by Carl Barks EELiE EEUlE EETdE EETrE BTrB BTdB 1943 EETdE BTdB EETdE BTdB BUlB BLiB Saludos Amigos EELiE EEUlE EETdE BTdB EETdE BTdB EETdE EETrE BTrB BTdB 1944 EETdE BTdB BUlB BLiB The company is short on money ; a theatrical re-release of Snow White and the Seven', src='en', tmp = 'es')
print("Chinese:", result.result_text)

# Chinese: Hello, what are you doing now?The class will start quickly.
# French: Hello, what are you doing now? Class will start soon.
# Spanish: Hello, what are you doing? The class will start soon.
# German: Hello, what are you doing right? Class will start soon.
# Russian: Hello, what are you doing now? The class will start soon.
# Italian: Hi, what are you doing now? The class will start soon.