import os

#Different dictionaries to convert the JSON data to the correct format
regions_dict_acov19dag = {'00': 'Riket', '10': 'Blekinge', '20': 'Dalarna', '09': 'Gotland', '21': 'Gävleborg','13': 'Halland', '23': 'Jämtland', '06': 'Jönköping', '08': 'Kalmar', '07': 'Kronoberg', '25': 'Norrbotten', '12': 'Skåne', '01': 'Stockholm', '04': 'Södermanland', '03': 'Uppsala', '17': 'Värmland', '24': 'Västerbotten', '22': 'Västernorrland', '19': 'Västmanland', '14': 'Västra Götaland', '18': 'Örebro', '05': 'Östergötland', "99":"Uppgift om län saknas"}
#ccov19kon had different regions for halland
regions_dict_PCRtestVAr = {"00":"00 Riket","01":"01 Stockholm","03":"03 Uppsala","04":"04 Södermanland","05":"05 Östergötland","06":"06 Jönköping","07":"07 Kronoberg","08":"08 Kalmar","09":"09 Gotland","10":"10 Blekinge","12":"12 Skåne","13":"13 Halland","14":"14 Västra Götaland","17":"17 Värmland","18":"18 Örebro","19":"19 Västmanland","20":"20 Dalarna","21":"21 Gävleborg","22":"22 Västernorrland","23":"23 Jämtland","24":"24 Västerbotten","25":"25 Norrbotten","99":"99 Uppgift om län saknas"}
regions_dict_according_ccov19kon = {'00': 'Riket', '10': 'Blekinge', '20': 'Dalarna', '09': 'Gotland', '21': 'Gävleborg',"11":"Halland", '23': 'Jämtland', '06': 'Jönköping', '08': 'Kalmar', '07': 'Kronoberg', '25': 'Norrbotten', '12': 'Skåne', '01': 'Stockholm', '04': 'Södermanland', '03': 'Uppsala', '17': 'Värmland', '24': 'Västerbotten', '22': 'Västernorrland', '19': 'Västmanland', '14': 'Västra Götaland', '18': 'Örebro', '05': 'Östergötland', "99":"Uppgift om län saknas"} 
kommun_dict = {'0114': '0114 Upplands Väsby', '0115': '0115 Vallentuna', '0117': '0117 Österåker', '0120': '0120 Värmdö', '0123': '0123 Järfälla', '0125': '0125 Ekerö', '0126': '0126 Huddinge', '0127': '0127 Botkyrka', '0128': '0128 Salem', '0136': '0136 Haninge', '0138': '0138 Tyresö', '0139': '0139 Upplands-Bro', '0140': '0140 Nykvarn', '0160': '0160 Täby', '0162': '0162 Danderyd', '0163': '0163 Sollentuna', '0180': '0180 Stockholm', '0181': '0181 Södertälje', '0182': '0182 Nacka', '0183': '0183 Sundbyberg', '0184': '0184 Solna', '0186': '0186 Lidingö', '0187': '0187 Vaxholm', '0188': '0188 Norrtälje', '0191': '0191 Sigtuna', '0192': '0192 Nynäshamn', '0305': '0305 Håbo', '0319': '0319 Älvkarleby', '0330': '0330 Knivsta', '0331': '0331 Heby', '0360': '0360 Tierp', '0380': '0380 Uppsala', '0381': '0381 Enköping', '0382': '0382 Östhammar', '0428': '0428 Vingåker', '0461': '0461 Gnesta', '0480': '0480 Nyköping', '0481': '0481 Oxelösund', '0482': '0482 Flen', '0483': '0483 Katrineholm', '0484': '0484 Eskilstuna', '0486': '0486 Strängnäs', '0488': '0488 Trosa', '0509': '0509 Ödeshög', '0512': '0512 Ydre', '0513': '0513 Kinda', '0560': '0560 Boxholm', '0561': '0561 Åtvidaberg', '0562': '0562 Finspång', '0563': '0563 Valdemarsvik', '0580': '0580 Linköping', '0581': '0581 Norrköping', '0582': '0582 Söderköping', '0583': '0583 Motala', '0584': '0584 Vadstena', '0586': '0586 Mjölby', '0604': '0604 Aneby', '0617': '0617 Gnosjö', '0642': '0642 Mullsjö', '0643': '0643 Habo', '0662': '0662 Gislaved', '0665': '0665 Vaggeryd', '0680': '0680 Jönköping', '0682': '0682 Nässjö', '0683': '0683 Värnamo', '0684': '0684 Sävsjö', '0685': '0685 Vetlanda', '0686': '0686 Eksjö', '0687': '0687 Tranås', '0760': '0760 Uppvidinge', '0761': '0761 Lessebo', '0763': '0763 Tingsryd', '0764': '0764 Alvesta', '0765': '0765 Älmhult', '0767': '0767 Markaryd', '0780': '0780 Växjö', '0781': '0781 Ljungby', '0821': '0821 Högsby', '0834': '0834 Torsås', '0840': '0840 Mörbylånga', '0860': '0860 Hultsfred', '0861': '0861 Mönsterås', '0862': '0862 Emmaboda', '0880': '0880 Kalmar', '0881': '0881 Nybro', '0882': '0882 Oskarshamn', '0883': '0883 Västervik', '0884': '0884 Vimmerby', '0885': '0885 Borgholm', '0980': '0980 Gotland', '1060': '1060 Olofström', '1080': '1080 Karlskrona', '1081': '1081 Ronneby', '1082': '1082 Karlshamn', '1083': '1083 Sölvesborg', '1214': '1214 Svalöv', '1230': '1230 Staffanstorp', '1231': '1231 Burlöv', '1233': '1233 Vellinge', '1256': '1256 Östra Göinge', '1257': '1257 Örkelljunga', '1260': '1260 Bjuv', '1261': '1261 Kävlinge', '1262': '1262 Lomma', '1263': '1263 Svedala', '1264': '1264 Skurup', '1265': '1265 Sjöbo', '1266': '1266 Hörby', '1267': '1267 Höör', '1270': '1270 Tomelilla', '1272': '1272 Bromölla', '1273': '1273 Osby', '1275': '1275 Perstorp', '1276': '1276 Klippan', '1277': '1277 Åstorp', '1278': '1278 Båstad', '1280': '1280 Malmö', '1281': '1281 Lund', '1282': '1282 Landskrona', '1283': '1283 Helsingborg', '1284': '1284 Höganäs', '1285': '1285 Eslöv', '1286': '1286 Ystad', '1287': '1287 Trelleborg', '1290': '1290 Kristianstad', '1291': '1291 Simrishamn', '1292': '1292 Ängelholm', '1293': '1293 Hässleholm', '1315': '1315 Hylte', '1380': '1380 Halmstad', '1381': '1381 Laholm', '1382': '1382 Falkenberg', '1383': '1383 Varberg', '1384': '1384 Kungsbacka', '1401': '1401 Härryda', '1402': '1402 Partille', '1407': '1407 Öckerö', '1415': '1415 Stenungsund', '1419': '1419 Tjörn', '1421': '1421 Orust', '1427': '1427 Sotenäs', '1430': '1430 Munkedal', '1435': '1435 Tanum', '1438': '1438 Dals-Ed', '1439': '1439 Färgelanda', '1440': '1440 Ale', '1441': '1441 Lerum', '1442': '1442 Vårgårda', '1443': '1443 Bollebygd', '1444': '1444 Grästorp', '1445': '1445 Essunga', '1446': '1446 Karlsborg', '1447': '1447 Gullspång', '1452': '1452 Tranemo', '1460': '1460 Bengtsfors', '1461': '1461 Mellerud', '1462': '1462 Lilla Edet', '1463': '1463 Mark', '1465': '1465 Svenljunga', '1466': '1466 Herrljunga', '1470': '1470 Vara', '1471': '1471 Götene', '1472': '1472 Tibro', '1473': '1473 Töreboda', '1480': '1480 Göteborg', '1481': '1481 Mölndal', '1482': '1482 Kungälv', '1484': '1484 Lysekil', '1485': '1485 Uddevalla', '1486': '1486 Strömstad', '1487': '1487 Vänersborg', '1488': '1488 Trollhättan', '1489': '1489 Alingsås', '1490': '1490 Borås', '1491': '1491 Ulricehamn', '1492': '1492 Åmål', '1493': '1493 Mariestad', '1494': '1494 Lidköping', '1495': '1495 Skara', '1496': '1496 Skövde', '1497': '1497 Hjo', '1498': '1498 Tidaholm', '1499': '1499 Falköping', '1715': '1715 Kil', '1730': '1730 Eda', '1737': '1737 Torsby', '1760': '1760 Storfors', '1761': '1761 Hammarö', '1762': '1762 Munkfors', '1763': '1763 Forshaga', '1764': '1764 Grums', '1765': '1765 Årjäng', '1766': '1766 Sunne', '1780': '1780 Karlstad', '1781': '1781 Kristinehamn', '1782': '1782 Filipstad', '1783': '1783 Hagfors', '1784': '1784 Arvika', '1785': '1785 Säffle', '1814': '1814 Lekeberg', '1860': '1860 Laxå', '1861': '1861 Hallsberg', '1862': '1862 Degerfors', '1863': '1863 Hällefors', '1864': '1864 Ljusnarsberg', '1880': '1880 Örebro', '1881': '1881 Kumla', '1882': '1882 Askersund', '1883': '1883 Karlskoga', '1884': '1884 Nora', '1885': '1885 Lindesberg', '1904': '1904 Skinnskatteberg', '1907': '1907 Surahammar', '1960': '1960 Kungsör', '1961': '1961 Hallstahammar', '1962': '1962 Norberg', '1980': '1980 Västerås', '1981': '1981 Sala', '1982': '1982 Fagersta', '1983': '1983 Köping', '1984': '1984 Arboga', '2021': '2021 Vansbro', '2023': '2023 Malung-Sälen', '2026': '2026 Gagnef', '2029': '2029 Leksand', '2031': '2031 Rättvik', '2034': '2034 Orsa', '2039': '2039 Älvdalen', '2061': '2061 Smedjebacken', '2062': '2062 Mora', '2080': '2080 Falun', '2081': '2081 Borlänge', '2082': '2082 Säter', '2083': '2083 Hedemora', '2084': '2084 Avesta', '2085': '2085 Ludvika', '2101': '2101 Ockelbo', '2104': '2104 Hofors', '2121': '2121 Ovanåker', '2132': '2132 Nordanstig', '2161': '2161 Ljusdal', '2180': '2180 Gävle', '2181': '2181 Sandviken', '2182': '2182 Söderhamn', '2183': '2183 Bollnäs', '2184': '2184 Hudiksvall', '2260': '2260 Ånge', '2262': '2262 Timrå', '2280': '2280 Härnösand', '2281': '2281 Sundsvall', '2282': '2282 Kramfors', '2283': '2283 Sollefteå', '2284': '2284 Örnsköldsvik', '2303': '2303 Ragunda', '2305': '2305 Bräcke', '2309': '2309 Krokom', '2313': '2313 Strömsund', '2321': '2321 Åre', '2326': '2326 Berg', '2361': '2361 Härjedalen', '2380': '2380 Östersund', '2401': '2401 Nordmaling', '2403': '2403 Bjurholm', '2404': '2404 Vindeln', '2409': '2409 Robertsfors', '2417': '2417 Norsjö', '2418': '2418 Malå', '2421': '2421 Storuman', '2422': '2422 Sorsele', '2425': '2425 Dorotea', '2460': '2460 Vännäs', '2462': '2462 Vilhelmina', '2463': '2463 Åsele', '2480': '2480 Umeå', '2481': '2481 Lycksele', '2482': '2482 Skellefteå', '2505': '2505 Arvidsjaur', '2506': '2506 Arjeplog', '2510': '2510 Jokkmokk', '2513': '2513 Överkalix', '2514': '2514 Kalix', '2518': '2518 Övertorneå', '2521': '2521 Pajala', '2523': '2523 Gällivare', '2560': '2560 Älvsbyn', '2580': '2580 Luleå', '2581': '2581 Piteå', '2582': '2582 Boden', '2583': '2583 Haparanda', '2584': '2584 Kiruna', '9999': '9999 Okänd'}
indicator_bcov19kom_dict = {'1': 'Antal fall', '2': 'Antal fall per 10 000 inv'}
indicator_ccov19kon_dict = {'1': 'Antal fall', '2': 'Antal fall per 100 000 inv'}
indicator_ccov19Reg_dict = {"1":"Antal fall", "11":"Antal fall per 100 000 inv","2":"Antal intensivvårdade fall","21":"Antal intensivvårdade fall per 100 000 inv","3":"Antal avlidna","31":"Antal avlidna per 100 000 inv"}
indicator_ccov19Regsasong_dict = {"1":"Antal fall","11":"Antal fall per 100 000 inv","2":"Antal intensivvårdade fall","21":"Antal intensivvårdade fall per 100 000 inv","3":"Antal avlidna fall","31":"Antal avlidna fall per 100 000 inv"}
indicator_PCRtestVAr_dict = {"0":"Antal testade","1":"Antal positiva","3":"Antal negativa","4":"Ej bedömbara","2":"Andel positiva"}
indicator_xcov19ivavDAG_dict = {"1":"Antal intensivvårdade fall","2":"Antal avlidna fall"}
indicator_ycov19ivavald_dict = {"1":"Antal intensivvårdade fall","2":"Antal intensivvårdade per 100 000 inv","3":"Antal avlidna fall","4":"Antal avlidna per 100 000 inv"}
indicator_ycov19ivavkov_dict = {"3":"Antal intensivvårdade fall","4":"Antal intensivvårdade per 100 000 inv","1":"Antal avlidna fall","2":"Antal avlidna per 100 000 inv"}
category_ecov19sabo_dict = {"1":"SÄBO","2":"Hemtjänst"}
gender_dict = {'1': 'Kvinnor', '2': 'Män', '3': 'Uppgift saknas'}
age_group_dict = {"1":"Alla åldrar","2":"0-19 år","3":"20-69 år","4":"70 år och äldre","5":"Uppgift om ålder saknas"}
age_group_dict_ycov19ivavald = {"0-49":"0-49 år","50-69":"50-69 år","70-":"70- år","Saknas":"Uppgift saknas"}
scb_municipalities = {'01': 'Stockholms län', '03': 'Uppsala län', '04': 'Södermanlands län', '05': 'Östergötlands län', '06': 'Jönköpings län', '07': 'Kronobergs län', '08': 'Kalmar län', '09': 'Gotlands län', '10': 'Blekinge län', '12': 'Skåne län', '13': 'Hallands län', '14': 'Västra Götalands län', '17': 'Värmlands län', '18': 'Örebro län', '19': 'Västmanlands län', '20': 'Dalarnas län', '21': 'Gävleborgs län', '22': 'Västernorrlands län', '23': 'Jämtlands län', '24': 'Västerbottens län', '25': 'Norrbottens län'}
categories_to_skip = ["Antal fall per 10 000 inv", "Antal intensivvårdade fall per 100 000 inv", "Antal intensivvårdade per 100 000 inv", "Antal fall per 100 000 inv", "Antal avlidna per 100 000 inv", "Antal avlidna fall per 100 000 inv"]
gender_suffix_dict = {"k": "Kvinnor", "m": "Män", "s": "Uppgift om kön saknas"}

#sort by reverse order using lambda to convert to int
folders = os.listdir("data")
#Sheets that can be found in excel
excel_dict_sheets = ['Antal per dag region', 'Antal avlidna per dag', 'Antal intensivvårdade per dag', 'Totalt antal per region', 'Totalt antal per kön', 'Totalt antal per åldersgrupp', 'Veckodata Region', 'Veckodata Kommun_stadsdel', 'Veckodata Riket']


#Defined Json_questions
"""
json_question = {        
    "query": [
            {
            "code": "Indikator",
            "selection": {
                "filter": "item",
                "values": [
                    "1",
                    ]
                }
            }
        ],
    "response": {
        "format": "json"
    }
}
"""

json_question = {
  "query": [],
  "response": {
    "format": "json"
  }
}

json_question2 = {
  "query": [],
  "response": {
    "format": "json-stat"
  }
}
#Use json_stat similar to get the specific dictionaries for the different indexes

json_question_PCR_k = {
  "query": [
    {
      "code": "Kön",
      "selection": {
        "filter": "item",
        "values": [
          "1"
        ]
      }
    },
  ],
  "response": {
    "format": "json"
  }
}


json_question_PCR_m = {
  "query": [
    {
      "code": "Kön",
      "selection": {
        "filter": "item",
        "values": [
          "2"
        ]
      }
    },
  ],
  "response": {
    "format": "json"
  }
}

json_question_PCR_s = {
  "query": [
    {
      "code": "Kön",
      "selection": {
        "filter": "item",
        "values": [
          "4"
        ]
      }
    },
  ],
  "response": {
    "format": "json"
  }
}

json_question_bcov19kom_1 = {
  "query": [
    {
      "code": "Indikator",
      "selection": {
        "filter": "item",
        "values": [
          "1"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}

json_question_bcov19kom_2 = {
  "query": [
    {
      "code": "Indikator",
      "selection": {
        "filter": "item",
        "values": [
          "2"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}



json_question_scb = {
  "query": [
    {
      "code": "Region",
      "selection": {
        "filter": "vs:RegionLän07",
        "values": [
          "01",
          "03",
          "04",
          "05",
          "06",
          "07",
          "08",
          "09",
          "10",
          "12",
          "13",
          "14",
          "17",
          "18",
          "19",
          "20",
          "21",
          "22",
          "23",
          "24",
          "25"
        ]
      }
    },
    {
      "code": "ContentsCode",
      "selection": {
        "filter": "item",
        "values": [
          "BE0101N1"
        ]
      }
    },
    {
      "code": "Tid",
      "selection": {
        "filter": "item",
        "values": [
          "2022"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}