# Előadásszöveg – QuantiMate, TDK 2026

---

## 1. Címdia

- Jó napot kívánok! Csizmadia Máté vagyok, az Eszterházy Károly Katolikus Egyetemről.
- A dolgozatom arról szól, hogyan lehet egy tőzsdei előrejelző rendszert úgy felépíteni, hogy ne csapja be saját magát.

## 2. Kutatási kérdés

- A kiinduló kérdésem egyszerű volt: meg lehet-e jósolni az NVIDIA árfolyamát és ingadozását egy napra, egy hétre és egy hónapra előre pontosabban, mint egy egyszerű tippel.
- Az egyszerű tipp árnál az, hogy holnap is annyi lesz, mint ma. Ingadozásnál az, hogy olyan lesz, mint az utóbbi időben.
- A kérdés nehezebb része a második fele: hogyan zárjam ki, hogy a modell csak véletlenül tűnjön jónak.
- A saját munkám három részből állt. Időrendbe illesztettem nyolc adatforrást, két modellváltozatot építettem egy közös keretre, és megterveztem, majd lefuttattam egy 18 ezer ismétléses kísérletet.

## 3. Adatok és időbeli illesztés

- Minden adat nyilvános. Az árak a Yahoo Finance-ről jönnek, a gyorsjelentések és a bennfentes ügyletek a SEC-től, a shortadatok a FINRA-tól. Ehhez jön egy geopolitikai kockázati index és egy kereskedelempolitikai bizonytalansági index.
- A táblázatban a jobb oldali oszlop a lényeg: egy adat csak attól a naptól kerülhet be, amikor ténylegesen nyilvános lett. A shortállományt például csak kilenc munkanappal később közlik.
- Ha ezt nem tartom be, a modell olyasmit tud, amit akkor senki nem tudhatott, és az eredmény hamisan jó lesz.
- Az adatokat egyszer töltöttem le és befagyasztottam. Futás közben a program nem ér el internetet.

## 4. Két változat, közös motor

- Két változatot hasonlítottam össze, ugyanazon a kódon.
- A V8 azt próbálja ki, segít-e, ha többféle adatot kap a modell.
- A V8.5 azt, segít-e, ha több mintán tanul. Itt 40 nagy amerikai részvény adatain tanul közösen, és a beállításait is maga keresi.
- A modell és a kiválasztás mindkettőben ugyanaz, így ha különbség van, az csak ebből a két dologból jöhet.
- A kísérlet mérete: két verzió, két cél, három időtáv, harminc időpont és ötven ismétlés. Ez összesen 18 ezer futás.

## 5. Egy futás menete

- Egy futás így néz ki. Kiválasztok egy napot, mondjuk egy pénteket, és csak az addig ismert adatot hagyom meg.
- Ebből készülnek a bemenetek. Utána a megelőző tíz évet felosztom: 70 százalékon tanul a modell, 15 százalékon választok a jelöltek közül, az utolsó 15 százalékot pedig csak a végén nézem meg. A határokon kihagyok néhány hetet, hogy ne szivárogjon át információ.
- Ezután becslem a piaci állapotokat, és betanítok több száz jelölt modellt.
- A jelölteknek két kapun kell átjutniuk. Ha valamelyiken elbuknak, marad az egyszerű tipp.
- A döntést rögzítem, és a kiválasztott modellt ötvenszer újratanítom, mindig más véletlenszámmal, hogy lássam, mennyire stabil.
- A tényleges értéket a program csak a legvégén olvassa ki, amikor a becslés már kész.

## 6. Célváltozó és modell

- A modell nem közvetlenül dollárárat tanul. Árnál a következő napok hozamát becsli, a mostani ingadozáshoz viszonyítva, és ezt számolom vissza árra. Így egy nyugodt 2016-os nap és egy ideges 2024-es nap ugyanazon a skálán van.
- A modell a LightGBM. Ez sok kis döntési fát épít egymás után, és mindegyik az előzőek hibáját javítja. A korábbi verziókban neurális hálókat is kipróbáltam, de ekkora adatmennyiségen a fák jobban teljesítettek.
- Minden jelölt háromféle súllyal is versenyez: teljes erővel, illetve félúton vagy negyedúton visszahúzva az egyszerű tipp felé.

## 7. HMM: milyen állapotban van a piac?

- A piac nem mindig ugyanúgy viselkedik. Vannak nyugodt emelkedő szakaszok, csökkenő szakaszok és turbulens napok. Ezeket az állapotokat közvetlenül nem látjuk, csak a hozamokból és az ingadozásból következtethetünk rájuk. Erre való a rejtett Markov-modell.
- A bal oldali ábrán a három állapot látható. A nyilak azt mutatják, milyen eséllyel vált át a piac egyik napról a másikra egyik állapotból a másikba.
- A képlet a napi frissítést írja le. A tegnapi valószínűségeket átvezetem az átmeneteken, majd megnézem, mennyire illik a mai megfigyelés az egyes állapotokhoz.
- A számítás csak az aznapig ismert adatot használja. A könyvtár alapbeállítása a teljes idősorból számolna, vagyis egy múltbeli nap állapotát a későbbi napok ismeretében becsülné meg. Ezért ezt a részt saját magam írtam meg.
- A három valószínűséget a modell új bemenetként kapja meg.

## 8. Kétlépcsős kapu

- Szerintem ez a munka legfontosabb része. Ha több száz jelöltet kipróbálok, szinte biztos, hogy valamelyik véletlenül jónak fog látszani.
- Ezért az első lépcső nem egyenként vizsgálja a jelölteket, hanem egyszerre mindet, a Hansen-féle SPA-próbával. A legjobb csak akkor megy tovább, ha az egész csoportot figyelembe véve is kicsi az esélye, hogy csak szerencséje volt.
- A második lépcsőben ugyanezt a modellt a tanítóidőszak négy elkülönített szakaszán is kipróbálom. Ha az előnye ott nem jön elő, akkor az csak egy időszak sajátossága volt.
- Ha bármelyik lépcső nem teljesül, a rendszer az egyszerű tippet adja ki. Ezt nem tekintem hibának: ez a rendszer helyes válasza arra, hogy itt nem tud jobbat.

## 9. Miért ezek a módszerek?

- Ebben a táblázatban azt foglaltam össze, mit miért választottam.
- A legtöbb sor ugyanarról szól. A kézenfekvő megoldások, például az adatok véletlenszerű összekeverése vagy a legjobb jelölt egyszerű kiválasztása, tőzsdei adaton túl szép eredményt adnak.
- Az utolsó sor az ingadozásra vonatkozik. A tényleges ingadozást csak zajosan tudjuk mérni, és ilyenkor a QLIKE nevű hibamérték rangsorolja helyesen a modelleket.

## 10. Első mérés és végleges futás

- Hogy miért kellett a kapu, azt az első mérésem mutatta meg. Ott még automatikusan a validáción legjobb modellt választottam.
- Az oszlopok azt mutatják, mennyivel tévedett többet a modell, mint az egyszerű tipp. Kapu nélkül a V8 4,1, a V8.5 2,7 százalékkal tévedett nagyobbat.
- Ugyanezen a feladaton, a kapuval a végleges futásban a V8 pontosan annyit téved, mint az egyszerű tipp, a V8.5 pedig egy hajszállal kevesebbet.
- A kapu tehát kiszűrte azokat a modelleket, amelyek csak a próbán tűntek jónak.

## 11. Ki volt pontosabb?

- Ez az ábra mind a 18 ezer futást mutatja. Zöld, ahol a modell volt pontosabb, piros, ahol az egyszerű tipp, szürke, ahol ugyanazt adták.
- A felső három sor az ár. Szinte csak szürkét látunk: a rendszer majdnem mindig arra jutott, hogy itt nem tud jobbat az egyszerű tippnél.
- Az alsó három sor az ingadozás, itt már sok a zöld. Másnapra a futások 69, illetve 74 százalékában a modell volt pontosabb.
- Hosszabb távon kisebb az előny, és többször maradt az egyszerű tipp.

## 12. Ingadozás holnapra

- Itt a harminc célnapot külön is lehet látni. A fekete a tényleges ingadozás, a barna a rendszer becslése, a szürke szaggatott vonal az egyszerű tipp.
- Az egyszerű tipp sokszor túlbecsül, főleg 2023-ban és 2024-ben, a becslés pedig közelebb marad a valósághoz. A harminc napból huszonkettőn a rendszer volt közelebb.
- Amit nem tud: a 2025 júliusi hirtelen ugrást egyik módszer sem látta előre.
- Árban a rendszer mindhárom távon az egyszerű tippet adja.

## 13. Összefoglalás

- Az árat szigorú ellenőrzés mellett nem sikerült jobban megjósolni annál, hogy holnap is annyi lesz, mint ma. Ezen sem a többféle adat, sem a több minta nem változtatott. Ez összhangban van azzal, amit a hatékony piacok elméletéből várni lehet.
- Az ingadozást viszont a modell átlagosan jobban jelzi előre, mint az egyszerű tipp, legjobban egy napra előre.
- Ezt még óvatosan kell kezelni, mert harminc időpont kevés ahhoz, hogy statisztikailag biztosak lehessünk benne.
- Amit a munkám hozzátesz: egy olyan keretrendszer, amely megismételhető, nem lát a jövőbe, és ha a modell nem jobb, azt ki is mondja. Sok hasonló rendszer éppen azért mutat jó eredményt, mert ezt nem ellenőrzi.
- A következő lépés az, hogy az ingadozásnál látott előnyt egy új, 2025 novembere utáni időszakon is megvizsgáljam, olyan adaton, amely eddig egyik döntésnél sem szerepelt.
- Köszönöm a figyelmet, szívesen válaszolok a kérdésekre.
