# Előadásszöveg – QuantiMate, TDK 2026

---

## 1. Címdia

- Jó napot kívánok! Csizmi vagyok, az Eszterházy Károly Katolikus Egyetemről.
- A dolgozatom arról szól, hogyan lehet egy tőzsdei előrejelző rendszert úgy felépíteni, hogy ne csapja be saját magát.

## 2. Kutatási kérdés

- A kiinduló kérdésem egyszerű volt: meg lehet-e jósolni az NVIDIA árfolyamát és ingadozását egy napra, egy hétre és egy hónapra előre pontosabban, mint egy egyszerű tippel.
- Az egyszerű tipp árnál az, hogy holnap is annyi lesz, mint ma. Ingadozásnál az, hogy olyan lesz, mint az utóbbi időben.
- A kérdés nehezebb része a második fele: hogyan zárjam ki, hogy a modell csak véletlenül tűnjön jónak.
- A saját munkám három részből állt. Időrendbe illesztettem nyolc adatforrást, két modellváltozatot építettem egy közös keretre, és megterveztem, majd lefuttattam egy 18 ezer ismétléses kísérletet.

## 3. Adatok és időbeli illesztés

- Minden adat nyilvános. Az árak a Yahoo Finance-ről jönnek, a gyorsjelentések és a bennfentes ügyletek a SEC-től, a shortadatok a FINRA-tól. Ehhez jön egy geopolitikai kockázati index és egy kereskedelempolitikai bizonytalansági index.
- A táblázatban a jobb oldali oszlop a lényeg: egy adat csak attól a naptól kerülhet be, amikor ténylegesen nyilvános lett. A shortállományt például óvatosságból csak kilenc munkanappal az elszámolási nap után használom, mert addigra biztosan nyilvános.
- Ha ezt nem tartom be, a modell olyasmit tud, amit akkor senki nem tudhatott, és az eredmény hamisan jó lesz.
- Az adatokat egyszer töltöttem le és befagyasztottam. Futás közben a program nem ér el internetet.

## 4. Két bemutatott verzió, közös motor

- A QuantiMate-et több lépésben, több verzióban fejlesztettem. Ma ezek közül kettőt mutatok be, a V8-at és a V8.5-öt, amelyek ugyanazon a kódon futnak.
- A V8 azt próbálja ki, segít-e, ha többféle adatot kap a modell.
- A V8.5 azt, segít-e, ha több mintán tanul. Itt 40 nagy, az amerikai tőzsdén kereskedett részvény adatain tanul közösen, és a beállításait is maga keresi.
- A modell és a kiválasztás mindkettőben ugyanaz, így ha különbség van, az csak ebből a két dologból jöhet.
- Ezen a két verzión két mérést végeztem. Az első egy 3000 futásos mérés volt, csak az egynapos árra, még a kiválasztási kapu nélkül.
- A végleges mérés ennél jóval nagyobb: két verzió, két cél, három időtáv, harminc időpont és ötven ismétlés, összesen 18 ezer futás.

## 5. Egy futás menete

- Egy futás így néz ki. Kiválasztok egy napot, mondjuk egy pénteket, és csak az addig ismert adatot hagyom meg.
- Ebből készülnek a bemenetek. Utána a megelőző tíz évet felosztom: 70 százalékon tanul a modell, 15 százalékon választok a jelöltek közül, az utolsó 15 százalékot pedig csak a végén nézem meg. A határokon kihagyok néhány hetet, hogy ne szivárogjon át információ.
- Ezután becslem a piaci állapotokat, és több száz jelöltet versenyeztetek: különböző adatcsomagokkal, fák számával és óvatossággal.
- A jelölteknek két kapun kell átjutniuk. Ha valamelyiken elbuknak, marad az egyszerű tipp.
- A döntést rögzítem, és a kiválasztott modellt ötvenszer újratanítom, mindig más véletlenszámmal, hogy lássam, mennyire stabil.
- A tényleges értéket a program csak a legvégén olvassa ki, amikor a becslés már kész.

## 6. Célváltozó és modell

- A modell nem közvetlenül dollárárat tanul. Árnál a következő napok hozamát becsli, a mostani ingadozáshoz viszonyítva, és ezt számolom vissza árra. Így egy nyugodt 2016-os nap és egy ideges 2024-es nap ugyanazon a skálán van.
- A modell a LightGBM. Ez sok kis döntési fát épít egymás után, és mindegyik az előzőek hibáját javítja.
- Minden jelölt háromféle súllyal is versenyez: teljes erővel, illetve félúton vagy negyedúton visszahúzva az egyszerű tipp felé.

## 7. HMM: milyen állapotban van a piac?

- A piac nem mindig ugyanúgy viselkedik. Vannak nyugodt emelkedő szakaszok, csökkenő szakaszok és turbulens napok. Ezeket közvetlenül nem látjuk, csak a hozamokból és az ingadozásból következtethetünk rájuk. Erre való a rejtett Markov-modell.
- A bal oldali ábrán a három állapot látható, a nyilak pedig azt, milyen eséllyel vált át a piac egyik napról a másikra.
- A képlet a napi frissítést írja le, és két részből áll.
- Az első rész a szumma, ez a várakozás. A tegnapi valószínűségeket (pₜ₋₁) végigvezetem az átmenettáblán (A), vagyis megnézem, honnan milyen eséllyel juthat ma a piac az adott állapotba.
- A második rész a bₖ, ez az illeszkedés: mennyire jellemző a mai hozam és ingadozás az adott állapotra.
- A kettőt összeszorzom, majd úgy osztom el, hogy a három valószínűség együtt 100 százalék legyen. Ezt jelzi az arányosság jele.
- Ha például tegnap emelkedő piac volt, de ma nagyot esik az ár és nagy a kilengés, akkor a turbulens állapot valószínűsége hirtelen megnő.
- A számítás csak az aznapig ismert adatot használja. A könyvtár alapbeállítása a későbbi napokat is figyelembe venné, ezért ezt a részt saját magam írtam meg.
- A három valószínűséget a modell minden napra új bemenetként kapja meg.

### Háttér a képlethez (nem kell elmondani, de kérdésnél jól jön)

A képlet:

> **pₜ(k) ∝ bₖ(xₜ) · Σⱼ pₜ₋₁(j) · Aⱼₖ**

ahol j és k az 1., 2. és 3. állapot, a szumma j = 1-től 3-ig megy.

Felolvasva: „p t k arányos b k x t-szer szumma j p t mínusz egy j A j k”.

- **xₜ, a mai megfigyelés:** két szám, a napi loghozam és az EWMA-volatilitás. Mindkettőt a tanítószakasz átlagával és szórásával standardizálom, hogy egy skálán legyenek.
- **pₜ(k):** annak a valószínűsége, hogy a piac a t-edik napon a k-adik állapotban van, a t-edik napig ismert adatok alapján. Ez a bal oldal, ezt számolom ki minden napra.
- **pₜ₋₁(j), a tegnapi tudás:** ugyanez az előző napra. Ebből indul a frissítés.
- **Aⱼₖ, az átmenettábla:** annak az esélye, hogy a piac egy nap alatt a j-edik állapotból a k-adikba kerül. Három-szor-hármas táblázat. Az átlójában jellemzően nagy számok vannak, mert a piac általában több napig ugyanabban az állapotban marad.
- **A szumma:** a három tegnapi állapotra összeadja, honnan milyen eséllyel lehet ma a k-adik állapotba jutni. Ez a mai várakozás, a mai adat megnézése előtt.
- **bₖ(xₜ), az illeszkedés:** minden állapotnak saját kétdimenziós normális eloszlása van, saját átlagos hozammal, ingadozással és szórásokkal. A bₖ azt mondja meg, mennyire valószínű a mai megfigyelés, ha a piac a k-adik állapotban van. A turbulens állapothoz nagy ingadozás tartozik, ezért egy nagy kilengésnél ennek az értéke lesz a legnagyobb.
- **A szorzás:** csak az az állapot kap nagy súlyt, amelyik a tegnapiból elérhető, és a mai adathoz is illik.
- **Az arányosság jele (∝):** a három szorzat összege nem egy. Mindegyiket elosztom a három összegével, így valószínűségeket kapok.
- **Példa:** tegnap 80 százalék emelkedő, 15 csökkenő, 5 turbulens. Ma nagy esés és nagy kilengés. A várakozás még az emelkedő felé húz, de a mai adathoz a turbulens illik a legjobban, ezért a normálás után a turbulens valószínűsége megugrik.
- **Honnan jönnek a paraméterek?** Az állapotok eloszlásait, az átmenettáblát és a kezdő valószínűségeket a tanítószakaszon becslem az EM-algoritmussal (Baum–Welch), 300 iterációval, rögzített véletlenszámmal (seed 42).
- **Honnan jönnek a nevek?** Utólag adom őket: a legnagyobb ingadozású állapot a turbulens, a másik kettőt az átlagos hozamuk alapján nevezem csökkenőnek, illetve emelkedőnek.
- **Miért saját kód?** A hmmlearn könyvtár beépített függvénye simított valószínűséget ad, ami a teljes idősort használja, vagyis egy 2021-es nap állapotához a 2022-es adatot is felhasználná. Az én számításom csak az első t napot nézi, és ezt egy teszt is ellenőrzi: a t-edik nap értéke ugyanaz, akár csak t-ig, akár a teljes sorra számolom.

## 8. Kétlépcsős kapu

- Szerintem ez a munka legfontosabb része. Ha több száz jelöltet kipróbálok, szinte biztos, hogy valamelyik véletlenül jónak fog látszani.
- Ezért az első lépcső nem egyenként vizsgálja a jelölteket, hanem egyszerre mindet, a Hansen-féle SPA-próbával. A legjobb csak akkor megy tovább, ha az egész csoportot figyelembe véve is kicsi az esélye, hogy csak szerencséje volt.
- A második lépcsőben ugyanezt a modellt a tanítóidőszak négy elkülönített szakaszán is kipróbálom. Ha az előnye ott nem jön elő, akkor az csak egy időszak sajátossága volt.
- Ha bármelyik lépcső nem teljesül, a rendszer az egyszerű tippet adja ki. Ezt nem tekintem hibának: ez a rendszer helyes válasza arra, hogy itt nem tud jobbat.

## 9. Miért ezek a módszerek?

- Ebben a táblázatban azt foglaltam össze, mit miért választottam.
- A legtöbb sor ugyanarról szól. A kézenfekvő megoldások, például az adatok véletlenszerű összekeverése vagy a legjobb jelölt egyszerű kiválasztása, tőzsdei adaton túl szép eredményt adnak.
- Az utolsó sor az ingadozásra vonatkozik. A tényleges ingadozást csak zajosan tudjuk mérni, és ilyenkor a QLIKE nevű hibamérték rangsorolja helyesen a modelleket.

## 10. Első és végleges mérés

- Hogy miért kellett a kapu, azt a 3000 futásos első mérés mutatta meg. Ugyanezen a két verzión futott, de ott még automatikusan a validáción legjobb modellt választottam, és más eredmény jött ki.
- Az oszlopok azt mutatják, mennyivel tévedett többet a modell, mint az egyszerű tipp. Kapu nélkül a V8 4,1, a V8.5 2,7 százalékkal tévedett nagyobbat.
- A 18 ezres végleges mérésben, ugyanezen a feladaton és ugyanazokon a napokon, a kapuval a V8 pontosan annyit téved, mint az egyszerű tipp, a V8.5 pedig egy hajszállal kevesebbet.
- A kapu tehát kiszűrte azokat a modelleket, amelyek csak a próbán tűntek jónak.

### Háttér (nem kell elmondani)

- A 3000 tervezett futásból 2900 készült el. A 6. dátumpárnál a HMM-illesztés numerikus hibával leállt; ezt a standardizálással javítottam. Ezért az ábra a közös 29 dátumpáron hasonlít.
- Az első mérés pillanatképéből mind a nyolc kiegészítő adatfájl hiányzott (gyorsjelentés, bennfentes ügyletek, shortadatok, 60 perces árak, GPR, EPU), így ott a kiegészítő adatrétegek nem versenyeztek: a V8 csak az alapjellemzőkkel, a V8.5 az alapjellemzőkkel és a 40 részvényes közös tanítással futott. A program ezt akkor csak naplózta; azóta a hiányzó forrás hibát ad, és a végleges mérésben mind megvolt.
- Az első mérésben a modell a futások 47, illetve 46 százalékában volt pontosabb az egyszerű tippnél, a többiben rosszabb. Átlagos hiba: V8 2,64%, V8.5 2,61%, egyszerű tipp 2,54%.
- A végleges mérésben ugyanezen a 29 páron: V8 2,54%, V8.5 2,53%.
- A két mérés eredményei közvetlenül összevethetők, mert az egyszerű tipp hibája mindkettőben pontosan 2,54%, vagyis ugyanazokon a napokon mértek.

## 11. Ki volt pontosabb?

- Ez az ábra mind a 18 ezer futást mutatja. Zöld, ahol a modell volt pontosabb, piros, ahol az egyszerű tipp, szürke, ahol ugyanazt adták.
- A felső három sor az ár. Szinte csak szürkét látunk: a rendszer majdnem mindig arra jutott, hogy itt nem tud jobbat az egyszerű tippnél.
- Az alsó három sor az ingadozás, itt már sok a zöld. Másnapra a futások 69, illetve 74 százalékában a modell volt pontosabb.
- Hosszabb távon kisebb az előny, és többször maradt az egyszerű tipp.

## 12. Ingadozás holnapra

- Itt a harminc célnapot külön is lehet látni. A fekete a tényleges ingadozás, a barna a rendszer becslése, a szürke szaggatott vonal az egyszerű tipp.
- Az egyszerű tipp sokszor túlbecsül, főleg 2023-ban és 2024-ben, a becslés pedig közelebb marad a valósághoz. A harminc napból huszonkettőn a rendszer volt közelebb.
- Amit nem tud: a 2025 júliusi hirtelen ugrást egyik módszer sem látta előre.
- Árban a rendszer szinte mindig az egyszerű tippet adja.

## 13. Összefoglalás

- Az árat szigorú ellenőrzés mellett nem sikerült jobban megjósolni annál, hogy holnap is annyi lesz, mint ma. Ezen sem a többféle adat, sem a több minta nem változtatott. Ez összhangban van azzal, amit a hatékony piacok elméletéből várni lehet.
- Az ingadozást viszont a modell átlagosan jobban jelzi előre, mint az egyszerű tipp, legjobban egy napra előre.
- Ezt még óvatosan kell kezelni, mert harminc időpont kevés ahhoz, hogy statisztikailag biztosak lehessünk benne.
- Amit a munkám hozzátesz: egy olyan keretrendszer, amely megismételhető, nem lát a jövőbe, és ha a modell nem jobb, azt ki is mondja. Hogy ez mennyit számít, azt az első mérés mutatta meg: ellenőrzés nélkül a modell rosszabb volt az egyszerű tippnél.
- A következő lépés az, hogy az ingadozásnál látott előnyt egy új, 2025 novembere utáni időszakon is megvizsgáljam, olyan adaton, amely eddig egyik döntésnél sem szerepelt.
- Köszönöm a figyelmet, szívesen válaszolok a kérdésekre.