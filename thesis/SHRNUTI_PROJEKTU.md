# Shrnutí bakalářské práce — datová podpora výběru zahraničních knih k překladu do češtiny

*Podklad pro konzultaci s vedoucím práce. Stav repozitáře k 14. 6. 2026.*

---

## 1. Cíl práce (důležité přesné rámování)

Práce staví datovou sadu a klasifikátor, který má nakladateli pomoci rozhodnout,
**které dosud nepřeložené zahraniční tituly stojí za akvizici** pro český trh.
Model se učí ze vzorců knih, které čeští nakladatelé v minulosti vybrali k
překladu a které následně zabodovaly v žebříčku.

Zásadní je formulace výstupu. Práce **netvrdí**, že model predikuje bestsellery
v absolutním smyslu. Tvrdí, že najde knihy „**podobné úspěšným minulým českým
akvizicím**". Tento rozdíl je ve výsledcích i v textu práce nutné držet
konzistentně — je to obrana proti námitce ohledně výběrového zkreslení (viz §7).

---

## 2. Datové zdroje

Spojují se tři zdroje:

- **NKC** — katalog Národní knihovny ČR ve formátu MARC21 (~10 GB XML). Zdroj
  záznamů o českých překladech (autor, originální název, rok vydání, ISBN,
  zdrojový jazyk, žánr).
- **Goodreads** — datový dump z roku 2017 (~9 GB knihy + ~15 GB recenze). Zdroj
  signálu o popularitě/žánru/hodnocení **původního (cizojazyčného) vydání**.
- **SCKN** — české žebříčky bestsellerů scrapnuté z sckn.cz (týdenní top 10,
  2003–současnost). Zdroj **labelů** (cílové proměnné).

Nejtěžší technický problém je **párování (matching)** záznamů NKC s odpovídající
knihou na Goodreads — odlišné názvy, odlišné písmo/diakritika, žádné společné
ISBN.

---

## 3. Pipeline (6 kroků, pořadí spuštění)

1. `scrape_sckn.py` → žebříčky SCKN
2. `parse_nkc.py` → české překlady z NKC + inventář pokrytí polí
3. `build_goodreads_lookup.py` → vyhledávací indexy nad Goodreads (~30 min)
4. `build_matched_dataset.py` → **párovací kaskáda** NKC → Goodreads + napojení
   labelů SCKN (~30 min)
5. `aggregate_reviews.py` → **pre-cutoff** statistiky recenzí (před rokem vydání)
6. `build_features.py` / notebook 05 → finální matice `X_train` / `y_train`

Kroky 1–3 jsou stabilní. Krok 4 (ladění kaskády) je nejvíce proměnlivá část;
kroky 5–6 je nutné po každé změně kroku 4 přepočítat.

---

## 4. Párovací kaskáda (jádro datové části)

Tříúrovňová kaskáda na záznam NKC, vyhrává první shoda:

1. **Vrstva 1** — autor přesně + název přesně (po NFKD normalizaci diakritiky).
2. **Vrstva 2** — autor přesně + název fuzzy (`token_set_ratio ≥ 88`).
3. **Vrstva 3** — autor fuzzy (`≥ 85`) + název přesně.
4. Jinak — `unmatched`.

Při více kandidátech se rozhoduje přes `work_id` Goodreads (totéž dílo v různých
vydáních): nejprve shoda jazyka, pak blízkost roku vydání k `czech_pub_year`, pak
počet hodnocení.

**Co bylo vědomě zavrženo:** vrstva pouze podle názvu (bez ověření autora) — v
předchozí verzi (v1) byla hlavním zdrojem šumu (správné labely spárované se
špatnými featurami). Také párování autor-fuzzy + název-fuzzy (výpočetně
neúnosné). Most přes Open Library (OCLC→ISBN) byl **empiricky zamítnut** — pokrytí
jen 0,7 %, a navíc selhával přesně v období, kde žije pozitivní kohorta (2010+).
Negativní výsledek je zdokumentován, což je v textu práce obhajitelné.

---

## 5. Klíčová čísla matchingu (`reports/match_funnel.json`)

- SCKN: 8 165 unikátních ISBN-13 v žebříčcích.
- **SCKN → NKC:** 3 397 (41,6 %) ISBN ze SCKN je v katalogu překladů NKC. Zbytek
  jsou převážně česky psané originály / nekniižní kategorie — mimo rozsah.
- **NKC → Goodreads:** 55 983 z 225 483 záznamů (24,8 %); vrstvy L1 = 39 376,
  L2 = 13 265, L3 = 3 342.
- **Bestsellery:** z 3 386 SCKN-pozitivních záznamů NKC se 1 777 (52,5 %) podaří
  spárovat s Goodreads — pozitivní tituly se párují ~2× častěji než průměr (jsou
  to populární knihy).
- **Modelovatelná sada:** 24 161 unikátních zahraničních děl, 1 287 pozitivních
  (5,3 %).

---

## 6. Label, pravidla proti úniku informací (leakage) a featury

**Label:** `sckn_bestseller = sckn_appearances >= 1` (alespoň jedno objevení v top
10 SCKN). Prahová hodnota je stanovena vědomě a nemění se bez diskuze.

**Zákaz časového úniku (klíčové pro obhajobu):** featury musí pocházet z doby
**před** `czech_pub_year`. Proto se `gr_ratings_count` a `gr_average_rating` ze
statického snapshotu z roku 2017 (tj. po vydání) **nesmí** dostat do `X_train`.
Místo nich se počítají **pre-cutoff** statistiky recenzí agregované na úroveň díla.

**Matice featur:** 16 featur, 24 161 řádků, 1 287 pozitivních (5,3 %, nepoměr
~17,8:1):

- Popularita: `pre_cutoff_ratings_count`, `pre_cutoff_avg_rating` (imputace 3,5
  při chybějící hodnotě), `log_pre_cutoff_ratings_count`, `has_precutoff_signal`.
- Žánrové podíly (7) z `gr_popular_shelves`: fiction, mystery, romance, scifi,
  nonfiction, ya, classics.
- Dummy proměnné zdrojového jazyka (4): eng, ger, fre, rus.
- `czech_pub_year` (slouží i jako klíč pro rozdělení dat — viz níže).

---

## 7. Modelování — stav a výsledky

**Kohorta 2003–2017.** Tituly s `czech_pub_year >= 2018` se pro modelování
vyřazují: jejich pozitivní podíl uměle klesá (6,4 % v 2016 → ~3 % v 2019) kvůli
driftu featur (snapshot 2017) a neúplným recentním labelům SCKN — artefakt, ne
signál.

**Časové rozdělení dat** (ne náhodné — úkol je „predikovat budoucnost"):
train ≤ 2014, val = 2015, test = {2016, 2017}. Velikosti: train 14 242 (835 poz.)
· val 1 480 (95) · test 3 198 (193). Pozitivní podíl napříč splity stabilní
(5,9 % / 6,4 % / 6,0 %).

**Hlavní metrika:** PR-AUC (average precision) + precision/recall-at-K
(K = 50/100/250). Accuracy a ROC-AUC jsou při základním podílu ~5–6 % zavádějící.

**Výsledky na validaci** (base rate 6,4 %):

| Model | PR-AUC | ROC-AUC | lift@50 |
|---|---|---|---|
| pouze popularita | 0,138 | 0,60 | 3,7× |
| logistická regrese (vyvážená) | 0,167 | 0,72 | 4,4× |
| **LightGBM** (vítěz) | **0,188** | **0,74** | **4,7×** |

Každý model překonává předchozí — vícerozměrný signál pomáhá (to je jádro
argumentu práce), ačkoli absolutní strop je skromný: jde o opravdu těžkou úlohu.
Kalibrace LightGBM (sigmoid na val) snížila Brier 0,167 → 0,057 beze změny pořadí.
Nepoměr tříd řešen přes `scale_pos_weight` (~16:1), bez SMOTE.

**Test split je zatím zapečetěný** — má se vyhodnotit jen jednou na úplném konci.

---

## 8. Známá omezení (vědomá, ne „chyby k opravě")

- **Goodreads = snapshot 2017** → knihy vydané česky po 2017 mají podhodnocený
  pre-cutoff signál.
- **Jen ~25 % záznamů NKC** se vůbec spáruje s Goodreads (strop dán 62 % pokrytím
  pole originálního názvu). Model je implicitně podmíněn tím, že „kniha je
  dohledatelná na Goodreads".
- **Silný nepoměr tříd** (~17,8:1).
- **Výběrové zkreslení** — to, které knihy se vůbec přeloží, je zkreslené. Podle
  zadání se **nepřekonává inženýrsky**, ale poctivě pojmenovává v textu.
- **Žádná zlatá množina** pro přesnost matchingu — místo toho strukturované ruční
  spot-checky (kritérium: ≤ 2/20 špatných shod na vrstvu).

---

## 9. Body k diskuzi s vedoucím

Tyto body bych otevřel na konzultaci:

1. **Rámování výsledků.** Je formulace „knihy podobné úspěšným minulým akvizicím"
   (nikoli „predikce bestsellerů") dostatečná obrana výběrového zkreslení pro
   obhajobu? Jak silně to v textu zdůraznit?

2. **Skromný absolutní výkon.** PR-AUC ~0,19 a lift@50 ~4,7× — je to pro
   bakalářskou práci dostačující výsledek, když je hlavní přínos *poctivě
   změřená obtížnost úlohy* a *funkční end-to-end pipeline*? Jak prezentovat
   „těžký problém" jako legitimní výsledek, ne jako neúspěch?

3. **Nízké pokrytí matchingu (~25 %).** Je strop daný 62% pokrytím originálního
   názvu v NKC akceptovatelné omezení, nebo vedoucí očekává další pokus o jeho
   zvýšení? (Most přes Open Library už byl zamítnut — mám připravenou
   dokumentaci negativního výsledku.)

4. **Absence zlaté množiny pro matching.** Stačí kvalitativní spot-check
   metodika místo number-based precision/recall? Toto je metodologicky
   nejnapadnutelnější místo práce.

5. **`czech_pub_year` jako featura.** V kódu je z modelových featur **vyřazen**
   (pod časovým splitem má každý testovací řádek rok nevídaný v tréninku, takže
   stromy jen extrapolují). Otázka v CLAUDE.md je tím fakticky vyřešena — chci
   potvrdit, že je to správné rozhodnutí a jak ho v textu zdůvodnit.

6. **Práh labelu = 1 objevení.** Je „alespoň jednou v top 10" smysluplná
   definice úspěchu, nebo by vedoucí chtěl citlivostní analýzu pro vyšší prahy
   (např. ≥ 3 objevení)?

7. **Co dál.** Zbývá: finální vyhodnocení na zapečetěném testu, SHAP atribuce pro
   text práce, a samotné sepsání (složka `thesis/` je zatím prázdná). Jaké je
   pořadí priorit a rozsah textové části?

---

## 10. Stav repozitáře

Datová pipeline (kroky 1–6) je hotová a proběhla end-to-end. Modelování rozjeté:
časový split, evaluační harness, baseliny i LightGBM jsou hotové. Zbývá finální
test, SHAP a sepsání práce. Notebook `08_modeling.ipynb` (orchestrace) je TODO.
