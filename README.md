#  Studiu Multivariat: Analiza Specificațiilor Automobilelor

Proiectul reprezintă un studiu complex asupra domeniului auto, investigând relațiile dintre diverse branduri, modele și motorizări. Studiul se bazează pe **13 variabile esențiale**, incluzând puterea, accelerația (0-100 km/h), cuplul, dimensiunile (lungime, capacitate portbagaj) și alte specificații tehnice.

##  Setul de Date

Datele utilizate sunt reale și provin de pe Kaggle: [Car Specification Dataset](https://www.kaggle.com/datasets/usefashrfi/car-specification-dataset).

- **Sursă inițială:** Peste 20.000 de specificații.
- **Procesare:** Deoarece analizele multivariate necesită o imagine statică, setul a fost filtrat pentru a include exclusiv observații din anul **2023**. Valorile absurde sau lipsă au fost eliminate.
- **Set final:** **1.693 de mașini**, aparținând a peste **50 de branduri**, cu peste 25 de modele fiecare.
- **Locație:** Fișierul `.csv` final se află în folderul *SET DE DATE*.

##  Structura Proiectului

Proiectul este documentat și organizat pentru a facilita atât vizualizarea rapidă, cât și execuția tehnică:

### 1. Documentație și Interpretare (`.ipynb`)
Rezultatele și pașii de execuție sunt detaliați în fișiere **Jupyter Notebook**. Acestea includ:
- Definiții și explicații teoretice.
- Interpretări ale rezultatelor.
- Grafice și tabele în format `.csv`.

### 2. Vizualizare Rapidă (`ANALIZE ÎN FORMAT HTML`)
Acest folder conține exportul în format `.html` al celor 5 fișiere notebook. Este ideal pentru o parcurgere rapidă a documentației fără a necesita rularea codului.

### 3. Scripturi și Output (`.py`)
Fiecare analiză are propriul folder care include:
- Scriptul `.py` aferent.
- Fișierele de tip output rezultate în urma rulării.
- *Notă:* Graficele pot fi vizualizate rulând scriptul sau direct în fișierele Notebook/HTML.
- *Notă:* Scriptul *utils.py* conține câteva funcții importante apelate în scripturile aferente analizelor multivariate.

##  Analize Multivariate Abordate

Proiectul acoperă 5 metode statistice distincte pentru a extrage informații relevante din setul de date:

1. **Analiza Componentelor Principale (PCA)**
2. **Analiza Factorială**
3. **Analiza Corelațiilor Canonice**
4. **Analiza Discriminantă Liniară (LDA)**
5. **Analiza Clusterilor Ierarhici**

##  Tehnologii Utilizate

Proiectul a fost realizat folosind limbajul **Python** și următoarele biblioteci pentru manipularea datelor, calcule statistice și vizualizare:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`