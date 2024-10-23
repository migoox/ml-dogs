## MILESTONE 1

### Info

- the data: https://zenodo.org/records/4660670 (DAPS)
  - F1, F7, F8, M3, M6, M8, form Class 1 (they are accepted)
  - others from Class 0 (they are rejected),
- reading the data: libka do wczytywania audio
- data anaysis:
  - musimy wybrać _train set_, _validation set_ oraz _test set_,
    - _train set_ i _test set_ TO ZBIORY ROZŁĄCZNE, tak samo _train set_ oraz _validation set_, ale idk. jak _validation set_ i _test set_,
    - _test set_ może zawierać dane z innych datasetów niż podesłany, ale na labach pokazujemy tylko DAPS
  - przeanalizować jakie mamy głosy i zastanowić się jak może to wpłynąć na model trenowany na naszym train secie,
  - zastanowić się co można spreprocesować,
    - na ten moment tylko zastanowić.
    - np. usuwanie szumu,
    - można też poprzycinać pliki audio tak aby nie zawierały początku i końca który ma sam szum,
  - jako, że nasz problem sprowadza się do image processingu można pomyśleć nad _vectorial features_
    - Wyklad 1 strona 34,
  - wybrać featury (inaczej atrybuty lub deskryptory)
    - z labów:
      - **amplituda**: tzn. kolor pixela na spektogramie,
      - **częstliwość**: oś $y$ na spektogramie,
- create spectogram: libka do spektogramów
- stworzyć bardzo prosty model (network),
  - `pythorch`
  - **IMPORANT**: The network for this milestone can be an unchanged model copied from a tutorial e.g.: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
  - trained to at least 0.6 macro-averaged f1-score on the _train set_
    - tutaj trzeba podstawić liczbę _false positivów_, _true positivów_, _false negativów_ i _true negativów_ do wzorków, nasz model musi miec wskaźnik co najmniej 0.6
    - other metrics: _False Acceptance Ratio_, _False Rejection Ratio_, _SNR_, _WER_
  - (optional) model displays the confidence level on the screen
- NA TEN MILESTONE NIE TRZEBA RAPORTU

### Todo

- [ ] Analiza danych
  - [ ] Przemyślenie i sporządzenie sensownych statystyk na temat wybranego data setu
  - [ ] Wybranie _train_set_, _validation_set_ oraz _test_set_
  - [ ] Przejrzenie datasetu oraz statystyk w celu oceny jego wad
  - [ ] Zastanowienie się co należałoby spreprocesować
  - [ ] Przemyślenie jakie _vectorial_features_ pasują do naszego problemu
- [ ] Skrypt pythonowy wczytujący dane oraz transformujący je na spektogramy
- [ ] Utworzenie skryptu z modelm na podstawie https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- [ ] Analiza rezultatów uzyskanych przez model
