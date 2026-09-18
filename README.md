# Aeris – Autorski Lokalny Model Językowy i Interfejs GUI



Aeris to lekki model generatywny oparty na architekturze dekodera Transformera (typ GPT), zaimplementowany od podstaw w PyTorchu. 
Projekt obejmuje autoregresyjną pętlę generowania tekstu, zoptymalizowany pipeline douczania oraz dedykowany interfejs graficzny stworzony w PyQt.







## Główne Cechy



* Własna Architektura Transformera: Zbudowana na modułach PyTorch z wykorzystaniem mechanizmu Multi-Head Self-Attention, maskowania kauzalnego, stabilizacji Pre-LayerNorm oraz skalowanych wektorów zanurzeń (embeddings).



* Kontrola Generowania Tekstu: Autoregresyjny sampling z możliwością regulacji temperatury, filtrowania Top-K oraz kar za powtórzenia (repetition penalty).



* Wydajny Proces Uczenia: Pętla treningowa wspierająca akumulację gradientów (gradient accumulation), kernele FlashAttention, optymalizator AdamW z selektywnym wykluczeniem parametrów bez wag (weight decay exclusion) oraz kosinusowy harmonogram uczenia (Cosine Annealing).



* Wielowątkowy Interfejs GUI: Aplikacja desktopowa w PyQt wykorzystująca osobne wątki robocze (QThread) i asynchroniczne sygnały (pyqtSignal), co zapobiega zacinaniu interfejsu podczas inferencji modelu.



* Dostrajanie Dziedzinowe: Model dotrenowany na wyselekcjonowanym korpusie dialogowym w konwencji użytkownik-asystent.







## Struktura Projektu



```text
aeris-transformer-core/
├── data/
│   └── sample_dataset.txt        # Przykładowy zbiór danych do pre-trainingu
├── data_adapt/
│   └── sample_adapt.txt          # Próbka danych do adaptacji dziedzinowej
├── model/
│   └── aeris_model.pt            # Wagi wyuczonego modelu (lokalnie, wykluczone z Git)
├── .gitignore                    # Reguły wykluczania dużych wag i plików cache
├── aeris_gui.py                  # Wielowątkowy interfejs graficzny PyQt
├── aeris_tokenizer_32k.model     # Model tokenizera SentencePiece
├── aeris_tokenizer_32k.vocab     # Słownik tokenizera SentencePiece
├── config.py                     # Konfiguracja hiperparametrów i wymiarów modelu
├── embedding.py                  # Moduł wektorowania tokenów i pozycji (Embeddings)
├── model.py                      # Architektura dekodera Transformera (Self-Attention i MLP)
├── README.md                     # Dokumentacja projektu
├── requirements.txt              # Zależności i wymagane biblioteki
├── train_full.py                 # Główny skrypt treningowy (pre-training)
└── train_adapt.py                # Skrypt adaptacji dziedzinowej (continual training)



```





## Przegląd Architektury



* Token \& Positional Embeddings: Konwersja identyfikatorów tokenów na wektory liczbowe skalowane przez pierwiastek z wymiaru modelu (sqrt(d\_model)) wraz z dodaniem uczonych wektorów pozycji.  



* Bloki Kauzalnego Transformera: Warstwy Multi-Head Attention z dolnotrójkątną maską kauzalną, wymuszającą przewidywanie tekstu krok po kroku.  



* Warstwy Feed-Forward: Gęste warstwy rozszerzające z nieliniowymi funkcjami aktywacji i połączeniami rezydualnymi (residual connections).  



* Głowica Generująca i Sampling: Projekcja liniowa do przestrzeni słownika (logity) z normalizacją Softmax dla próbkowania probabilistycznego. 





## Uruchomienie Projektu



1. Wymagania wstępne

Upewnij się, że masz zainstalowanego Pythona 3.10+ oraz aktywowane środowisko wirtualne: 



```bash

python -m venv venv

# W systemie Windows:

.\\venv\\Scripts\\activate

# W systemie Linux/macOS:

source venv/bin/activate



```



2. Instalacja zależności

Zainstaluj wymagane pakiety:



```bash

pip install -r requirements.txt



```



3. Trening bazowy (Pre-training)

Aby przetestować proces uczenia na przykładowych danych:



```bash

python train_full.py



```



4. Dalsze douczanie (Adaptacja dziedzinowa)


```bash

python train_adapt.py



```



5. Uruchomienie Aplikacji Desktopowej

Uruchomienie graficznego interfejsu użytkownika:



```bash

python aeris_gui.py



```



## Dalszy Rozwój (Roadmap)



* Pamięć Długoterminowa: Integracja z bazą wektorową (RAG) do trwałego zapamiętywania kontekstu między sesjami.



* Wykonywanie Narzędzi (Tool Calling): Dodanie możliwości uruchamiania zewnętrznych skryptów i zapytań API.



* Pętla Agentowa: Rozbudowa architektury w stronę autonomicznego agenta realizującego wieloetapowe zadania decyzyjne.


* Optymalizacja Pod Urządzenia Brzegowe (Edge AI): Kwantyzacja wag (int8 / int4) oraz eksport do formatu ONNX / TensorRT w celu redukcji narzutu pamięciowego.


* Integracja z Platformami Robotyki: Wykorzystanie zoptymalizowanego modelu jako lokalnego modułu decyzyjno-dialogowego na dedykowanym sprzęcie wbudowanym.

