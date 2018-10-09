# Confronto di classificatori per il Dataset WDBC
## Elaborato per il corso di Analisi Dati
### Miriam Santoro
#### aa. 2017/2018, Università di Bologna - Corso di laurea magistrale in Fisica Applicata

## Scopo del progetto
Il presente lavoro è stato realizzato in linguaggio `Python3` e si pone come obiettivo l'implementazione e la valutazione di:
1. 10 classificatori creati utilizzando le funzioni della libreria `sklearn`,
2. un classificatore basato su una rete neurale costruito tramite la libreria `pytorch`.

Il progetto ha previsto anche l'implementazioni di altre librerie, oltre quelle già menzionate, quali:
1. `numpy`
2. `pandas`
3. `matplotlib`
4. `scipy`
5. `graphviz`

## Esecuzione del progetto
I classificatori sono stati implementati in due script diversi a seconda della libreria utilizzata. Nello specifico:
1. `Classification.py` contiene i classificatori definiti usando `sklearn`; nello specifico:
    1. 10 classificatori implementati con possibilità di scegliere le features;
    2. Gli stessi 10 classificatori implementati tenendo conto delle 10 features migliori.
2. `NNC.py` contiene il classificatore definito usando Pytorch; nello specifico:
    1. Il classificatore implementato con possibilità di scegliere le features
    2. Lo stesso classificatore implementato tenendo conto delle 10 features migliori 
   
Questi due file Python sono stati implementati in `main.py`, script che è necessario eseguire per ottenere i risultati.
Inoltre, in `main.py` è stato importato anche lo script `Plotting.py`, utile per visualizzare i vari plot che verranno descritti nell'apposita sezione **Plotting**.

## Dataset (Note preliminari)
Il WDBC *Wisconsis Diagnostic Breast Cancer* è un dataset contenente 569 istanze, corrispondenti a 569 pazienti, ognuno dei quali classificato tramite un Id-paziente, un lettera indicante il tipo di tumore (maligno o benigno) e 30 features.
In particolare, le features legate al tumore indicano, in ordine:
1. Raggio (media delle distanze dal centro ai punti sul perimetro)
2. Texture (deviazione standard dei valori in scala di grigi)
3. Perimetro 
4. Area
5. Smoothness (variazione locale nelle lunghezze del raggio)
6. Compattezza (perimetro^2/area -1.0)
7. Concavità (gravità delle porzioni concave del contorno)
8. Punti concavi (numeri di porzioni concave del contorno)
9. Simmetria
10. Dimensioni frattali ("approssimazione coastline" -1)

Le features sono state calcolate da un'immagine digitalizzata di un fine aspiratore (FNA) di una massa del seno e descrivono le caratteristiche dei nuclei cellulari nell'immagine. Nello specifico per ogni immagine si hanno 30 features corrispondenti alla media, all'errore stardard e alla "peggiore" o più grande di ogni misura nell'elenco puntato e ogni cifra è stata acquisita con 4 digits. 

Inoltre, è necessario aggiungere che le 569 istanze possono essere divise in due classi, la prima avente 357 tumori benigni e la seconda avente 212 tumori maligni.


Una volta estratti i dati tramite la libreria `pandas` e aver disposto features e labels in array si è utilizzata la funzione `model_selection.train_test_split` contenuta in `sklearn` per splittare in maniera diversa il dataset in training set e test set. Questi ultimi sono stati divisi come segue:
1. 90% training, 10% test
2. 80% training, 20% test
3. 50% training, 50% test 
4. 25% training, 75% test

Le divisioni sono state pensate in modo da evitare l'overfitting e valutare in varie condizioni le performance di tutti i classificatori utilizzati.
Le diverse divisioni sono state implementate tramite un ciclo for all'interno di tutti i classificatori, come è mostrato nelle seguenti righe di codice:

```python
seq = [.9, .8, .5, .25]
for i in seq:
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=i, test_size= 1-i, 
                                                                        random_state=0)
```   

Per quanto riguarda il classificatore implementato tramite la libreria `pytorch` è stato necessario sfuttare il one hot encoding, un processo tramite cui è possibile convertire le labels in numeri binari. In questo caso, quindi, sono stati associati $[1,0]$ e $[0,1]$ alle due possibili uscite (rispettivamente $B$ e $M$) e questo passaggio è risultato indispensabile in quanto `pytorch` è in grado di funzionare solo su dati numerici.

## Analisi dei classificatori utilizzati
Si analizzano e commentano di seguito i classificatori utilizzati in questo progetto.

### Classificatori da `sklearn`
I classificatori implementati grazie `sklearn` sono i seguenti:
1. LogReg (*Logistic Regression classifier*)
2. SVM (*Support Vector Machine classifier*)
3. DTC (*Decision Tree Classifier*)
4. KNC (*K Neighbor Classifier*)
5. RFC (*Random Forest Classifier*)
6. MLP (*Multi Layer Perceptron classifier*)
7. ABC (*Ada Boost Classifier*)
8. GNB (*Gaussian Naive Bayes classifier*)
9. QDA (*Quadratic Discriminant Analysis classifier*)
10. SGD (*Stochastic Gradient Descent classifier*)

In un primo momento, ciascuno di questi classificatori è stato eseguito e valutato al variare del numero di features. Nello specifico sono state prese in considerazione le seguenti casistiche:
1. 1 features 
2. 9 features
3. 16 features
4. 30 features

Per fare ciò è stato solo necessario cambiare i parametri d'ingresso alle funzioni utilizzate per ciascun classificatore, come è possibile notare nel seguente esempio di codice:
```python
#da main.py
l1 = Classification.LogReg(2,3)
m1 = Classification.LogReg(2,9)
n1 = Classification.LogReg(2,16)
o1 = Classification.LogReg(2,30)
```

Inoltre, dopo che, tramite la funzione `stats.spearmanr`, implementata all'interno di `Histo` in `Plotting.py`, sono state individuate le 10 features più correlate alle labels, sono stati realizzati classificatori uguali a quelli precedenti ma in modo che prendessero come input queste 10 specifiche features. 
Per fare ciò, si è aggiunto un 10 al nome delle funzioni precedenti usate per i classificatori e non sono stati forniti parametri di input nell'argomento della funzione.
La chiamata a uno di questi classificatori è mostrata nella seguente riga di codice, tratta dal main:
```python
p1 = Classification.LogReg10()
```

#### 1. LogReg
Si usa l'implementazione standard del classificatore Logistic Regression contenuto nella libreria `sklearn.linear_model`. Questo è usato per valutare i parametri di un modello statistico al fine di modellare una variabile dipendente binaria, ovvero una variabile con due possibili valori (nel nostro caso etichettati con "B"=0 e "M"=1).
La funzione utilizzata in questo modello è chiamata logistica ed è una funzione del tipo:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)={\frac&space;{L}{1&plus;e^{-k(x-x_{0})}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)={\frac&space;{L}{1&plus;e^{-k(x-x_{0})}}}" title="f(x)={\frac {L}{1+e^{-k(x-x_{0})}}}" /></a>

dove:
- e = numero di Eulero,
- x0 = valore sull'asse delle x del punto a metà della funzione sigmoide,
- L =  massimo valore della curva
- k = ripidezza della curva.

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl = linear_model.LogisticRegression(C=2.5)
```
si sono usati:
- *C=2.5* come inverso della forza di regolarizzazione, indicante la tolleranza di casi mal classificati, in quanto questo valore è poco al di sopra del valore di default 1 ed è utile per evitare errori di overfitting;
- altri parametri lasciati di default tra cui:
    - *solver='liblinear'* come risolutore, in quanto per piccoli dataset è una buona scelta
    - *penalty='l2'* come termine di penalizzazione
    
In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

























