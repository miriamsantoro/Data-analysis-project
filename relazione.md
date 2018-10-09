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

#### 2. SVM
Si usa l'implementazione standard della Support Vector Classification contenuta nella libreria `sklearn.svm`. Questo classificatore è detto anche classificatore a massimo margine perchè allo stesso tempo minimizza l'errore empirico di classificazione e massimizza il margine geometrico. Nello specifico, cerca gli iperpiani di separazioni ottimali tramite una funzione obiettivo senza minimi locali.

Tra i vantaggi nell'uso del SVC possiamo trovare il fatto che:
- sia efficiente dal punto di vista della memoria in quanto usa un subset di punti di training nella funzione di decisione (chiamati support vectors) 
- sia versatile poichè per come funzioni di decisione si possono specificare diverse funzioni Kernel. Queste permettono di proiettare il problema iniziale su uno spazio di dimensioni superiore senza grandi costi computazionali e ottenendo separazioni basate anche su superfici non lineari.

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl=svm.SVC(kernel='linear')    
```
si sono usati:
- come funzione kernel, una funzione lineare del tipo:
<a href="https://www.codecogs.com/eqnedit.php?latex=\langle&space;x,&space;x'\rangle" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\langle&space;x,&space;x'\rangle" title="\langle x, x'\rangle" /></a>
- altri parametri di default, tra cui:
   - *C=1.0* come parametro di penalità per il termine di errore

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 3. DTC
Si usa l'implementazione standard del Decision Tree Classifier contenuto nella libreria `sklearn.tree`. Questo classificatore è usato per valutare i parametri di un modello al fine di predire il valore di un target variabile, apprendendo semplici regole di decisione dedotte dalle features.

Tra i vantaggi di questo classificatore è possible trovare il fatto che:
- sia semplice da capire ed interpretare in quanto si basa su un modello white box in cui l'interpretazione di una condizione è facilmente spiegata dalla logica booleana
- gli alberi (*trees*) possano essere visualizzati
- il costo di predire i dati usando l'albero sia logaritmico nel numero di punti di dati usati per allenare l'albero stesso.


Nello specifico, per quanto riguarda il classificatore implementato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl = tree.DecisionTreeClassifier()      
```
si sono usati i parametri di default, tra cui:
- *criterion='gini'* come funzione per misurare la qualità dello split; 
- *splitter='best'* come strategia usata per scegliere lo split ad ogni nodo in modo da scegliere quello migliore;
- *max_features=None* come numero di features da considerare quando si guarda allo split migliore. In questo caso max_features=n_features;
- *max_depth=None* come massima profondità dell'albero None. Questo significa che i nodi vengono estesi fin quando tutte le foglie sono pure.

Inoltre, l'equazione di diminuzione di impurità pesata (che governa lo split) è:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{N_t}{N}&space;\cdot&space;\left(\text{impurity}&space;-&space;\frac{N_{t_R}}{N_t}&space;\cdot&space;\text{(right-impurity)}&space;-&space;\frac{N_{t_L}}{N_t}&space;\cdot&space;\text{(left-impurity)}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{N_t}{N}&space;\cdot&space;\left(\text{impurity}&space;-&space;\frac{N_{t_R}}{N_t}&space;\cdot&space;\text{(right-impurity)}&space;-&space;\frac{N_{t_L}}{N_t}&space;\cdot&space;\text{(left-impurity)}&space;\right)" title="\frac{N_t}{N} \cdot \left(\text{impurity} - \frac{N_{t_R}}{N_t} \cdot \text{(right-impurity)} - \frac{N_{t_L}}{N_t} \cdot \text{(left-impurity)} \right)" /></a>

dove:
- *N* è il numero totale di campioni
- *N_t* è il numero di campioni al nodo corrente
- *N_{t_L}* è il numero di campioni nella parte destra
- *N_{t_R}* è il numero di campioni nella parte sinistra

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 4. KNC
Si usa l'implementazione standard del K Neighbors Classifier contenuto nella libreria `sklearn.neighbors`. Questo classificatore non cerca di costruire un modello interno generale, ma semplicemente memorizza le istanze dei dati di training; quindi la classificazione è calcolata da un semplice voto di maggioranza dei vicini più vicini ad ogni punto: il punto di ricerca è assegnato alla classe che ha il maggior numero di rappresentanti nei vicini più vicini del punto.

Questo classificatore, quindi, si basa su k vicini, dove k è un valore intero specificato dall'utente e la sua scelta ottimale dipende fortemente dai dati. Ad esempio, in generale, una k più grande sopprime gli effetti del rumore ma rende i confini di classificazione meno distinti.

Nello specifico, per quanto riguarda il classificatore implementato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl = neighbors.KNeighborsClassifier(n_neighbors=3)       
```
si sono usati:
- *n_neighbors=3* come numero di vicini;
- altri parametri di default tra cui:
    - *weights='uniform'* come funzione per i pesi usata per la previsione. I pesi uniformi portano a pesare equamente tutti i punti in ogni vicinato
    - *algorithm='auto'* come algoritmo usato per calcolare i vicini. Questo è l'algoritmo più appropriato sulla base dei valori passati dal metodo di fit.
    - *metric='minkowski'* come metrica ovvero distanza usata per l'albero.
    
In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 5. RFC 
Si usa l'implementazione standard del Random Forest Classifier contenuto nella libreria `sklearn.ensemble`. Questo classificare fa il fit di un numero di classificatori decision trees su vari sotto-campioni del dataset e usa la media per migliorare l'accuratezza predittiva e controllare l'over-fitting. La grandezza dei sotto campioni è uguale a quella del campione di input iniziale ma i campioni sono disegnati con rimpiazzamento dal set di training. 
Inoltre, quando si fa la divisione di un nodo durante la costruzione dell'albero, lo split che viene scelto non è il migliore tra tutte le features ma tra un subset random di features. A causa di questa randomicità, il bias della foresta dovrebbe aumentare (rispetto a quello di un singolo albero non-random) ma, grazie alla media, la sua varianza diminuisce e compensa di più l'aumento del bias determinando un modello migliore.

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl=ensemble.RandomForestClassifier(max_depth=15, n_estimators=10, max_features=1)      
```
si sono usati:
- *max_depth=15* come massima espansione dell'albero;
- *n_estimators=10* come numero di alberi (estimatori) nella foresta;
- *max_features=1* come numero di features da considerare quando si guarda al migliore split;
- altri parametri di default, tra cui:
    - *criterion='gini'* come funzione per misurare la qualità dello split;
    - $*min_samples_split=2* come numero minimo di campioni richiesto per dividere un nodo interno.

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 6. MLP
Si usa l'implementazione standard del Multi Layer Perceptron contenuto nella libreria `sklearn.neural_network`. Questo classificatore si basa su una rete neurale e su un allenamento iterativo. Infatti, ad ogni step temporale vengono calcolate le derivate parziali della funzione di perdita rispetto ai parametri del modello per aggiornare i parametri stessi.
Può avere anche un termine di regolarizzazione aggiunto alla funzione di perdita che restringe i parametri del modello per prevenire l'overfitting.

Nello specifico, per quanto riguarda il classificatore utilizzato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl = neural_network.MLPClassifier(activation='logistic', solver='lbfgs', max_iter=1000 )       
```
si sono usati:
- *activation='logistic'* come funzione di attivazione per gli strati nascosti. Questa è una funzione logistica sigmoidale che restituisce:
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&plus;exp(-x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;exp(-x)}" title="f(x) = \frac{1}{1+exp(-x)}" /></a>
   
- *solver='lbfgs'* come risolutore per l'ottimizzazione dei pesi. Questo è un ottimizzatore facente parte dei metodi quasi-newtoniani ed è stato scelto in quanto per piccoli datasets, come il wdbc, converge più velocemente e ha migliori performance;
- *max_iter=1000* come massimo numero di iterazioni per la convergenza;
- altri parametri di default, tra cui:
    - *hidden_layer_sizes=(100,)}* come dimensioni degli strati nascosti;
    - *alpha=0.0001* come parametro di penalizzazione L2;
    - *batch_size=min(200,n_samples)}* come grandezza di batch;
    - *learning_rate=costant=0.001* come frequenza di apprendimento per l'aggiornamento dei pesi.
 
In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 7. ABC
Si usa l'implementazione standard dell'Ada Boost Classifier contenuto nella libreria `sklearn.ensemble`. Questo classificatore inizialmente fa il fit di un modello debole (leggermente migliori di quelli random) sul dataset e poi fitta copie aggiuntive dello stesso modello sullo stesso dataset ma aggiustando i pesi di istanze di training classificate non correttamente in modo che i classificatori seguenti si focalizzino di più su casi difficili. 

Iniziamente, questi pesi sono tutti settati a 1/N, così che il primo step alleni semplicemente un debole modello dei dati originali; tuttavia, per ogni iterazione successiva, sono modificati individualmente in modo che:
- i pesi associati agli esempi di training che non sono predetti correttamente vengano aumentati;
- i pesi associati agli esempi di training che sono predetti correttamente vengano diminuiti.

Successivamente l'algoritmo di apprendimento viene riapplicato per ripesare i dati. In questo modo ogni volta che le iterazioni procedono, gli esempi difficili da predire ricevono un'influenza sempre crescente e ogni modello debole viene forzato a concentrarsi sugli esempi che mancano dai precedenti nella sequenza.
Infine, le previsioni da ciascuno di loro vengono combinate attraverso un voto di maggioranza pesata per produrre la previsione finale. 

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl=ensemble.AdaBoostClassifier()       
```
si sono usati i parametri di default, tra cui:
- *base_estimator=None=DecisionTreeClassifier(max_depth=1)}* come estimatore base da cui è costruito l'ensemble potenziato;
- *n_estimators=50* come numero di estimatori a cui viene concluso il potenziamento;
- *learning_rate=1* come frequenza di apprendimento;
- *algorithm='SAMME.R'* come algoritmo di potenziamento. Questo converge velocemente, raggiungendo un errore minore di testing con minori iterazioni di boosting.

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 8. GNB
Si usa l'implementazione standard del Gaussian Naive Bayes contenuto nella libreria `sklearn.naive_bayes`. Questo classificatore si basa sull'applicazione del teorema di Bayes, con l'assunzione di una forte (naive) indipendenza condizionata tra ogni coppia di features, dato il valore della variabile di classe.
Il teorema di Bayes stabilisce che, data la variabile di classe $y$ e il vettore di feature dipendente $x_1$ attraverso $x_n$, si ha:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y&space;\mid&space;x_1,&space;\dots,&space;x_n)&space;=&space;\frac{P(y)&space;P(x_1,&space;\dots&space;x_n&space;\mid&space;y)}&space;{P(x_1,&space;\dots,&space;x_n)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y&space;\mid&space;x_1,&space;\dots,&space;x_n)&space;=&space;\frac{P(y)&space;P(x_1,&space;\dots&space;x_n&space;\mid&space;y)}&space;{P(x_1,&space;\dots,&space;x_n)}" title="P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)} {P(x_1, \dots, x_n)}" /></a>

L'assunzione di indipendenza naive condizionale, invece, stabilisce che:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(x_i&space;|&space;y,&space;x_1,&space;\dots,&space;x_{i-1},&space;x_{i&plus;1},&space;\dots,&space;x_n)&space;=&space;P(x_i&space;|&space;y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_i&space;|&space;y,&space;x_1,&space;\dots,&space;x_{i-1},&space;x_{i&plus;1},&space;\dots,&space;x_n)&space;=&space;P(x_i&space;|&space;y)" title="P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y)" /></a>

Questo classificatore implementa l'algoritmo Gaussian Naive Bayes per la classificazione in cui si assume che la likelihood delle features sia gaussiana:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(x_i&space;\mid&space;y)&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2_y}}&space;\exp\left(-\frac{(x_i&space;-&space;\mu_y)^2}{2\sigma^2_y}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_i&space;\mid&space;y)&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2_y}}&space;\exp\left(-\frac{(x_i&space;-&space;\mu_y)^2}{2\sigma^2_y}\right)" title="P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)" /></a>

dove i parametri <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_y" title="\sigma_y" /></a> e <a href="https://www.codecogs.com/eqnedit.php?latex=\mu_y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu_y" title="\mu_y" /></a> sono stimati usando il massimo della likelihood.

Tra i vantaggi di questo classificatore troviamo il fatto che:
- richieda una piccola quantità di dati di training per stimare i parametri necessari;
- possa essere estremamente veloce se confrontato con metodi più sofisticati;
- il disaccoppiamento delle distribuzioni di features condizionali delle classi possa essere stimato indipendentemente come una distribuzione unidimensionale e questo aiuta ad ridurre i problemi che derivano dalla 'maledizione' della dimensionalità.


Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl=naive_bayes.GaussianNB()    
```
in cui si sono utilizzati i parametri di default, ovvero:
- *priors=(n_classes,)}* come probabilità a priori delle classi;
- *var_smoothing=1e-9* come porzione della varianza più grande tra tutte le features. Questa viene aggiunta alle varianze per il calcolo della stabilità.

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 9. QDA
Si usa l'implementazione standard della Quadratic Discriminant Analysis contenuta nella libreria `sklearn.discriminant_analysis`. Questo classificatore può essere ottenuto da semplici modelli probabilistici che modellano la distribuzione condizionale di classe dei dati $P(X|y=k)$ per ogni classe $k$. 
Le previsioni possono essere ottenute usando la regola di Bayes:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(y=k&space;|&space;X)&space;=&space;\frac{P(X&space;|&space;y=k)&space;P(y=k)}{P(X)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y=k&space;|&space;X)&space;=&space;\frac{P(X&space;|&space;y=k)&space;P(y=k)}{P(X)}" title="P(y=k | X) = \frac{P(X | y=k) P(y=k)}{P(X)}" /></a>

Nello specifico, $P(X|y)$ è modellizzato come una distribuzione Gaussiana multivariata con densità:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(X&space;|&space;y=k)&space;=&space;\frac{1}{(2\pi)^{d/2}&space;|\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2}&space;(X-\mu_k)^t&space;\Sigma_k^{-1}&space;(X-\mu_k)\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(X&space;|&space;y=k)&space;=&space;\frac{1}{(2\pi)^{d/2}&space;|\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2}&space;(X-\mu_k)^t&space;\Sigma_k^{-1}&space;(X-\mu_k)\right)" title="P(X | y=k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2} (X-\mu_k)^t \Sigma_k^{-1} (X-\mu_k)\right)" /></a>

dove *d* è il numero di features.

In poche parole, questo è un classificatore con un limite di decisione (superficie di separazione) quadratico che è generato fittando le densità condizionali delle classi e usando la regola di Bayes. Il modello fitta una densità Gaussiana ad ogni classe.

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl=discriminant_analysis.QuadraticDiscriminantAnalysis()      
```
sono stati usati come parametri quelli di default, tra cui:
- *priors=n_classes* come priori sulle classi;
- *tol=1.0e-4* come soglia usata per la stima del rango.

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.

#### 10. SGD
Si usa l'implementazione standard della Stochastic Gradient Descent contenuta nella libreria `sklearn.linear_model`. Questo è un classificatore lineare con apprendimento tramite il gradiente di discesa stocastica (SGD); nello specifico, questo implica che per ogni campione:
- viene stimato il gradiente di perdita;
- viene aggiornato il modello man mano con una frequenza di apprendimento decrescente.

Il regolarizzatore è un termine di penalità che viene aggiunto alla funzione di perdita e che restringe i parametri del modello.

Nello specifico, per quanto riguarda il classificatore usato in questo progetto e mostrato di seguito:
```python
#da Classification.py
cl = linear_model.SGDClassifier(loss="perceptron", penalty="elasticnet", max_iter=600)   
```
si sono usati:
- *loss='perceptron'* come funzione di perdita, ovvero come perdita lineare usata dall'algoritmo del percettrone;
- *penalty='elasticnet'* come penalità (termine di regolarizzazione). Questo è un termine che viene aggiunto alla funzione di perdita, restringe i parametri del modello e, in questo caso, è una combinazione di L2 (norma euclidea quadrata) e L1 (norma assoluta);
- *max_iter=600* come massimo numero di passi sui dati di training;
- altri parametri di default, tra cui:
    - *tol=1e-3* come criterio di stop.

In seguito all'addestramento del classificatore tramite la funzione `fit`, viene utilizzata la funzione `predict` per fare previsioni sui dati di test e vengono calcolati rispettivamente report di classificazione, matrice di confusione e accuratezza.























