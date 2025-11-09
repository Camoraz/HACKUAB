# Repte DELOITTE: Disseny de noves línies de metro a Barcelona

Aquest projecte forma part del repte Deloitte **DataRide**, amb l’objectiu de dissenyar noves línies de metro a Barcelona utilitzant dades obertes, anàlisi de dades i tècniques d’optimització.

## Contingut del projecte

1. **Recopilació i preprocesament de dades**  
   Es recullen dades obertes de l’Ajuntament de Barcelona, TMB, ATM, INE/IDESCAT i OpenStreetMap per crear una base de dades unificada d’estacions, barris i fluxos de mobilitat.

2. **Anàlisi exploratori de dades (EDA)**  
   Es calcula la **puntuació esperada de recorregut** entre barris utilitzant A* i \emph{sampling}, considerant els desplaçaments a peu i la població de cada barri. Això permet identificar zones amb deficiències i oportunitats d’expansió.

3. **Disseny de línies de metro amb algorisme genètic**  
   - Representació de les línies: primera estació absoluta, següents com desplaçaments relatius.  
   - Funció de fitness combinant: temps esperat de recorregut, rectitud, homogeneïtat de separació i longitud total.  
   - Operadors de creuament i mutació amb soroll gaussià i \emph{annealing}.  
   - Generació de línies òptimes segons criteris combinats.

4. **Predicció de demanda i xatbot**  
   - Model de sèries temporals **BATS** per predir la demanda futura per estació.  
   - Xatbot en català amb pipeline: preprocessament → NER → reconeixement d’intencions → gestor de diàleg → generació de resposta.  
   - Fallback amb **AIna** per consultes complexes.

## Resultats

- Anàlisi i millora del temps de recorregut mitjà entre barris.
- Estacions noves generades amb algoritmes genètics que equilibren cobertura i eficiència.
- Prediccions precises i XatBot.

## Requisits

- Python 3.8+  
- Biblioteques: `numpy`, `pandas`, `matplotlib`, `scipy`, `networkx`, `spacy` (opcional: `flair`)  

