# Green AI Atividade 3: Hyperdimensional Classification

Este repositório refere-se ao código da disciplina Green AI, para a resolução da atividade 3.

O código é um exemplo de classificação utilizando o paradigma de Hyperdimensional Computing usando o modelo Centróide.

Para rodar o código certifique-se que está na pasta base do projeto (hdc_grai por padrão) e execute o comando:

`python src/main.py --dataset <nome-do-dataset> --encoding <record ou ngrams>`

Para o argumento dataset são suportados os datasets:
 - glass
 - hepatitis
 - abalone
 - abalone_stratified

Para a opção encoding são suportadas as opções:
 - record: Usa-se o record-based encoding para codificar os hipervetores
 - ngrams: Usa-se o ngrams-based encoding para codificar os hipervetores

O programa salva a acuracia na tabela "results.csv", caso o arquivo não exista, ele é criado.

Já disponível nessa versão há experimentos preliminares com os datasets glass, hepatitis e abalone, comparando a acurácia para cada opção de encoding.