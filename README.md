# BetterRecommendationProject
### algorithms and experiment for recommendation
Notes:
1. Please download the data yoochoose-clicks.dat, etc into the data folder
2. For session model:\
    a. run the data preprocessing script to generate the train, validation, test file

experiment data:
           train     validation  test
events:    582314    58162       71174
session    143185    12368       15314
Item:      17667     6314        6314


performance:
1. GRU-hidden size: 50, batch size: 50, GRU layers: 3, dropout 0.5, lr = 0.01, epochs = 40
    => mean_losses: 1.00381, mean_recall: 0.00017969, mean_mrr: 4.743145*10-5
2. GRU-hidden size: 100, batch size: 50, GRU layers: 3, dropout 0.5, lr = 0.01, epochs = 40
    => mean_losses: 1.00366, mean_reacall: 0.0001078, mean_mrr: 1.511*10-5
3. GRU-hidden size: 100, batch size: 50, GRU layers: 4, dropout 0.5, lr = 0.01, epochs = 40
    => mean_losses: 1.006524, mean_recall: 8.984*10-5, mean_mrr: 1.0393*10-5

