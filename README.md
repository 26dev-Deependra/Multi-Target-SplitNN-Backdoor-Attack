# Multi-Label Backdoor attack in SplitNN based Federated Learning.

This repo consisit of three main executables:

1. clean -> train_clean.py

   ````python
        python train_clean.py --clean-epoch 80 --dup 0 --multies 4 --unit 0.25
         ```

   ````

2. pois -> train_pois.py

   ```python
   python train_pois.py --label 0 --dup 0 --magnification 6 --multies 4 --unit 0.25 --clean-epoch 80
   ```

3. eval -> eval.py

   ````python
   python eval.py --multies 4 --unit 0.25```
   ````
