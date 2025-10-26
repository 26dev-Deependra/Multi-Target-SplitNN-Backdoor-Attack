# Multi-Label Backdoor attack in SplitNN based Federated Learning.

This repo consisit of three main executables:

1. clean -> train_clean.py

   ```python
       python train_clean.py --clean-epoch 80 --dup 0 --multies 4 --unit 0.25
   ```

2. pois -> train_pois.py

   ```python
   python train_pois.py --label 0 --dup 0 --magnification 6 --multies 4 --unit 0.25 --clean-epoch 80
   ```

3. eval -> eval.py

   ```python
   python eval.py --multies 4 --unit 0.25
   ```

i am going to provide a code base on paper titled as Backdoor-Attack-Against-Split-Neural-Network-Based-Vertical-Federated-Learning in python following code consist of seven files .py files i want the code to be extended for multi-label attack
