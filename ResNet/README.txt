Per avviare i test ResNet:
    Single-Social:
        python test.py --experiment="Single" --fold=0 --social="Twitter"
        python test.py --experiment="Single" --fold=0 --social="Facebook"
        python test.py --experiment="Single" --fold=0 --social="Instagram"
    
    Multi-Social:
        python test.py --experiment="Multi" --fold=0
    
    Cross-Social:
        python test.py --experiment="Cross" --social="Twitter"
        python test.py --experiment="Cross" --social="Facebook"
        python test.py --experiment="Cross" --social="Instagram"
      
