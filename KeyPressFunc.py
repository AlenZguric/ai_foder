import pygame

def init():
    pygame.init()
    #Postavljanje prozora  

def getKey(keyName):

    #Postavljanje varijable koja označava da tipka nije pritisnuta
    answer = False

    # Ovdje se koristi pass jer samo želimo obraditi događaje, a ne raditi ništa s njima
    for eve in pygame.event.get():pass

    # Dohvaćanje svih tipki koje su trenutno pritisnute
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    if keyInput[myKey]:
        answer = True
    pygame.display.update()
    

    return answer



if __name__ == '__main__' :
    init()   