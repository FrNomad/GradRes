import sys
import time
import random
import numpy as np
from time_util import *

def progressBar(size, progress) :
    bar_num = int(size*progress)
    return "[%s%s]"%('-'*bar_num,' '*(size-bar_num))

epoches = 20
loss_total = 0
for epoch in range(1, epoches + 1) :

    total_cycle = 500
    epoch_loss = 0
    for cycle in range(1, total_cycle + 1) :
        
        if cycle > 1 :
            sys.stdout.write('\r')
            sys.stdout.write(' '*120)
            sys.stdout.write('\r')
        total_progress = epoch/epoches
        epoch_progress = cycle/total_cycle
        time_elapsed = random.randint(1, 100)
        sys.stdout.flush()
        sys.stdout.write("Epoch %d (%d%%) : %s [%d/%d] (%d%%) %s" % (epoch, total_progress * 100, progressBar(25, epoch_progress), 
                                                                    cycle, total_cycle, epoch_progress * 100, asMinutes(time_elapsed)))                             
        if cycle < total_cycle :
            try :
                sys.stdout.write(" (total exp : %s)" % asMinutes(time_elapsed*epoches/(epoch-1+(cycle-1)/total_cycle)))
            except :
                sys.stdout.write(" (total exp : %s)" % "inf")

    print(' -- loss: %.4f' % (epoch_loss/total_cycle))