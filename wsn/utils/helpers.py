
def total_chain_distance(chain):
    return sum(chain[i].distance(chain[i+1]) for i in range(len(chain)-1))