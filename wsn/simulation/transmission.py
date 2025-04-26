
from config import PACKET_SIZE

def transmit(chain, sink):
    energy_used = 0
    for i in range(len(chain) - 1):
        tx = chain[i]
        rx = chain[i + 1]
        energy_used += tx.consume_tx(PACKET_SIZE, tx.distance(rx))
        energy_used += rx.consume_rx(PACKET_SIZE)

    # Last node sends to sink
    last = chain[-1]
    energy_used += last.consume_tx(PACKET_SIZE, last.distance(sink))
    return energy_used