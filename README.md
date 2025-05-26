# WSN Routing Algorithms: A*, Dijkstra, SPIN, and GPSR Comparison

This project benchmarks and visualizes the performance of four routing algorithms—**A***, **Dijkstra**, **SPIN**, and **GPSR**—for temperature and humidity monitoring in Wireless Sensor Networks (WSNs) with environmental penalties.

---

## Features

- **Simulates a WSN** (grid topology, random long-range links, harsh environmental zones)
- **Implements and compares:**
  - **A*** (heuristic, energy/environmentally-aware routing)
  - **Dijkstra** (classic shortest-path)
  - **SPIN** (data-centric, flooding-based)
  - **GPSR** (geographic greedy routing)
- **Benchmarks key metrics:**
  - Latency (ms)
  - Path Length (hops)
  - Nodes Explored
  - Harsh Nodes Traversed
  - Path Cost (energy/environmental penalty)
- **Scalability analysis** for different network sizes
- **Bar charts and scalability plots** for visual comparison

---

## How It Works

- The WSN is modeled as a grid with environmental penalties for high temperature and humidity.
- Each algorithm finds a route from source to target, optimizing for its own metric.
- The code benchmarks each algorithm over multiple iterations for accuracy.
- Results are displayed as tables and graphs.

---

## Usage

1. **Install requirements** (if any):

```bash
pip install networkx matplotlib numpy
```

2. **Run the script**:

```bash
python your_script_name.py
```

3. **View the output:**
   - Detailed comparison tables in the terminal.
   - Bar charts and scalability line plots (see below).

---

## Output Examples

### Comparison Table

```
Final WSN Routing Algorithm Comparison
=================================================================
Metric               A*              Dijkstra        SPIN            GPSR            Improvement (A* vs others)
-----------------------------------------------------------------
time                 0.47            0.61            0.03            0.05            ...
path_length          11              9               7               8               ...
nodes_explored       40              87              85              70              ...
harsh_nodes          0               0               0               0               ...
path_cost            120.00          150.00          410.00          130.00          ...
```

### Key Insights

- **A*** explores far fewer nodes and finds much lower-cost (energy-efficient) paths.
- **Dijkstra** finds the shortest path but ignores environmental cost.
- **SPIN** may have lower latency in small grids, but is less efficient in large/harsh environments.
- **GPSR** (once implemented) will be compared on similar metrics.

---

### Visualizations

**Scalability Plots**

**Routing Comparison Bar Chart**

---

## Metrics Explained

- **Latency (ms):** Average time to find a route.
- **Path Length:** Number of hops from source to target.
- **Nodes Explored:** Number of nodes visited during routing.
- **Harsh Nodes:** Nodes in high temperature/humidity zones traversed.
- **Path Cost:** Sum of edge weights (energy/environmental penalty).

```---

## License

This project is for academic and research use. Please acknowledge the authors if used in publications.

---

**Contact:**
For questions or collaboration, please contact [Your Name] at [Your Email].

---

**Enjoy comparing A*, Dijkstra, SPIN, and GPSR for WSN routing!**


```
